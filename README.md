# OpenLLM

A GPT-2 inference engine written from scratch in plain Java: no ML libraries, no native code, no Python at runtime. It loads OpenAI's original GPT-2 weights straight from a `.safetensors` file, tokenizes your prompt with its own byte-level BPE implementation, runs the transformer on the CPU, and generates text.

OpenLLM is a learning project. The goal is an engine where every stage (tokens, embeddings, Q/K/V, attention, the residual stream, logits, the KV cache) is ordinary, readable Java you can step through in a debugger.

```
> .\gradlew run --args="-t 0 Once upon a time"
Welcome to openLLM!
Model: SMALL

Once upon a time, the world was a place of great beauty and great danger. ...
Runtime: 5.29s
Tokens/Sec: 56.8t/s
```

## Features

- **Tensor library**: row-major `float[]` storage with shapes and strides, views (`reshape`, `transpose`), `contiguous()`, and `allClose` for testing.
- **CPU backend**: `matmul`, `add`, `addBias`, `scale`, GELU (tanh approximation), numerically stable `softmax`, and `layerNorm`, all behind a `Backend` interface so another backend can slot in later.
- **Multithreaded matmul**: the prompt is split across threads by rows; each new token uses split-K, where every thread multiplies its own contiguous block of weight rows and the partial results are summed.
- **Safetensors reader**: parses the JSON header and loads each tensor directly from the file.
- **GPT-2 tokenizer**: byte-to-unicode mapping, GPT-2's pre-split regex, BPE merges, `encode` and `decode`.
- **GPT-2 model**: token and position embeddings, multi-head causal self-attention, MLP, pre-norm residual blocks, final layer norm, and a tied output projection.
- **KV cache**: the prompt is processed once (prefill), then each new token runs through the model alone. Attention reads keys and values straight from the cache, one thread per head, with no copies.
- **Sampling**: temperature and top-k, with greedy decoding when the temperature is 0 or k is 1.
- **Command-line interface** built with picocli: model size, temperature and top-k flags, plus `--help`.
- **All four GPT-2 sizes** (small, medium, large, xl).
- **Tests checked against a reference**: attention, MLP, full blocks and final logits are compared with values from a NumPy implementation of GPT-2.

## Requirements

- Java 25. The Gradle toolchain will use or download it.
- The GPT-2 weights (see below). They're not in the repo because of their size.
- Enough RAM for the model you pick. Small fits in about 1 GB of heap; xl needs about 9 GB.

## Getting the models

Every size shares the same tokenizer, so `vocab.json` and `merges.txt` only need to be downloaded once, into `gpt2-small/`. Each size needs its own `model.safetensors`. From the repo root, in PowerShell:

```powershell
mkdir models\gpt2-small
curl.exe -L -o models\gpt2-small\model.safetensors https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
curl.exe -L -o models\gpt2-small\vocab.json https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
curl.exe -L -o models\gpt2-small\merges.txt https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt

# Optional larger models: medium shown here; large and xl follow the same pattern
mkdir models\gpt2-medium
curl.exe -L -o models\gpt2-medium\model.safetensors https://huggingface.co/openai-community/gpt2-medium/resolve/main/model.safetensors
```

The resulting layout:

```
models/
  gpt2-small/    model.safetensors, vocab.json, merges.txt
  gpt2-medium/   model.safetensors
  gpt2-large/    model.safetensors
  gpt2-xl/       model.safetensors
```

| Model  | Parameters | Layers | Heads | Width | File size |
|--------|-----------:|-------:|------:|------:|----------:|
| small  | 124M       | 12     | 12    | 768   | 0.5 GB    |
| medium | 355M       | 24     | 16    | 1024  | 1.5 GB    |
| large  | 774M       | 36     | 20    | 1280  | 3.2 GB    |
| xl     | 1.5B       | 48     | 25    | 1600  | 6.4 GB    |

## Usage

```powershell
# GPT-2 small (the default), temperature 0.8, top-k 40
.\gradlew run --args="Once upon a time"

# Pick a size and sampling settings
.\gradlew run --args="-m medium -t 0.7 -k 50 Once upon a time"

# Greedy decoding: the same prompt always gives the same text
.\gradlew run --args="-t 0 Once upon a time"

# List every option
.\gradlew run --args="--help"
```

| Option | Default | Meaning |
|---|---|---|
| `-m`, `--model` | `small` | `small`, `medium`, `large` or `xl` |
| `-t`, `--temp` | `0.8` | Sampling temperature; `0` means greedy |
| `-k`, `--top-k` | `40` | Sample only from the k most likely tokens |

Options go before the prompt. Each run generates 300 tokens and prints the runtime and tokens per second.

On macOS or Linux, use `./gradlew` instead.

Run the tests with:

```powershell
.\gradlew test
```

The model tests need `models/gpt2-small/` in place.

## Performance

Measured on a laptop with an AMD Ryzen 9 7940HS (8 cores, 16 threads), 32 GB of RAM, Java 25 and Windows. Generation is greedy, so every run of the same prompt produces the same text.

### Single-threaded, 30 tokens

Generating 30 tokens from "Once upon a time" with GPT-2 small:

| Step | Time | Tokens/s |
|---|---:|---:|
| First working version | ~115 s | ~0.26 |
| i-k-j matmul loop order, no array copies on reads | 18.9 s | 1.6 |
| Compute logits for the last position only | ~12 s | 2.5 |
| KV cache | 1.36 s | ~22 |

### Multithreaded, 300 tokens

Generating 300 tokens from "The meaning of life is" with temperature 0:

| Step | Small | Medium |
|---|---:|---:|
| Parallel matmul (row split for the prompt, column split for each new token) | not measured | 38.0 s (7.9 tok/s) |
| Attention reads the KV cache directly, parallel over heads | 8.7 s (34 tok/s) | 19.2 s (15.6 tok/s) |
| Split-K matmul for each new token | **5.29 s (56.8 tok/s)** | **14.64 s (20.5 tok/s)** |

Generating each new token is limited by memory bandwidth: every weight is read once per token. Medium's 1.4 GB of fp32 weights at 20.5 tok/s is about 29 GB/s.

To reproduce a row:

```powershell
.\gradlew run --args="-m medium -t 0 The meaning of life is"
```

## How it works

1. **Tokenize**: the prompt is split with GPT-2's regex, mapped to byte-level unicode, and merged with BPE into token ids.
2. **Embed**: each id selects a row of the token embedding matrix `wte`, and its position adds a row of `wpe`.
3. **Transformer blocks** (12 to 48, depending on size). Each block applies `x = x + attention(layerNorm(x))`, then `x = x + mlp(layerNorm(x))`.
   - Attention projects to Q, K and V, splits into heads, computes `softmax(QKᵀ / √64)` with a causal mask, and multiplies by V.
   - The MLP expands to 4× width, applies GELU, and projects back down.
4. **Logits**: a final layer norm, then a multiply by `wteᵀ` (GPT-2 ties the input and output embeddings), gives one score for each of the 50,257 tokens.
5. **Generate**: divide the logits by the temperature, keep the top-k, apply softmax, draw a token, append it, and repeat. Each new token's keys and values are added to the KV cache, so earlier tokens are never recomputed.

## Project layout

```
app/src/main/java/openllm/
  tensors/    Tensor
  backend/    Backend interface, CpuBackend
  io/         SafetensorReader, TensorInfo
  tokenizer/  Gpt2Tokenizer
  model/      Gpt2Config, GptModel, Gpt2Weights, LayerWeights, Gpt2Model, KvCache, Qkv
  generate/   Generator, Sampler
  cli/        Main
```

## Roadmap

1. A model interface, so generation doesn't depend on GPT-2, plus streaming output
2. One config-driven decoder that reads a Hugging Face `config.json`
3. A modern model, Qwen3-0.6B (bf16, RMSNorm, RoPE, SwiGLU, grouped-query attention)
4. A chat harness: chat templates, stop tokens, a streaming REPL, top-p sampling
5. Speed for bigger models: Q8_0 quantization, SIMD via the Vector API, an fp16 KV cache
6. 4B to 8B models with memory-mapped and sharded weights
7. Tool calling and agents, with JSON-constrained decoding
8. An OpenAI-compatible server
9. Mixture-of-experts models and a GPU backend

## Acknowledgements

The model weights and tokenizer are OpenAI's GPT-2, downloaded from Hugging Face ([openai-community/gpt2](https://huggingface.co/openai-community/gpt2)).
