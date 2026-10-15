# OpenLLM

A GPT-2 inference engine written from scratch in plain Java: no ML libraries, no native code, no Python at runtime. It loads OpenAI's original GPT-2 weights straight from a `.safetensors` file, tokenizes your prompt with its own byte-level BPE implementation, runs the transformer on the CPU, and generates text.

OpenLLM is a learning project. The goal is an engine where every stage (tokens, embeddings, Q/K/V, attention, the residual stream, logits, the KV cache) is ordinary, readable Java you can step through in a debugger.

```
> .\gradlew run --args="Once upon a time"
Welcome to openLLM!
Model: SMALL

Once upon a time, the world was a place of great beauty and great danger. The world was a place of great danger, and the world was a place of great
Runtime: 1.36s
```

## Features

- **Tensor library**: row-major `float[]` storage with shapes and strides, views (`reshape`, `transpose`), `contiguous()`, and `allClose` for testing.
- **CPU backend**: `matmul` (cache-friendly i-k-j loop order), `add`, `addBias`, `scale`, GELU (tanh approximation), numerically stable `softmax`, and `layerNorm`, all behind a `Backend` interface so another backend can slot in later.
- **Safetensors reader**: parses the JSON header and loads each tensor directly from the file.
- **GPT-2 tokenizer**: byte-to-unicode mapping, GPT-2's pre-split regex, BPE merges, `encode` and `decode`.
- **GPT-2 model**: token and position embeddings, multi-head causal self-attention, MLP, pre-norm residual blocks, final layer norm, and a tied output projection.
- **KV cache**: the prompt is processed once (prefill), then each new token runs through the model alone and attends to cached keys and values.
- **Greedy generation** that stops at the end-of-text token or the context limit.
- **All four GPT-2 sizes** (small, medium, large, xl), chosen with a command-line flag.
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
# GPT-2 small (the default)
.\gradlew run --args="Once upon a time"

# Pick a size with --model=small|medium|large|xl, given before the prompt
.\gradlew run --args="--model=medium Once upon a time"
```

On macOS or Linux, use `./gradlew` instead.

Run the tests with:

```powershell
.\gradlew test
```

The model tests need `models/gpt2-small/` in place.

## Performance

Measured on a desktop CPU, single-threaded, generating 30 tokens from "Once upon a time" with GPT-2 small:

| Step | Time |
|---|---:|
| First working version | ~115 s |
| i-k-j matmul loop order, no array copies on reads | 18.9 s |
| Compute logits for the last position only | ~12 s |
| KV cache | **1.36 s** (~22 tokens/s) |

GPT-2 medium runs at about 10 tokens/s with the same setup.

## How it works

1. **Tokenize**: the prompt is split with GPT-2's regex, mapped to byte-level unicode, and merged with BPE into token ids.
2. **Embed**: each id selects a row of the token embedding matrix `wte`, and its position adds a row of `wpe`.
3. **Transformer blocks** (12 to 48, depending on size). Each block applies `x = x + attention(layerNorm(x))`, then `x = x + mlp(layerNorm(x))`.
   - Attention projects to Q, K and V, splits into heads, computes `softmax(QKᵀ / √64)` with a causal mask, and multiplies by V.
   - The MLP expands to 4× width, applies GELU, and projects back down.
4. **Logits**: a final layer norm, then a multiply by `wteᵀ` (GPT-2 ties the input and output embeddings), gives one score for each of the 50,257 tokens.
5. **Generate**: pick the highest-scoring token, append it, and repeat. Each new token's keys and values are added to the KV cache, so earlier tokens are never recomputed.

## Project layout

```
app/src/main/java/openllm/
  tensors/    Tensor
  backend/    Backend interface, CpuBackend
  io/         SafetensorReader, TensorInfo
  tokenizer/  Gpt2Tokenizer
  model/      Gpt2Config, GptModel, Gpt2Weights, LayerWeights, Gpt2Model, KvCache, Qkv
  generate/   Generator
  cli/        Main
```

## Roadmap

- Sampling: temperature and top-k
- Command-line flags for token count and sampling settings
- Faster CPU matmul (multithreading, SIMD via the Vector API)
- Visualizing attention patterns and the residual stream
- Modern architectures (RoPE, RMSNorm, SwiGLU, grouped-query attention)
- A GPU backend

## Acknowledgements

The model weights and tokenizer are OpenAI's GPT-2, downloaded from Hugging Face ([openai-community/gpt2](https://huggingface.co/openai-community/gpt2)).
