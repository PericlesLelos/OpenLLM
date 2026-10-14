package openllm.cli;

import java.io.IOException;
import java.nio.file.Path;

import openllm.backend.CpuBackend;
import openllm.generate.Generator;
import openllm.model.Gpt2Model;
import openllm.model.Gpt2Weights;
import openllm.tokenizer.Gpt2Tokenizer;
import openllm.backend.Backend;

/** Command-line entry point. Currently a placeholder that prints a greeting. */
public class Main {

    private static final int MAX_NEW_TOKENS = 30;
    public static void main(String[] args) throws IOException{
        System.out.println("Welcome to openLLM!");
        
        Gpt2Weights weights = Gpt2Weights.load(Path.of("../models/gpt2/model.safetensors"));
        Backend backend = new CpuBackend();
        Gpt2Model model = new Gpt2Model(weights, backend);
        Gpt2Tokenizer tokenizer = new Gpt2Tokenizer(Path.of("../models/gpt2/vocab.json"), Path.of("../models/gpt2/merges.txt"));
        Generator generator = new Generator(model, tokenizer);

        long start = System.nanoTime();
        String output = generator.generate(String.join(" ", args), MAX_NEW_TOKENS);
        long runtime = System.nanoTime() - start;
        System.out.println(output);
        System.out.println("Runtime: " + runtime / 1_000_000_000.0 + "s");
        System.out.println("Tokens/Sec: " + MAX_NEW_TOKENS / (runtime / 1_000_000_000.0) + "t/s");
    }
}