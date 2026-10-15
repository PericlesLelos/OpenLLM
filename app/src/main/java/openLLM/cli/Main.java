package openllm.cli;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;

import openllm.backend.CpuBackend;
import openllm.generate.Generator;
import openllm.generate.Sampler;
import openllm.model.Gpt2Model;
import openllm.model.Gpt2Weights;
import openllm.model.GptModel;
import openllm.tokenizer.Gpt2Tokenizer;
import openllm.backend.Backend;

/** Command-line entry point. Currently a placeholder that prints a greeting. */
public class Main {

    private static final int MAX_NEW_TOKENS = 100;
    public static void main(String[] args) throws IOException{
        String[] promptArgs = args;
        GptModel choice = GptModel.SMALL;

        if(args.length > 0 && args[0].startsWith("--model=")) {
            String name = args[0].substring("--model=".length());
            choice = GptModel.valueOf(name.toUpperCase());
            promptArgs = Arrays.copyOfRange(args, 1, args.length);
        }
        System.out.println("Welcome to openLLM!");
        System.out.println("Model: " + choice + "\n");
        Gpt2Weights weights = Gpt2Weights.load(choice.weightsPath(), choice.config());
        Backend backend = new CpuBackend();
        Gpt2Model model = new Gpt2Model(weights, backend);
        Sampler sampler = new Sampler(0.5f, 30, System.nanoTime(), backend);
        Gpt2Tokenizer tokenizer = new Gpt2Tokenizer(Path.of("../models/gpt2-small/vocab.json"), Path.of("../models/gpt2-small/merges.txt"));
        Generator generator = new Generator(model, tokenizer, sampler);

        long start = System.nanoTime();
        String output = generator.generate(String.join(" ", promptArgs), MAX_NEW_TOKENS);
        long runtime = System.nanoTime() - start;
        System.out.println(output);
        System.out.println("Runtime: " + runtime / 1_000_000_000.0 + "s");
        System.out.println("Tokens/Sec: " + MAX_NEW_TOKENS / (runtime / 1_000_000_000.0) + "t/s");
    }
}