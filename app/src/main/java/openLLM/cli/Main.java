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

    public static void main(String[] args) throws IOException{
        System.out.println("Welcome to openLLM!");
        
        Gpt2Weights weights = Gpt2Weights.load(Path.of("../models/gpt2/model.safetensors"));
        Backend backend = new CpuBackend();
        Gpt2Model model = new Gpt2Model(weights, backend);
        Gpt2Tokenizer tokenizer = new Gpt2Tokenizer(Path.of("../models/gpt2/vocab.json"), Path.of("../models/gpt2/merges.txt"));
        Generator generator = new Generator(model, tokenizer);

        System.out.println(generator.generate(String.join(" ", args), 30));
    }
}