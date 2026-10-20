package openllm.cli;

import java.nio.file.Path;

import openllm.backend.CpuBackend;
import openllm.generate.Generator;
import openllm.generate.Sampler;
import openllm.model.Gpt2Model;
import openllm.model.Gpt2Weights;
import openllm.model.GptModel;
import openllm.tokenizer.Gpt2Tokenizer;
import openllm.backend.Backend;

import java.util.List;
import java.util.concurrent.Callable;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

@Command(name = "openllm", mixinStandardHelpOptions = true, description = "placeholder help command")
public class Main implements Callable <Integer>{
    @Option(names = {"-m", "--model"}, defaultValue = "SMALL",
        description = "small, medium, large or xl (default: ${DEFAULT-VALUE})"
    )
    GptModel modelName;
    @Option(names = {"-t", "--temp"}, defaultValue = "0.8",
        description = "temperature of the model (default: ${DEFAULT-VALUE})"
    )
    float temperature;
    @Option(names = {"-k", "--top-k"}, defaultValue = "40",
        description = "sample only from the k most likely tokens (default: ${DEFAULT-VALUE})"
    )
    int topK;
    @Parameters(arity = "1..*", description = "the prompt text"
    ) 
    List<String> prompt;

    private static final int MAX_NEW_TOKENS = 30;
    @Override
    public Integer call() throws Exception {
        System.out.println("Welcome to openLLM!");
        System.out.println("Model: " + modelName + "\n");
        Gpt2Weights weights = Gpt2Weights.load(modelName.weightsPath(), modelName.config());
        Backend backend = new CpuBackend();
        Gpt2Model model = new Gpt2Model(weights, backend);
        Sampler sampler = new Sampler(temperature, topK, System.nanoTime(), backend);
        Gpt2Tokenizer tokenizer = new Gpt2Tokenizer(Path.of("../models/gpt2-small/vocab.json"), Path.of("../models/gpt2-small/merges.txt"));
        Generator generator = new Generator(model, tokenizer, sampler);

        long start = System.nanoTime();
        String output = generator.generate(String.join(" ", prompt), MAX_NEW_TOKENS);
        long runtime = System.nanoTime() - start;
        System.out.println(output);
        System.out.println("Runtime: " + runtime / 1_000_000_000.0 + "s");
        System.out.println("Tokens/Sec: " + MAX_NEW_TOKENS / (runtime / 1_000_000_000.0) + "t/s");
        return 0;
    }

    public static void main(String[] args) {
        int out = new CommandLine(new Main()).setCaseInsensitiveEnumValuesAllowed(true).execute(args);
        System.exit(out);
    }
}