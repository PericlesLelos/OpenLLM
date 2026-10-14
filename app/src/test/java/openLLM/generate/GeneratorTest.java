package openllm.generate;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Path;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import openllm.backend.CpuBackend;
import openllm.model.Gpt2Model;
import openllm.model.Gpt2Weights;
import openllm.tokenizer.Gpt2Tokenizer;

public class GeneratorTest {
    // Gradle runs tests from the app/ folder, so the model folder is one level up.
    private static final Path MODEL = Path.of("../models/gpt2/model.safetensors");
    private static final Path VOCAB = Path.of("../models/gpt2/vocab.json");
    private static final Path MERGES = Path.of("../models/gpt2/merges.txt");

    private static Generator generator;

    @BeforeAll static void loadOnce() throws IOException {
        Gpt2Model model = new Gpt2Model(Gpt2Weights.load(MODEL), new CpuBackend());
        generator = new Generator(model, new Gpt2Tokenizer(VOCAB, MERGES));
    }

    // Asking for zero new tokens should give the prompt back unchanged (encode then decode round trip).
    @Test void zeroNewTokensReturnsThePrompt() {
        assertEquals("Hello world,", generator.generate("Hello world,", 0));
    }

    // One greedy step: the top next token after "Hello world," is " I" (id 314).
    @Test void oneNewTokenAppendsTheTopPrediction() {
        assertEquals("Hello world, I", generator.generate("Hello world,", 1));
    }

    // Greedy continuation computed with numpy from model.safetensors, one token at a time.
    // Stops at 4 tokens because the 5th choice (" what") wins by only 0.04, which float rounding could flip.
    @Test void greedyContinuationMatchesReference() {
        assertEquals("Hello world, I'm not sure", generator.generate("Hello world,", 4));
    }

    // Greedy decoding is deterministic: the same prompt always gives the same text.
    @Test void greedyIsDeterministic() {
        assertEquals(generator.generate("The cat", 3), generator.generate("The cat", 3));
    }
}
