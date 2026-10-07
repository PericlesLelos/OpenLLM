package openllm.model;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Path;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class Gpt2WeightsTest {
    // Gradle runs tests from the app/ folder, so the model folder is one level up.
    private static final Path GPT2 = Path.of("../models/gpt2/model.safetensors");

    // Loading reads the whole ~500 MB file, so do it once and share it across tests.
    private static Gpt2Weights weights;

    @BeforeAll static void loadOnce() throws IOException {
        weights = Gpt2Weights.load(GPT2);
    }

    @Test void configIsGpt2Small() {
        assertEquals(Gpt2Config.GPT2_SMALL, weights.config());
    }

    @Test void globalTensorsHaveTheRightShapes() {
        assertArrayEquals(new int[] {50257, 768}, weights.wte().shape());
        assertArrayEquals(new int[] {1024, 768}, weights.wpe().shape());
        assertArrayEquals(new int[] {768}, weights.lnFWeight().shape());
        assertArrayEquals(new int[] {768}, weights.lnFBias().shape());
    }

    @Test void loadsTwelveLayers() {
        assertEquals(12, weights.layers().length);
        for (LayerWeights layer : weights.layers()) {
            assertNotNull(layer);
        }
    }

    @Test void layerTensorsHaveTheRightShapes() {
        // Check the last layer too, so we know the "h." + i + "." prefix works past h.0.
        for (LayerWeights layer : new LayerWeights[] {weights.layers()[0], weights.layers()[11]}) {
            assertArrayEquals(new int[] {768}, layer.ln1Weight().shape());
            assertArrayEquals(new int[] {768}, layer.ln1Bias().shape());
            assertArrayEquals(new int[] {768, 2304}, layer.attnQkvWeight().shape());
            assertArrayEquals(new int[] {2304}, layer.attnQkvBias().shape());
            assertArrayEquals(new int[] {768, 768}, layer.attnProjWeight().shape());
            assertArrayEquals(new int[] {768}, layer.attnProjBias().shape());
            assertArrayEquals(new int[] {768}, layer.ln2Weight().shape());
            assertArrayEquals(new int[] {768}, layer.ln2Bias().shape());
            assertArrayEquals(new int[] {768, 3072}, layer.mlpFcWeight().shape());
            assertArrayEquals(new int[] {3072}, layer.mlpFcBias().shape());
            assertArrayEquals(new int[] {3072, 768}, layer.mlpProjWeight().shape());
            assertArrayEquals(new int[] {768}, layer.mlpProjBias().shape());
        }
    }

    @Test void finalLayerNormHasTheRightValues() {
        // Same values the SafetensorReader test checks, read through the model object this time.
        assertEquals(1.39708f, weights.lnFWeight().get(0), 1e-5f);
        assertEquals(0.001087f, weights.lnFBias().get(0), 1e-5f);
    }
}
