package openllm.io;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import org.junit.jupiter.api.Test;

import openllm.tensors.Tensor;

public class SafetensorReaderTest {
    // Gradle runs tests from the app/ folder, so the model folder is one level up.
    private static final Path GPT2 = Path.of("../models/gpt2/model.safetensors");

    @Test void readHeaderFindsAllGpt2Tensors() throws IOException {
        Map<String, TensorInfo> tensors = SafetensorReader.readHeader(GPT2);
        assertEquals(160, tensors.size());
    }

    @Test void readHeaderReadsShapeAndOffsets() throws IOException {
        Map<String, TensorInfo> tensors = SafetensorReader.readHeader(GPT2);
        TensorInfo wte = tensors.get("wte.weight");
        assertNotNull(wte);
        assertEquals("F32", wte.dtype());
        assertArrayEquals(new int[] {50257, 768}, wte.shape());
        // 4 bytes per float32, so the byte range must match the element count.
        assertEquals(50257L * 768 * 4, wte.end() - wte.start());
    }

    @Test void everyTensorsByteRangeMatchesItsShape() throws IOException {
        for (TensorInfo t : SafetensorReader.readHeader(GPT2).values()) {
            long elements = 1;
            for (int d : t.shape()) {
                elements *= d;
            }
            assertEquals(elements * 4, t.end() - t.start(), t.name());
        }
    }

    // Expected values were read straight from model.safetensors with a separate script.
    @Test void loadTensorReadsLnFWeight() throws IOException {
        TensorInfo info = SafetensorReader.readHeader(GPT2).get("ln_f.weight");
        Tensor t = SafetensorReader.loadTensor(GPT2, info);
        assertArrayEquals(new int[] {768}, t.shape());
        assertEquals(1.39708f, t.get(0), 1e-5f);
        assertEquals(1.374953f, t.get(1), 1e-5f);
        assertEquals(1.886957f, t.get(2), 1e-5f);
        assertEquals(1.168837f, t.get(3), 1e-5f);
    }

    @Test void loadTensorReadsLnFBias() throws IOException {
        TensorInfo info = SafetensorReader.readHeader(GPT2).get("ln_f.bias");
        Tensor t = SafetensorReader.loadTensor(GPT2, info);
        assertArrayEquals(new int[] {768}, t.shape());
        assertEquals(0.001087f, t.get(0), 1e-5f);
        assertEquals(0.036529f, t.get(1), 1e-5f);
        assertEquals(-0.067296f, t.get(2), 1e-5f);
        assertEquals(0.000164f, t.get(3), 1e-5f);
    }

    // wte.weight is the token embedding table, about 154 MB, so this one is slower.
    @Test void loadTensorReadsTokenEmbeddings() throws IOException {
        TensorInfo info = SafetensorReader.readHeader(GPT2).get("wte.weight");
        Tensor t = SafetensorReader.loadTensor(GPT2, info);
        assertArrayEquals(new int[] {50257, 768}, t.shape());
        assertEquals(-0.110103f, t.get(0, 0), 1e-5f);
        assertEquals(-0.039267f, t.get(0, 1), 1e-5f);
        assertEquals(0.033108f, t.get(0, 2), 1e-5f);
        assertEquals(0.133826f, t.get(0, 3), 1e-5f);
    }
}
