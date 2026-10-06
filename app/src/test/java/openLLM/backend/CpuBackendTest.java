package openllm.backend;

import org.junit.jupiter.api.Test;

import openllm.tensors.Tensor;


import static org.junit.jupiter.api.Assertions.*;

public class CpuBackendTest {
    @Test void matmulRejectsBadDimensions() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 1, 6);
        Backend b = new CpuBackend();
        assertThrows(IllegalArgumentException.class,
            () -> {
                Tensor t2 = b.matmul(t, t1);
    });
    }

    @Test void matmulRejectsBadRank() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6},  6);
        Tensor t1 = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 6);
        Backend b = new CpuBackend();
        assertThrows(IllegalArgumentException.class, 
            () -> {
                Tensor t2 = b.matmul(t, t1);
        });
        
    }
    @Test void matmulSuccedsWithGoodDimAndRank() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = t.transpose(0, 1); 
        Tensor product = Tensor.of(new float[]{14, 32, 32, 77}, 2, 2);
        Backend b = new CpuBackend();  
        assertTrue((b.matmul(t, t1).allClose(product,0.001f)));
    }

    @Test void addRejectBadShape() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 3, 2);
        Backend b = new CpuBackend();
        assertThrows(IllegalArgumentException.class,
            () -> b.add(t, t1));     

    }

    @Test void addSucceedsWithGoodInput() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Backend b = new CpuBackend();
        Tensor tTruth = Tensor.of(new float[]{2f, 4f, 6f, 8f, 10f, 12f}, 2, 3);
        assertTrue(b.add(t, t1).allClose(tTruth, 0.000f));
    }

    @Test void addSuccessWithTranspose() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4}, 2, 2);
        Tensor t1 = t.transpose(0, 1);
        Backend b = new CpuBackend();
        Tensor tTruth = Tensor.of(new float[]{2f, 5f, 5f, 8f}, 2, 2);
        assertTrue(b.add(t, t1).allClose(tTruth, 0.000f));
    }

    @Test void addDoesntModifyOriginalInput() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Backend b = new CpuBackend();
        b.add(t, t1);
        assertEquals(1, t.get(0, 0));
    }

    @Test void addBiasRejectBadBiasDimension() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 1, 6);
        Tensor t1 = Tensor.of(new float[] {1, 1, 1}, 3); 
        Backend b = new CpuBackend();
        assertThrows(IllegalArgumentException.class, () -> b.addBias(t, t1));
    }

    @Test void addBiasWorkWithProperInput() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = Tensor.of(new float[] {1, 1, 1}, 3); 
        Tensor valid = Tensor.of(new float[] {2, 3, 4, 5, 6, 7}, 2, 3);
        Backend b = new CpuBackend();
        assertTrue(b.addBias(t, t1).allClose(valid, 0.001f));
    }

    @Test void scaleSucceedsWithGoodInput() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        float factor = 0.5f;
        Backend b = new CpuBackend();
        Tensor valid = Tensor.of(new float[] {0.5f, 1f, 1.5f, 2f, 2.5f, 3f}, 2, 3);
        assertTrue(b.scale(t, factor).allClose(valid, 0.001f));
    }

    @Test void scaleSucceedsWithTranspose() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor tT = t.transpose(0, 1);
        float factor = 0.5f;
        Backend b = new CpuBackend();
        Tensor valid = Tensor.of(new float[] {0.5f, 2f, 1f, 2.5f, 1.5f, 3f}, 3, 2);
        assertTrue(b.scale(tT, factor).allClose(valid, 0.001f));
    }

    @Test void scaleDoesntModifyInitial() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        float factor = 0.5f;
        Backend b = new CpuBackend();
        b.scale(t, factor);
        assertTrue(t.allClose(Tensor.of(new float[]{1,2,3,4,5,6}, 2, 3), 0f)); 
    }

    @Test void geluOfValues() {
        Tensor t = Tensor.of(new float[] {0, 1, -1, 2, -3, 10, -10}, 7);
        Tensor valid = Tensor.of(new float[] {0, 0.841192f, -0.158808f, 1.954598f, -0.003637f, 10, 0}, 7);
        Backend b = new CpuBackend();
        assertTrue(b.gelu(t).allClose(valid, 1e-5f));
    }

    @Test void softMaxOfValues() {
        Tensor t = Tensor.of(new float[] {1, 2, 3}, 3);
        Tensor valid = Tensor.of(new float[]{0.090031f, 0.244728f, 0.665241f}, 3);
        Backend b  = new CpuBackend();
        assertTrue(b.softmax(t).allClose(valid, 1e-5f));
    }

    @Test void softMaxStabilityTest() {
        Tensor t = Tensor.of(new float[] {1000, 1001, 1002}, 3);
        Tensor valid = Tensor.of(new float[]{0.090031f, 0.244728f, 0.665241f}, 3);
        Backend b  = new CpuBackend();
        assertTrue(b.softmax(t).allClose(valid, 1e-5f));
    }

    @Test void softMaxDifferenceTest() {
        Tensor t = Tensor.of(new float[] {-1, 0, 1}, 3);
        Tensor valid = Tensor.of(new float[]{0.090031f, 0.244728f, 0.665241f}, 3);
        Backend b  = new CpuBackend();
        assertTrue(b.softmax(t).allClose(valid, 1e-5f));
    }

    @Test void softMaxAllSameValues() {
        Tensor t = Tensor.of(new float[] {0, 0, 0, 0}, 4);
        Tensor valid = Tensor.of(new float[]{0.25f, 0.25f, 0.25f, 0.25f}, 4);
        Backend b  = new CpuBackend();
        assertTrue(b.softmax(t).allClose(valid, 1e-5f));
    }

    @Test void softMaxOnTwoDTensor() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        // Each row is softmaxed on its own. [4, 5, 6] has the same gaps as [1, 2, 3],
        // so both rows give the same probabilities.
        Tensor valid = Tensor.of(new float[] {
            0.090031f, 0.244728f, 0.665241f,
            0.090031f, 0.244728f, 0.665241f}, 2, 3);
        Backend b = new CpuBackend();
        assertTrue(b.softmax(t).allClose(valid, 1e-5f));
    }

    @Test void softMaxRowsSumToOne() {
        Tensor t = Tensor.of(new float[] {-5, 0, 5, 2, 2, 7}, 2, 3);
        Backend b = new CpuBackend();
        Tensor out = b.softmax(t);
        for (int i = 0; i < 2; i++) {
            float sum = 0;
            for (int j = 0; j < 3; j++) {
                sum += out.get(i, j);
            }
            assertEquals(1f, sum, 1e-5f);
        }
    }

    @Test void layerNormBasic() {
        Tensor x = Tensor.of(new float[] {1, 2, 3, 4}, 1, 4);
        Tensor gamma = Tensor.of(new float[] {1, 1, 1, 1}, 4);
        Tensor beta = Tensor.of(new float[] {0, 0, 0, 0}, 4);
        Tensor valid = Tensor.of(new float[] {-1.341635f, -0.447212f, 0.447212f, 1.341635f}, 1, 4);
        Backend b = new CpuBackend();
        assertTrue(b.layerNorm(x, gamma, beta, 1e-5f).allClose(valid, 1e-5f));
    }

    @Test void layerNormAppliesGammaAndBeta() {
        Tensor x = Tensor.of(new float[] {1, 2, 3, 4}, 1, 4);
        Tensor gamma = Tensor.of(new float[] {2, 2, 2, 2}, 4);
        Tensor beta = Tensor.of(new float[] {1, 1, 1, 1}, 4);
        Tensor valid = Tensor.of(new float[] {-1.683271f, 0.105576f, 1.894424f, 3.683271f}, 1, 4);
        Backend b = new CpuBackend();
        assertTrue(b.layerNorm(x, gamma, beta, 1e-5f).allClose(valid, 1e-5f));
    }

    @Test void layerNormPerColumnGammaAndBeta() {
        Tensor x = Tensor.of(new float[] {1, 2, 3, 4}, 1, 4);
        Tensor gamma = Tensor.of(new float[] {1, 0.5f, 2, -1}, 4);
        Tensor beta = Tensor.of(new float[] {0, 1, 0, 0.5f}, 4);
        Tensor valid = Tensor.of(new float[] {-1.341635f, 0.776394f, 0.894424f, -0.841635f}, 1, 4);
        Backend b = new CpuBackend();
        assertTrue(b.layerNorm(x, gamma, beta, 1e-5f).allClose(valid, 1e-5f));
    }

    @Test void layerNormConstantRowGivesZerosNotNaN() {
        Tensor x = Tensor.of(new float[] {5, 5, 5, 5}, 1, 4);
        Tensor gamma = Tensor.of(new float[] {1, 1, 1, 1}, 4);
        Tensor beta = Tensor.of(new float[] {0, 0, 0, 0}, 4);
        Tensor valid = Tensor.of(new float[] {0, 0, 0, 0}, 1, 4);
        Backend b = new CpuBackend();
        assertTrue(b.layerNorm(x, gamma, beta, 1e-5f).allClose(valid, 1e-5f));
    }

    @Test void layerNormNormalizesEachRowSeparately() {
        // Row 2 is row 1 times 10; after normalizing, both rows come out the same.
        Tensor x = Tensor.of(new float[] {1, 2, 3, 4, 10, 20, 30, 40}, 2, 4);
        Tensor gamma = Tensor.of(new float[] {1, 1, 1, 1}, 4);
        Tensor beta = Tensor.of(new float[] {0, 0, 0, 0}, 4);
        Tensor valid = Tensor.of(new float[] {
            -1.341635f, -0.447212f, 0.447212f, 1.341635f,
            -1.341641f, -0.447214f, 0.447214f, 1.341641f}, 2, 4);
        Backend b = new CpuBackend();
        assertTrue(b.layerNorm(x, gamma, beta, 1e-5f).allClose(valid, 1e-5f));
    }

    @Test void layerNormRejectsWrongGammaLength() {
        Tensor x = Tensor.of(new float[] {1, 2, 3, 4}, 1, 4);
        Tensor gamma = Tensor.of(new float[] {1, 1, 1}, 3);
        Tensor beta = Tensor.of(new float[] {0, 0, 0, 0}, 4);
        Backend b = new CpuBackend();
        assertThrows(IllegalArgumentException.class, () -> b.layerNorm(x, gamma, beta, 1e-5f));
    }
}
