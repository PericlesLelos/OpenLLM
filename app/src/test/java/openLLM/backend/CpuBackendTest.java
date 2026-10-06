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
}
