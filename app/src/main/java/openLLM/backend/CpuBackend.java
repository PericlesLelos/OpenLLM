package openllm.backend;

import openllm.tensors.Tensor;

public final class CpuBackend implements Backend {
    
    @Override
    public Tensor matmul(Tensor t1, Tensor t2) {
        if (t1.rank() != 2 || t2.rank() != 2) {
            throw new IllegalArgumentException("Tensors must be of rank 2 to be multiplied");
        }
        int[] t1Shape = t1.shape();
        int[] t2Shape = t2.shape();
        if (t1Shape[1] != t2Shape[0]) {
            throw new IllegalArgumentException("interior matrix dimensions must match for matmul");
        }
        
        Tensor out = Tensor.zeros(t1Shape[0], t2Shape[1]);
        for (int i = 0; i < t1Shape[0]; i++) {
            for(int j = 0; j < t2Shape[1]; j++) {
                float sum = 0;
                for (int k = 0; k < t1Shape[1]; k++) {
                    sum += t1.get(i, k) * t2.get(k, j);
                }
                out.set(sum, i, j);
            }
        }
        return out;
    }
}
