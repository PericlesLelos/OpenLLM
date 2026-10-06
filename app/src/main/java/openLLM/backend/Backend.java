package openllm.backend;

import openllm.tensors.Tensor;

public interface Backend {

    Tensor matmul(Tensor a, Tensor b);
    Tensor add(Tensor a, Tensor b);
    Tensor addBias(Tensor x, Tensor bias);
    Tensor scale(Tensor x, float factor);
    Tensor gelu(Tensor x);
    Tensor softmax(Tensor x);
    Tensor layerNorm(Tensor x, Tensor gamma, Tensor beta, float eps);

} 