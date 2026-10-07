package openllm.model;

import openllm.tensors.Tensor;

public record LayerWeights(
    Tensor ln1Weight,
    Tensor ln1Bias,
    Tensor attnQkvWeight,
    Tensor attnQkvBias,
    Tensor attnProjWeight,
    Tensor attnProjBias,
    Tensor ln2Weight,
    Tensor ln2Bias,
    Tensor mlpFcWeight,
    Tensor mlpFcBias,
    Tensor mlpProjWeight,
    Tensor mlpProjBias
) {}
