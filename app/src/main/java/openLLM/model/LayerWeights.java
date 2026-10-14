package openllm.model;

import openllm.tensors.Tensor;

/**
 * The learned weights of one transformer block ({@code h.<i>.*} in the file).
 *
 * <p>Matrices are stored {@code [in, out]}, so {@code matmul(x, W)} needs no
 * transpose. Shapes below are for GPT-2 small ({@code nEmbd = 768}).
 *
 * @param ln1Weight first layer norm scale ({@code ln_1.weight}), {@code [768]}
 * @param ln1Bias first layer norm shift ({@code ln_1.bias}), {@code [768]}
 * @param attnQkvWeight Q, K and V projections packed side by side ({@code attn.c_attn.weight}), {@code [768, 2304]}
 * @param attnQkvBias bias for the packed QKV projection ({@code attn.c_attn.bias}), {@code [2304]}
 * @param attnProjWeight attention output projection ({@code attn.c_proj.weight}), {@code [768, 768]}
 * @param attnProjBias attention output bias ({@code attn.c_proj.bias}), {@code [768]}
 * @param ln2Weight second layer norm scale ({@code ln_2.weight}), {@code [768]}
 * @param ln2Bias second layer norm shift ({@code ln_2.bias}), {@code [768]}
 * @param mlpFcWeight MLP expansion ({@code mlp.c_fc.weight}), {@code [768, 3072]}
 * @param mlpFcBias MLP expansion bias ({@code mlp.c_fc.bias}), {@code [3072]}
 * @param mlpProjWeight MLP projection back down ({@code mlp.c_proj.weight}), {@code [3072, 768]}
 * @param mlpProjBias MLP projection bias ({@code mlp.c_proj.bias}), {@code [768]}
 */
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
