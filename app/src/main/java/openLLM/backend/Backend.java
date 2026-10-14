package openllm.backend;

import openllm.tensors.Tensor;

/**
 * The math operations the model needs, kept behind an interface so the
 * model code never depends on how they are computed.
 *
 * <p>{@link CpuBackend} is the plain-Java implementation. A faster backend
 * (multithreaded, SIMD or GPU) can be swapped in later without changing the
 * model. Every operation returns a new tensor and leaves its inputs unchanged.
 */
public interface Backend {

    /**
     * Matrix multiplication: {@code [m, k] x [k, n] -> [m, n]}.
     *
     * @param a left matrix, shape {@code [m, k]}
     * @param b right matrix, shape {@code [k, n]}
     * @return the product, shape {@code [m, n]}
     * @throws IllegalArgumentException if either input is not 2-D or the inner dimensions differ
     */
    Tensor matmul(Tensor a, Tensor b);
    /**
     * Element-wise sum of two tensors of the same shape. Used for residual connections.
     *
     * @param a first tensor
     * @param b second tensor, same shape as {@code a}
     * @return {@code a + b}
     * @throws IllegalArgumentException if the shapes differ
     */
    Tensor add(Tensor a, Tensor b);
    /**
     * Adds a 1-D bias to every row of a 2-D tensor (broadcast over rows).
     *
     * @param x input, shape {@code [rows, cols]}
     * @param bias one value per column, shape {@code [cols]}
     * @return {@code x} with {@code bias[j]} added to every element in column {@code j}
     * @throws IllegalArgumentException if the ranks are wrong or the column counts differ
     */
    Tensor addBias(Tensor x, Tensor bias);
    /**
     * Multiplies every element by a constant, e.g. the {@code 1/sqrt(headDim)}
     * factor in attention.
     *
     * @param x input tensor
     * @param factor the multiplier
     * @return {@code x * factor}, same shape as {@code x}
     */
    Tensor scale(Tensor x, float factor);
    /**
     * Applies GELU element-wise, using the tanh approximation GPT-2 was trained
     * with: {@code 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))}.
     *
     * @param x input tensor
     * @return GELU of each element, same shape as {@code x}
     */
    Tensor gelu(Tensor x);
    /**
     * Softmax over the last dimension: each row is turned into probabilities
     * that are positive and sum to 1. Subtracts the row max first so large
     * inputs don't overflow {@code exp}.
     *
     * @param x a 1-D vector or a 2-D tensor (softmax applied to each row)
     * @return a tensor of the same shape whose rows sum to 1
     * @throws IllegalArgumentException if {@code x} is not 1-D or 2-D
     */
    Tensor softmax(Tensor x);
    /**
     * Layer normalization of each row: subtract the row mean, divide by
     * {@code sqrt(variance + eps)}, then scale by {@code gamma} and shift by
     * {@code beta} column by column.
     *
     * @param x input, shape {@code [rows, cols]}
     * @param gamma learned scale, shape {@code [cols]}
     * @param beta learned shift, shape {@code [cols]}
     * @param eps small constant that avoids dividing by zero (GPT-2 uses {@code 1e-5})
     * @return the normalized tensor, shape {@code [rows, cols]}
     * @throws IllegalArgumentException if the ranks are wrong or {@code gamma}/{@code beta} don't match the column count
     */
    Tensor layerNorm(Tensor x, Tensor gamma, Tensor beta, float eps);

} 