package openllm.model;

import openllm.tensors.Tensor;

/**
 * The query, key and value tensors produced by one block's QKV projection.
 * Each has shape {@code [T, nEmbd]}, one row per token.
 *
 * @param q queries: what each token is looking for
 * @param k keys: what each token offers to be matched against
 * @param v values: what each token passes along when attended to
 */
public record Qkv(Tensor q, Tensor k, Tensor v) {}
