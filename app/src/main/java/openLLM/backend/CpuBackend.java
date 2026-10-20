package openllm.backend;

import java.util.Arrays;
import java.util.stream.IntStream;

import openllm.tensors.Tensor;

/**
 * A straightforward single-threaded {@link Backend} written in plain Java.
 *
 * <p>Correctness and readability come first: loops are written out so each
 * operation can be checked against its formula. Inputs are made contiguous
 * before their flat arrays are read, so views (e.g. a transpose) work too.
 */
public final class CpuBackend implements Backend {

    private static final int CORES = Runtime.getRuntime().availableProcessors();

    /** {@code sqrt(2/pi)}, the constant in the tanh approximation of GELU. */
    private static final double SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);
    
    @Override
    public Tensor matmul(Tensor t1, Tensor t2) {
        t1 = t1.contiguous();
        t2 = t2.contiguous();
        if (t1.rank() != 2 || t2.rank() != 2) {
            throw new IllegalArgumentException("Tensors must be of rank 2 to be multiplied");
        }
        int[] t1Shape = t1.shape();
        int[] t2Shape = t2.shape();
        float[] t1Data = t1.rawData();
        float[] t2Data = t2.rawData();
        if (t1Shape[1] != t2Shape[0]) {
            throw new IllegalArgumentException("interior matrix dimensions must match for matmul");
        }
        
        int m = t1Shape[0];
        int inner = t1Shape[1];
        int n = t2Shape[1];
        float[] outData = new float[m * n];
        if (m == 1) {
            // Decode: a single row, so split the output columns across cores instead.
            int chunks = Math.min(CORES, n);
            IntStream.range(0, chunks).parallel().forEach(c -> {
                int jStart = c * n / chunks;
                int jEnd = (c + 1) * n / chunks;
                for (int k = 0; k < inner; k++) {
                    float a = t1Data[k];
                    int rowOffset = k * n;
                    for (int j = jStart; j < jEnd; j++) {
                        outData[j] += a * t2Data[rowOffset + j];
                    }
                }
            });
        } else {
            // Prefill: many rows, each computed independently on its own core.
            IntStream.range(0, m).parallel().forEach(i -> {
                for (int k = 0; k < inner; k++) {
                    float a = t1Data[i * inner + k];
                    for (int j = 0; j < n; j++) {
                        outData[i * n + j] += a * t2Data[k * n + j];
                    }
                }
            });
        }
        return Tensor.of(outData, m, n);
    }

    @Override
    public Tensor add(Tensor a, Tensor b) {
        if (!Arrays.equals(a.shape(), b.shape())) {
            throw new IllegalArgumentException("tensors must have the same shape to be added");
        }
        a = a.contiguous();
        b = b.contiguous();
        float[] aData = a.rawData();
        float[] bData = b.rawData();
        float[] out = new float[a.size()];
        for (int i = 0; i < a.size(); i++) {
            out[i] = aData[i] + bData[i];
        }
        return Tensor.of(out, a.shape());
    }

    @Override
    public Tensor addBias(Tensor x, Tensor bias) {
        if(x.rank() != 2 || bias.rank() != 1) {
            throw new IllegalArgumentException("Tensors x and bias must have ranks 2 and 1 respectively");
        }
        x = x.contiguous();
        float[] xData = x.rawData();
        float[] biasData = bias.rawData();
        float[] outData = new float[x.size()];
        int[] xShape = x.shape();
        if(xShape[1] != bias.shape()[0]) {
            throw new IllegalArgumentException("Tensors x and bias must have the same number of cols");
        }   
        for (int i = 0; i < xShape[0]; i++) {
            for (int j = 0; j < xShape[1]; j++) {
                outData[i * xShape[1] + j] = xData[i * xShape[1] + j] + biasData[j];
            }
        }

        return Tensor.of(outData, xShape);
        
    }

    @Override
    public Tensor scale(Tensor x, float factor) {
        x = x.contiguous();
        float[] xData = x.getData();
        for (int i = 0; i < x.size(); i++) {
            xData[i] *= factor;
        }    
        return Tensor.of(xData, x.shape());
    }

    @Override // gelu(x) = 0.5 · x · (1 + tanh( √(2/π) · (x + 0.044715 · x³) ))
    public Tensor gelu(Tensor x) {
        x = x.contiguous();
        float[] xData = x.getData();
        for (int i = 0; i < x.size(); i++) {
            xData[i] = 0.5f * xData[i] * (1 + (float) Math.tanh(SQRT_2_OVER_PI * (xData[i] + 0.044715 * xData[i] * xData[i] * xData[i])));
        }
        return Tensor.of(xData, x.shape());
    }

    @Override // softmax(x)_j = exp(x_j) / Σ_k exp(x_k)
    public Tensor softmax(Tensor x) {
        x = x.contiguous();
        float[] xData = x.getData();
        int[] xShape = x.shape();
        int rows, cols;
        if (x.rank() == 1) {
            rows = 1;
            cols = xShape[0];
        } else if (x.rank() == 2) {
            rows = xShape[0];
            cols = xShape[1];
        } else {
            throw new IllegalArgumentException("softmax expects a 1-D or 2-D tensor");
        }
        for (int i = 0; i < rows; i++) {
            float max = xData[i * cols];
            // Find row max
            for (int j = 0; j < cols; j++) {
                max = Math.max(max, xData[i * cols + j]);
            }
            float sum = 0;
            // compute and sum e_j's
            for (int j = 0; j < cols; j++) {
                xData[i * cols + j] = (float) Math.exp((xData[i * cols + j]) - max);
                sum += xData[i * cols + j];
            }
            // divide by e_j
            for (int j = 0; j < cols; j++) {
                xData[i * cols + j] /= sum;
            }
        }
        return Tensor.of (xData, xShape);
            
    }

    @Override
    public Tensor layerNorm(Tensor x, Tensor gamma, Tensor beta, float eps) {
        if (x.rank() != 2 || gamma.rank() != 1 || beta.rank() != 1) {
            throw new IllegalArgumentException("layerNorm expects x of rank 2 and gamma, beta of rank 1");
        }
        int[] xShape = x.shape();
        int rows = xShape[0];
        int cols = xShape[1];
        if (gamma.size() != cols || beta.size() != cols) {
            throw new IllegalArgumentException("gamma and beta must have one value per column of x");
        }
        x = x.contiguous();
        float[] xData = x.getData();
        float[] gammaData = gamma.rawData();
        float[] betaData = beta.rawData();
        for (int i = 0; i < rows; i++) {
            float sum = 0;
            for (int j = 0; j < cols; j++) {
                sum += xData[i * cols + j];
            }
            float mean = sum / cols;
            float sqDiffSum = 0;
            for (int j = 0; j < cols; j++) {
                float d = xData[i * cols + j] - mean;
                sqDiffSum += d * d;
            }
            float var = sqDiffSum / cols;
            float std = (float) Math.sqrt(var + eps);
            for (int j = 0; j < cols; j++) {
                xData[i * cols + j] = (xData[i * cols + j] - mean) / std * gammaData[j] + betaData[j];
            }
        }
        return Tensor.of(xData, xShape);
    }
}
