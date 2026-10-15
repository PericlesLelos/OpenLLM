package openllm.generate;

import java.util.Arrays;
import java.util.Random;

import openllm.backend.Backend;
import openllm.tensors.Tensor;

public class Sampler {
    private final float temperature;
    private final int topK;
    private final Random random;
    private final Backend backend;

    public Sampler(float temperature, int topK, long seed, Backend backend) {
        if (temperature < 0) {
            throw new IllegalArgumentException("Temperature cannot be less than 0");
        }
        if (topK < 1) {
            throw new IllegalArgumentException("topK must be 1 or greater");
        }
        this.temperature = temperature;
        this.topK = topK;
        this.random = new Random(seed);
        this.backend = backend;
    }

    public int sample(Tensor logits) {
        if (logits.shape()[0] != 1) {
            throw new IllegalArgumentException("expected logits for one position, got " + logits.shape()[0] + " rows");
        }
        if (temperature == 0) {
            return argMax(logits);
        }
        logits = logits.contiguous();
        logits = backend.scale(logits, 1 / temperature);
        float[] logitsData = logits.getData();
        float[] sortedLogits = Arrays.copyOf(logitsData, logitsData.length);
        Arrays.sort(sortedLogits);
        int k = Math.min(topK, logitsData.length);
        float cutoff = sortedLogits[sortedLogits.length - k];
        for (int i = 0; i < logitsData.length; i++) {
            if (logitsData[i] < cutoff) {
                logitsData[i] = Float.NEGATIVE_INFINITY;
            }
        }
        float[] probabilities = backend.softmax(Tensor.of(logitsData, sortedLogits.length)).rawData();
        float r = random.nextFloat();
        float cumulative = 0;
        int i = 0;
        int lastNonzero = -1;
        while (i < probabilities.length && cumulative <= r) {
            if (probabilities[i] > 0) {
                lastNonzero = i;
                cumulative += probabilities[i];                
            }   
            i++;        
        }
        return lastNonzero;

    }

    private int argMax(Tensor x) {
        x = x.contiguous();
        float[] xData = x.rawData();
        float max = Float.NEGATIVE_INFINITY;
        int pos = 0;
        for (int i = 0; i < xData.length; i++) {
            if (xData[i] > max) {
                max = xData[i];
                pos = i;
            }
        }
        return pos;
    }
}
