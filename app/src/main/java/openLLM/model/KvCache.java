package openllm.model;

import java.util.Arrays;

import openllm.tensors.Tensor;

public class KvCache {
    
    private final float[][] kBuffers;
    private final float[][] vBuffers;
    private final int nLayer;
    private final int capacity;
    private final int nEmbd;
    private int tokensStored;

    /** Creates a cache that can hold a full context window (nCtx tokens). */
    public KvCache(Gpt2Config gpt2Config) {
        this(gpt2Config, gpt2Config.nCtx());
    }

    /**
     * Creates a cache that can hold at most {@code capacity} tokens.
     * Use a small capacity for one-off calls so the buffers stay small.
     *
     * @throws IllegalArgumentException if capacity is not between 1 and nCtx
     */
    public KvCache(Gpt2Config gpt2Config, int capacity) {
        if (capacity < 1 || capacity > gpt2Config.nCtx()) {
            throw new IllegalArgumentException("capacity must be between 1 and nCtx " + gpt2Config.nCtx() + ", got " + capacity);
        }
        this.nLayer = gpt2Config.nLayer();
        this.capacity = capacity;
        this.nEmbd = gpt2Config.nEmbd();
        this.kBuffers = new float[nLayer][capacity * nEmbd];
        this.vBuffers = new float[nLayer][capacity * nEmbd];
    }

    float[] getKBuffer(int layer) {
        return kBuffers[layer];
    }

    float[] getVBuffer(int layer) {
        return vBuffers[layer];
    }
    void append(int layer, Tensor k, Tensor v){  
        int newRows = k.shape()[0];
        k = k.contiguous();
        v = v.contiguous();
        if (tokensStored + newRows > capacity) {
            throw new IndexOutOfBoundsException("cache is full: " + tokensStored + " stored + " + newRows + " new > capacity " + capacity);
        }
        int start = tokensStored * nEmbd;
        System.arraycopy(k.rawData(), 0, kBuffers[layer], start, newRows * nEmbd);
        System.arraycopy(v.rawData(), 0, vBuffers[layer], start, newRows * nEmbd);
       
    }
    void advance(int n) {
        if(tokensStored + n <= capacity) {
            tokensStored += n;
        } else {
            throw new IndexOutOfBoundsException("advance would exceed capacity: " + tokensStored + " stored + " + n + " new > capacity " + capacity);
        }
    }
    Tensor readK(int layer, int rows){
        return Tensor.of(Arrays.copyOf(kBuffers[layer], rows * nEmbd), rows, nEmbd);
    }
    Tensor readV(int layer, int rows){
        return Tensor.of(Arrays.copyOf(vBuffers[layer], rows * nEmbd), rows, nEmbd);
    }
    int getCount() {
        return this.tokensStored;
    }
}
