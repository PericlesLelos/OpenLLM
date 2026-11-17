package openllm.model;

import openllm.backend.Backend;
import openllm.tensors.Tensor;

public interface LanguageModel {
    public int contextLength();
    public Backend backend();
    public KvCache newCache();
    public Tensor nextTokenLogits(int[] ids, KvCache cache);
}
