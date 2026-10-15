package openllm.generate;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import openllm.backend.Backend;
import openllm.model.Gpt2Model;
import openllm.model.KvCache;
import openllm.tensors.Tensor;
import openllm.tokenizer.Gpt2Tokenizer;

public class Generator {
    private Gpt2Model model;
    private Gpt2Tokenizer tokenizer;
    private Sampler sampler;
    
    public Generator (Gpt2Model model, Gpt2Tokenizer tokenizer) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.sampler = new Sampler(0, 1, 0, model.backend());
    }
    public Generator(Gpt2Model model, Gpt2Tokenizer tokenizer, Sampler sampler) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.sampler = sampler;
    }
    
    public String generate(String prompt, int maxNewTokens) {
        KvCache cache = model.newCache();
        List<Integer> ids = new ArrayList<Integer>();
        int[] input = tokenizer.encode(prompt);
        ids.addAll(Arrays.stream(input).boxed().toList());
        boolean done = false;
        for (int i = 0; i < maxNewTokens && !done && ids.size() < 1024; i++) {
            Tensor logits = model.nextTokenLogits(input, cache);
            int bestIdx = sampler.sample(logits);
            if (bestIdx == 50256) {
                done = true;
            } else {
                ids.add(bestIdx);
                input = new int[]{bestIdx};
            }
        }
        return tokenizer.decode(ids.stream().mapToInt(Integer::intValue).toArray());
    }
}
