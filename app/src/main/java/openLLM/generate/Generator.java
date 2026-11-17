package openllm.generate;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

import openllm.model.KvCache;
import openllm.model.LanguageModel;
import openllm.tensors.Tensor;
import openllm.tokenizer.Tokenizer;

public class Generator {
    private LanguageModel model;
    private Tokenizer tokenizer;
    private Sampler sampler;
    
    public Generator (LanguageModel model, Tokenizer tokenizer) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.sampler = new Sampler(0, 1, 0, model.backend());
    }
    public Generator(LanguageModel model, Tokenizer tokenizer, Sampler sampler) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.sampler = sampler;
    }
    
    public String generate(String prompt, int maxNewTokens, Consumer<String> onToken) {
        KvCache cache = model.newCache();
        List<Integer> ids = new ArrayList<Integer>();
        int[] input = tokenizer.encode(prompt);
        ids.addAll(Arrays.stream(input).boxed().toList());
        boolean done = false;
        for (int i = 0; i < maxNewTokens && !done && ids.size() < model.contextLength(); i++) {
            Tensor logits = model.nextTokenLogits(input, cache);
            int bestIdx = sampler.sample(logits);
            if (bestIdx == tokenizer.eosId()) {
                done = true;
            } else {
                ids.add(bestIdx);
                input = new int[]{bestIdx};
                onToken.accept(tokenizer.decode(new int[]{bestIdx}));
            }
        }
        return tokenizer.decode(ids.stream().mapToInt(Integer::intValue).toArray());
    }
}
