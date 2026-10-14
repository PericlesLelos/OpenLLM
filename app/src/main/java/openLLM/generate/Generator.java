package openllm.generate;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import openllm.model.Gpt2Model;
import openllm.tensors.Tensor;
import openllm.tokenizer.Gpt2Tokenizer;

public class Generator {
    private Gpt2Model model;
    private Gpt2Tokenizer tokenizer;
    
    public Generator(Gpt2Model model, Gpt2Tokenizer tokenizer) {
        this.model = model;
        this.tokenizer = tokenizer;
    }
    
    public String generate(String prompt, int maxNewTokens) {
        List<Integer> ids = new ArrayList<Integer>();
        ids.addAll(Arrays.stream(tokenizer.encode(prompt)).boxed().collect(Collectors.toCollection(ArrayList::new)));
        boolean done = false;
        for (int i = 0; i < maxNewTokens && !done && ids.size() < 1024; i++) {
            int[] passIds = ids.stream().mapToInt(Integer::intValue).toArray();
            Tensor logits = model.forward(passIds);
            int bestIdx = 0;
            int[] logitsShape = logits.shape();
            int lastRow = logitsShape[0] - 1;
            for (int j = 1; j < logitsShape[1]; j++) {
                if (logits.get(lastRow, j) > logits.get(lastRow, bestIdx)) {
                    bestIdx = j;
                } 
            }
            if (bestIdx == 50256) {
                done = true;
            } else {
                ids.add(bestIdx);
            }
        }
        return tokenizer.decode(ids.stream().mapToInt(Integer::intValue).toArray());
    }
}
