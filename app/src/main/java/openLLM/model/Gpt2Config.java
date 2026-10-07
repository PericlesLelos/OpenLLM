package openllm.model;

public record Gpt2Config(int nLayer, int nHead, int nEmbd, int vocabSize, int nCtx){
    public static final Gpt2Config GPT2_SMALL = new Gpt2Config(12, 12, 768, 50257, 1024);
}
