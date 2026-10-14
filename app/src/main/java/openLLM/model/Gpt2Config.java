package openllm.model;

/**
 * The hyperparameters that fix a GPT-2 model's size.
 *
 * @param nLayer number of transformer blocks
 * @param nHead number of attention heads per block
 * @param nEmbd embedding width: the length of each token's vector
 * @param vocabSize number of tokens in the vocabulary
 * @param nCtx context length: the most tokens the model can see at once
 */
public record Gpt2Config(int nLayer, int nHead, int nEmbd, int vocabSize, int nCtx){
    /** GPT-2 small (124M parameters): 12 layers, 12 heads, width 768. */
    public static final Gpt2Config GPT2_SMALL = new Gpt2Config(12, 12, 768, 50257, 1024);
}
