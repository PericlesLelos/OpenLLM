package openllm.tokenizer;

public interface Tokenizer {
    public int[] encode(String s);
    public String decode(int[] ids);
    public int eosId();
}
