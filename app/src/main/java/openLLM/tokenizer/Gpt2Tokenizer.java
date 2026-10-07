package openllm.tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class Gpt2Tokenizer {
    
    private static final char[] BYTE_TO_CHAR = buildByteToChar();
    private static final Map<Character, Integer> CHAR_TO_BYTE = buildCharToByte(BYTE_TO_CHAR);
    private Map<String, Integer> tokenToId = new HashMap<>();
    private Map<Integer, String> idToToken = new HashMap<>();

    public Gpt2Tokenizer(Path vocabPath) throws IOException {
        String vocab = Files.readString(vocabPath);
        JsonObject vocabJson = JsonParser.parseString(vocab).getAsJsonObject();
        for (Map.Entry<String, JsonElement> e : vocabJson.entrySet()) {
            tokenToId.put(e.getKey(), e.getValue().getAsInt());
            idToToken.put(e.getValue().getAsInt(), e.getKey());
        }
    }

    public int vocabSize(){
        return tokenToId.size();
    }

    public int tokenToId(String token) {
        Integer out = tokenToId.get(token);
        if (out == null) { throw new IllegalArgumentException("Unknown token: " + token); }
        return out;
    }

    public String idToToken(Integer id) {
        String out = idToToken.get(id);
        if(out == null) { throw new IllegalArgumentException("Unknown id: " + id); }
        return out;
    }
    
    static char[] buildByteToChar() {
        char[] table = new char[256];
        int n = 0;
        for (int i = 0; i < 256; i++) {
            if(
                (i >= 33 && i <= 126) ||
                (i >= 161 && i <= 172) ||
                (i >= 174 && i <= 255)
            ) {
                table[i] = (char) i;

            } else {
                table[i] = (char) (n + 256);
                n++;
            }

        }
        return table;
    }
    static Map<Character, Integer> buildCharToByte(char[] table) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < 256; i++) {
            map.put(table[i], i);
        }
        return map;
    }


}
