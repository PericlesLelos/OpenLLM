package openllm.tokenizer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

public class Gpt2Tokenizer {
    
    private static final char[] BYTE_TO_CHAR = buildByteToChar();
    private static final Map<Character, Integer> CHAR_TO_BYTE = buildCharToByte(BYTE_TO_CHAR);
    private static final Pattern PRE_SPLIT = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
    private final Map<String, Integer> tokenToId = new HashMap<>();
    private final Map<Integer, String> idToToken = new HashMap<>();
    private final Map<String, Integer> mergeRanks = new HashMap<>();

    public Gpt2Tokenizer(Path vocabPath, Path mergesPath) throws IOException {
        String vocab = Files.readString(vocabPath);
        JsonObject vocabJson = JsonParser.parseString(vocab).getAsJsonObject();
        for (Map.Entry<String, JsonElement> e : vocabJson.entrySet()) {
            tokenToId.put(e.getKey(), e.getValue().getAsInt());
            idToToken.put(e.getValue().getAsInt(), e.getKey());
        }
        List<String> lines = Files.readAllLines(mergesPath);
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i);
            if (line.isBlank()) { continue; }
            mergeRanks.put(line, i - 1);
        }
    }

    int mergeRank(String a, String b) {
        Integer rank = mergeRanks.get(a + " " + b);
        return rank == null ? -1 : rank;
    }

    public int vocabSize(){
        return tokenToId.size();
    }

    public int tokenToId(String token) {
        Integer out = tokenToId.get(token);
        if (out == null) { throw new IllegalArgumentException("Unknown token: " + token); }
        return out;
    }

    public String idToToken(int id) {
        String out = idToToken.get(id);
        if(out == null) { throw new IllegalArgumentException("Unknown id: " + id); }
        return out;
    }

    public String decode(int[] ids) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        for (int id : ids) {
            String token = idToToken(id);
            for (char c : token.toCharArray()) {
                out.write(CHAR_TO_BYTE.get(c));
            }            
        }
        return out.toString(StandardCharsets.UTF_8);
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

    List<String> preSplit(String text) {
        List<String> chunks = new ArrayList<>();
        Matcher m = PRE_SPLIT.matcher(text); 
        while (m.find()) { chunks.add(m.group()); }
        return chunks;
    }

    List<String> bpe(String word) {
        List<String> parts = new ArrayList<>();
        for (char c : word.toCharArray()) {
            parts.add(String.valueOf(c));
        }    
        while (true) {
            int bestRank = Integer.MAX_VALUE;
            int bestIndex = -1;
            for (int i = 0; i < parts.size() - 1; i++) {
                int rank = mergeRank(parts.get(i), parts.get(i + 1));
                if (rank < bestRank && rank >= 0) {
                    bestRank = rank;
                    bestIndex = i;
                }               
            }
            if (bestIndex == -1) {
                    return parts;
            }
            parts.set(bestIndex, parts.get(bestIndex) + parts.get(bestIndex + 1));
            parts.remove(bestIndex + 1);
        }
    }

    public int[] encode(String s) {
        List<Integer> out = new ArrayList<>();
        for (String chunk : preSplit(s)) {
            byte[] bytes = chunk.getBytes(StandardCharsets.UTF_8);
            StringBuilder sb = new StringBuilder();
            for (byte b : bytes) {
                int i = b & 0xFF;
                sb.append(BYTE_TO_CHAR[i]);
            }
            String chars = sb.toString();
            List<String> parts = bpe(chars);
            for (String part : parts) {
                out.add(tokenToId(part));
            }

        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }
}
