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

/**
 * GPT-2's byte-level BPE tokenizer: turns text into token ids and back.
 *
 * <p>Encoding has four stages:
 * <ol>
 *   <li>Pre-split the text into word-like chunks with a regex.</li>
 *   <li>Turn each chunk into UTF-8 bytes and map each byte to a printable
 *       stand-in character, so every string can be represented.</li>
 *   <li>Run BPE: repeatedly merge the adjacent pair with the best (lowest)
 *       rank from {@code merges.txt}.</li>
 *   <li>Look up each resulting piece in {@code vocab.json}.</li>
 * </ol>
 * Decoding reverses the byte mapping and decodes the bytes as UTF-8.
 */
public class Gpt2Tokenizer {
    
    /** Byte value (0-255) to its stand-in character. See {@link #buildByteToChar()}. */
    private static final char[] BYTE_TO_CHAR = buildByteToChar();
    /** The inverse of {@link #BYTE_TO_CHAR}. */
    private static final Map<Character, Integer> CHAR_TO_BYTE = buildCharToByte(BYTE_TO_CHAR);
    /**
     * GPT-2's pre-split pattern: common English contractions, runs of letters,
     * runs of digits, runs of other symbols (each optionally with one leading
     * space), and whitespace.
     */
    private static final Pattern PRE_SPLIT = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");
    /** Token string to id, from {@code vocab.json}. */
    private final Map<String, Integer> tokenToId = new HashMap<>();
    /** Id to token string, the reverse of {@link #tokenToId}. */
    private final Map<Integer, String> idToToken = new HashMap<>();
    /** Merge pair (two pieces joined by a space) to its rank; lower ranks merge first. */
    private final Map<String, Integer> mergeRanks = new HashMap<>();

    /**
     * Loads the vocabulary and merge rules.
     *
     * @param vocabPath path to {@code vocab.json} (token string to id)
     * @param mergesPath path to {@code merges.txt}; the first line is a version header, and each later line is a merge whose rank is its position
     * @throws IOException if either file can't be read
     */
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

    /**
     * Returns the rank of merging {@code a} and {@code b}, or -1 if that pair
     * is not a merge rule.
     */
    int mergeRank(String a, String b) {
        Integer rank = mergeRanks.get(a + " " + b);
        return rank == null ? -1 : rank;
    }

    /**
     * @return the number of tokens in the vocabulary (50,257 for GPT-2)
     */
    public int vocabSize(){
        return tokenToId.size();
    }

    /**
     * @param token a token string in byte-mapped form, e.g. {@code "Ġworld"} for " world"
     * @return its id
     * @throws IllegalArgumentException if the token is not in the vocabulary
     */
    public int tokenToId(String token) {
        Integer out = tokenToId.get(token);
        if (out == null) { throw new IllegalArgumentException("Unknown token: " + token); }
        return out;
    }

    /**
     * @param id a token id
     * @return its token string, in byte-mapped form
     * @throws IllegalArgumentException if the id is not in the vocabulary
     */
    public String idToToken(int id) {
        String out = idToToken.get(id);
        if(out == null) { throw new IllegalArgumentException("Unknown id: " + id); }
        return out;
    }

    /**
     * Turns token ids back into text. All bytes are collected first and decoded
     * as UTF-8 once at the end, so characters split across tokens (such as
     * emoji) come out whole.
     *
     * @param ids token ids
     * @return the decoded text
     * @throws IllegalArgumentException if an id is not in the vocabulary
     */
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
    
    /**
     * Builds GPT-2's byte-to-character table. Printable bytes (33-126, 161-172,
     * 174-255) map to themselves. The other 68 bytes (control characters, space
     * and a few others) map to {@code 256 + n} in order, so space becomes
     * {@code Ġ} (U+0120) and newline becomes {@code Ċ} (U+010A).
     */
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
    
    /** Builds the reverse of {@code table}: stand-in character to byte value. */
    static Map<Character, Integer> buildCharToByte(char[] table) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < 256; i++) {
            map.put(table[i], i);
        }
        return map;
    }

    /** Splits text into chunks with {@link #PRE_SPLIT}. BPE never merges across chunks. */
    List<String> preSplit(String text) {
        List<String> chunks = new ArrayList<>();
        Matcher m = PRE_SPLIT.matcher(text); 
        while (m.find()) { chunks.add(m.group()); }
        return chunks;
    }

    /**
     * Runs byte-pair encoding on one byte-mapped chunk. Starts with one piece
     * per character, then repeatedly merges the adjacent pair with the lowest
     * rank until no pair is a merge rule.
     *
     * @param word a chunk already converted with {@link #BYTE_TO_CHAR}
     * @return the final pieces, each of which is a vocabulary token
     */
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

    /**
     * Turns text into GPT-2 token ids.
     *
     * @param s the text
     * @return token ids; empty for an empty string
     */
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
