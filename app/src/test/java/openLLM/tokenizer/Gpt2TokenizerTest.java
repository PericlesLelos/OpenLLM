package openllm.tokenizer;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class Gpt2TokenizerTest {
    // Gradle runs tests from the app/ folder, so the model folder is one level up.
    private static final Path VOCAB = Path.of("../models/gpt2/vocab.json");

    private static Gpt2Tokenizer tokenizer;

    @BeforeAll static void loadVocab() throws IOException {
        tokenizer = new Gpt2Tokenizer(VOCAB);
    }

    @Test void printableBytesMapToThemselves() {
        char[] table = Gpt2Tokenizer.buildByteToChar();
        assertEquals('A', table['A']);
        assertEquals('!', table['!']);
        assertEquals('~', table['~']);
    }

    @Test void spaceMapsToGWithDot() {
        char[] table = Gpt2Tokenizer.buildByteToChar();
        assertEquals('\u0120', table[' ']);   // 'Ġ'
    }

    @Test void newlineMapsToCWithDot() {
        char[] table = Gpt2Tokenizer.buildByteToChar();
        assertEquals('\u010A', table['\n']);  // 'Ċ'
    }

    @Test void allByteCharactersAreDistinct() {
        char[] table = Gpt2Tokenizer.buildByteToChar();
        Set<Character> seen = new HashSet<>();
        for (char c : table) {
            seen.add(c);
        }
        assertEquals(256, seen.size());
    }

    @Test void charToByteIsTheExactReverse() {
        char[] table = Gpt2Tokenizer.buildByteToChar();
        Map<Character, Integer> reverse = Gpt2Tokenizer.buildCharToByte(table);
        assertEquals(256, reverse.size());
        for (int b = 0; b < 256; b++) {
            assertEquals(b, reverse.get(table[b]), "byte " + b);
        }
    }

    @Test void vocabHasOneEntryPerEmbeddingRow() {
        assertEquals(50257, tokenizer.vocabSize());
    }

    @Test void tokenToIdFindsKnownTokens() {
        assertEquals(15496, tokenizer.tokenToId("Hello"));
        assertEquals(995, tokenizer.tokenToId("\u0120world"));   // "Ġworld" = " world"
        assertEquals(50256, tokenizer.tokenToId("<|endoftext|>"));
    }

    @Test void idToTokenIsTheReverse() {
        assertEquals("Hello", tokenizer.idToToken(15496));
        assertEquals("\u0120world", tokenizer.idToToken(995));
    }

    @Test void unknownTokenOrIdThrows() {
        assertThrows(IllegalArgumentException.class, () -> tokenizer.tokenToId("notARealToken!!"));
        assertThrows(IllegalArgumentException.class, () -> tokenizer.idToToken(50257));
    }

    @Test void decodeJoinsTokensIntoText() {
        assertEquals("Hello world", tokenizer.decode(new int[] {15496, 995}));
    }

    @Test void decodeOfNoTokensIsEmpty() {
        assertEquals("", tokenizer.decode(new int[] {}));
    }

    @Test void decodeRejoinsACharacterSplitAcrossTokens() {
        // The emoji is 4 UTF-8 bytes split across all three tokens, so decode must
        // collect all the bytes before turning them into text.
        assertEquals(" \uD83D\uDE42", tokenizer.decode(new int[] {12520, 247, 224}));   // " 🙂"
    }
}
