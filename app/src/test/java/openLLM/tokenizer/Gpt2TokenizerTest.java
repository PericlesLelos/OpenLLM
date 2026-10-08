package openllm.tokenizer;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class Gpt2TokenizerTest {
    // Gradle runs tests from the app/ folder, so the model folder is one level up.
    private static final Path VOCAB = Path.of("../models/gpt2/vocab.json");
    private static final Path MERGES = Path.of("../models/gpt2/merges.txt");

    private static Gpt2Tokenizer tokenizer;

    @BeforeAll static void loadVocab() throws IOException {
        tokenizer = new Gpt2Tokenizer(VOCAB, MERGES);
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

    @Test void mergeRankFollowsLineOrder() {
        assertEquals(0, tokenizer.mergeRank("\u0120", "t"));   // "Ġ t" is the first merge
        assertEquals(2, tokenizer.mergeRank("h", "e"));
    }

    @Test void unknownPairHasNoRank() {
        assertEquals(-1, tokenizer.mergeRank("q", "z"));
    }

    @Test void preSplitKeepsTheLeadingSpaceWithTheWord() {
        assertEquals(List.of("Hello", " world"), tokenizer.preSplit("Hello world"));
    }

    @Test void preSplitSeparatesContractionsAndSymbols() {
        assertEquals(List.of("I", "'m", " here", "!!"), tokenizer.preSplit("I'm here!!"));
    }

    @Test void preSplitLeavesTheLastSpaceForTheNextWord() {
        assertEquals(List.of("a", " ", " b"), tokenizer.preSplit("a  b"));
    }

    @Test void preSplitHandlesDigitsPunctuationAndNewlines() {
        assertEquals(List.of("price", ":", " 42", " dollars", "\n"), tokenizer.preSplit("price: 42 dollars\n"));
    }

    @Test void bpeMergesACommonWordIntoOneToken() {
        assertEquals(List.of("Hello"), tokenizer.bpe("Hello"));
        assertEquals(List.of("\u0120world"), tokenizer.bpe("\u0120world"));   // "Ġworld"
    }

    @Test void bpeStopsWhenNoPairCanMerge() {
        assertEquals(List.of("\u0120token", "ization"), tokenizer.bpe("\u0120tokenization"));
        assertEquals(List.of("\u0120Open", "LL", "M"), tokenizer.bpe("\u0120OpenLLM"));
    }

    @Test void bpeOfASingleCharacterIsThatCharacter() {
        assertEquals(List.of("!"), tokenizer.bpe("!"));
    }

    @Test void encodeHelloWorld() {
        assertArrayEquals(new int[] {15496, 995}, tokenizer.encode("Hello world"));
    }

    @Test void encodeSplitsAnUnknownWordIntoSeveralTokens() {
        assertArrayEquals(new int[] {40, 1101, 2615, 4946, 3069, 44, 0},
                tokenizer.encode("I'm building OpenLLM!"));
    }

    @Test void encodeHandlesEmojiAndNewlines() {
        assertArrayEquals(new int[] {15496, 32485, 198}, tokenizer.encode("Hello \uD83D\uDE42\n"));   // "Hello 🙂\n"
    }

    @Test void encodeOfEmptyTextIsNoTokens() {
        assertArrayEquals(new int[] {}, tokenizer.encode(""));
    }

    // The strongest check of the whole tokenizer: any text must survive encode then decode unchanged.
    @Test void decodeUndoesEncode() {
        String[] samples = {
            "Hello world",
            "I'm building OpenLLM!",
            "  multiple   spaces\tand\ttabs\n\n",
            "na\u00EFve caf\u00E9 \u2014 \u65E5\u672C\u8A9E",   // "naïve café — 日本語"
            "price: $42.50 (approx.)",
        };
        for (String text : samples) {
            assertEquals(text, tokenizer.decode(tokenizer.encode(text)), text);
        }
    }
}
