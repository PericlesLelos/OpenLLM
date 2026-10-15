package openllm.model;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Path;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import openllm.backend.CpuBackend;
import openllm.tensors.Tensor;

public class Gpt2ModelTest {
    // Gradle runs tests from the app/ folder, so the model folder is one level up.
    private static final Path GPT2 = Path.of("../models/gpt2-small/model.safetensors");
    private static final float TOL = 1e-5f;

    // Loading reads the whole ~500 MB file, so do it once and share it across tests.
    private static Gpt2Weights weights;
    private static Gpt2Model model;

    @BeforeAll static void loadOnce() throws IOException {
        weights = Gpt2Weights.load(GPT2);
        model = new Gpt2Model(weights, new CpuBackend());
    }

    // ---- embed ----

    @Test void embedHasOneRowPerToken() {
        Tensor x = model.embed(new int[] {15496, 995});
        assertArrayEquals(new int[] {2, 768}, x.shape());
    }

    // "Hello world" = [15496, 995]. Expected values read from model.safetensors with Python:
    // wte[15496] + wpe[0] and wte[995] + wpe[1].
    @Test void embedMatchesReferenceValues() {
        Tensor x = model.embed(new int[] {15496, 995});
        assertEquals(-0.087471f, x.get(0, 0), TOL);
        assertEquals(-0.330112f, x.get(0, 1), TOL);
        assertEquals(0.015230f, x.get(0, 2), TOL);
        assertEquals(-0.124828f, x.get(1, 0), TOL);
        assertEquals(0.098074f, x.get(1, 1), TOL);
        assertEquals(-0.089250f, x.get(1, 2), TOL);
    }

    // Every row t must be token embedding + position embedding for position t.
    @Test void embedAddsTokenAndPositionRows() {
        int[] ids = {40, 1101, 2615, 4946};
        Tensor x = model.embed(ids);
        for (int t = 0; t < ids.length; t++) {
            float[] tok = weights.wte().row(ids[t]).getData();
            float[] pos = weights.wpe().row(t).getData();
            for (int j = 0; j < 768; j++) {
                assertEquals(tok[j] + pos[j], x.get(t, j), TOL, "t=" + t + " j=" + j);
            }
        }
    }

    // The same token at different positions gets different vectors, because wpe differs.
    @Test void sameTokenDiffersByPosition() {
        Tensor x = model.embed(new int[] {995, 995});
        assertFalse(x.row(0).allClose(x.row(1), TOL));
    }

    @Test void embedAcceptsFullContext() {
        Tensor x = model.embed(new int[1024]);
        assertArrayEquals(new int[] {1024, 768}, x.shape());
    }

    @Test void embedRejectsBadInput() {
        assertThrows(IllegalArgumentException.class, () -> model.embed(new int[0]));
        assertThrows(IllegalArgumentException.class, () -> model.embed(new int[1025]));
        assertThrows(IllegalArgumentException.class, () -> model.embed(new int[] {-1}));
        assertThrows(IllegalArgumentException.class, () -> model.embed(new int[] {50257}));
    }

    // ---- qKVProjection ----

    // Two tokens on purpose: with one token, a wrong row width in the split goes unnoticed.
    // Expected values computed from model.safetensors with numpy:
    // qkv = embed("Hello world") @ h.0.attn.c_attn.weight + h.0.attn.c_attn.bias, split into thirds.
    @Test void qkvProjectionMatchesReferenceValues() {
        Tensor x = model.embed(new int[] {15496, 995});
        Qkv qkv = model.qKVProjection(x, weights.layers()[0]);
        float tol = 1e-3f; // sums of 768 products, so allow a little float drift
        assertEquals(-0.40681f, qkv.q().get(0, 0), tol);
        assertEquals(-0.58026f, qkv.q().get(1, 0), tol);
        assertEquals(-1.98615f, qkv.q().get(1, 767), tol);
        assertEquals(-11.22215f, qkv.k().get(0, 0), tol);
        assertEquals(-8.23653f, qkv.k().get(1, 0), tol);
        assertEquals(5.15074f, qkv.k().get(1, 767), tol);
        assertEquals(0.48205f, qkv.v().get(0, 0), tol);
        assertEquals(0.05549f, qkv.v().get(1, 0), tol);
        assertEquals(0.64005f, qkv.v().get(1, 767), tol);
    }

    @Test void qkvProjectionShapes() {
        Tensor x = model.embed(new int[] {40, 1101, 2615});
        Qkv qkv = model.qKVProjection(x, weights.layers()[0]);
        assertArrayEquals(new int[] {3, 768}, qkv.q().shape());
        assertArrayEquals(new int[] {3, 768}, qkv.k().shape());
        assertArrayEquals(new int[] {3, 768}, qkv.v().shape());
    }

    // ---- columnSlice ----

    // x = [[0, 1, 2, 3, 4, 5],
    //      [6, 7, 8, 9, 10, 11]]
    private static Tensor counting2x6() {
        float[] data = new float[12];
        for (int i = 0; i < 12; i++) data[i] = i;
        return Tensor.of(data, 2, 6);
    }

    @Test void columnSliceFromTheStart() {
        Tensor s = model.columnSlice(counting2x6(), 0, 2);
        assertArrayEquals(new int[] {2, 2}, s.shape());
        assertArrayEquals(new float[] {0, 1, 6, 7}, s.getData());
    }

    @Test void columnSliceFromTheMiddle() {
        Tensor s = model.columnSlice(counting2x6(), 2, 3);
        assertArrayEquals(new int[] {2, 3}, s.shape());
        assertArrayEquals(new float[] {2, 3, 4, 8, 9, 10}, s.getData());
    }

    @Test void columnSliceToTheEnd() {
        Tensor s = model.columnSlice(counting2x6(), 4, 2);
        assertArrayEquals(new float[] {4, 5, 10, 11}, s.getData());
    }

    // A transposed tensor is a non-contiguous view; the slice must follow its logical layout.
    // transpose of [[0,1,2],[3,4,5]] is [[0,3],[1,4],[2,5]]; its column 1 is [3, 4, 5].
    @Test void columnSliceWorksOnAView() {
        Tensor t = Tensor.of(new float[] {0, 1, 2, 3, 4, 5}, 2, 3).transpose(0, 1);
        Tensor s = model.columnSlice(t, 1, 1);
        assertArrayEquals(new int[] {3, 1}, s.shape());
        assertArrayEquals(new float[] {3, 4, 5}, s.getData());
    }

    // Slicing every head and checking one value per head catches a wrong row width.
    @Test void columnSliceEveryHeadOfRealQ() {
        Tensor x = model.embed(new int[] {15496, 995});
        Tensor q = model.qKVProjection(x, weights.layers()[0]).q();
        for (int h = 0; h < 12; h++) {
            Tensor head = model.columnSlice(q, h * 64, 64);
            assertArrayEquals(new int[] {2, 64}, head.shape());
            assertEquals(q.get(1, h * 64 + 5), head.get(1, 5), 0f, "head " + h);
        }
    }

    // ---- causalMask ----

    @Test void causalMaskHidesOnlyFutureTokens() {
        float[] ones = new float[9];
        java.util.Arrays.fill(ones, 1f);
        Tensor m = model.causalMask(Tensor.of(ones, 3, 3));
        float ninf = Float.NEGATIVE_INFINITY;
        // Row i keeps columns 0..i; columns after i are the future.
        assertArrayEquals(new float[] {
            1, ninf, ninf,
            1, 1,    ninf,
            1, 1,    1
        }, m.getData());
    }

    @Test void causalMaskLeavesASingleTokenAlone() {
        Tensor m = model.causalMask(Tensor.of(new float[] {0.5f}, 1, 1));
        assertEquals(0.5f, m.get(0, 0));
    }

    // ---- attention ----

    // Expected values computed with numpy from model.safetensors for layer 0:
    // x = embed([15496, 995, 11]) ("Hello world,"), then QKV, 12 masked heads, merge, c_proj.
    @Test void attentionMatchesReferenceValues() {
        Tensor x = model.embed(new int[] {15496, 995, 11});
        Tensor a = model.attention(x, weights.layers()[0]);
        float tol = 1e-3f;
        assertArrayEquals(new int[] {3, 768}, a.shape());
        assertEquals(0.01187f, a.get(0, 0), tol);
        assertEquals(-0.45021f, a.get(0, 1), tol);
        assertEquals(0.23581f, a.get(0, 767), tol);
        assertEquals(0.32232f, a.get(1, 0), tol);
        assertEquals(1.72910f, a.get(1, 1), tol);
        assertEquals(0.33914f, a.get(1, 767), tol);
        assertEquals(3.41921f, a.get(2, 0), tol);
        assertEquals(1.76305f, a.get(2, 1), tol);
        assertEquals(-0.05696f, a.get(2, 767), tol);
    }

    // Causality: adding a later token must not change the outputs for earlier tokens.
    @Test void attentionIgnoresFutureTokens() {
        Tensor two = model.attention(model.embed(new int[] {15496, 995}), weights.layers()[0]);
        Tensor three = model.attention(model.embed(new int[] {15496, 995, 11}), weights.layers()[0]);
        for (int t = 0; t < 2; t++) {
            for (int j = 0; j < 768; j++) {
                assertEquals(two.get(t, j), three.get(t, j), 1e-4f, "t=" + t + " j=" + j);
            }
        }
    }

    // With one token, the only thing it can attend to is itself.
    @Test void attentionWorksForASingleToken() {
        Tensor a = model.attention(model.embed(new int[] {15496}), weights.layers()[0]);
        assertArrayEquals(new int[] {1, 768}, a.shape());
        assertEquals(0.01187f, a.get(0, 0), 1e-3f);
        assertEquals(0.23581f, a.get(0, 767), 1e-3f);
    }

    // ---- mlp ----

    // Expected values computed with numpy for layer 0: x = embed([15496, 995, 11]),
    // then c_fc + bias, gelu (tanh approximation), c_proj + bias.
    @Test void mlpMatchesReferenceValues() {
        Tensor x = model.embed(new int[] {15496, 995, 11});
        Tensor m = model.mlp(x, weights.layers()[0]);
        float tol = 1e-3f;
        assertArrayEquals(new int[] {3, 768}, m.shape());
        assertEquals(0.83019f, m.get(0, 0), tol);
        assertEquals(-0.71008f, m.get(0, 1), tol);
        assertEquals(-4.32848f, m.get(0, 767), tol);
        assertEquals(-3.44305f, m.get(1, 0), tol);
        assertEquals(0.82211f, m.get(1, 1), tol);
        assertEquals(0.73786f, m.get(1, 767), tol);
        assertEquals(-0.88727f, m.get(2, 0), tol);
        assertEquals(0.00458f, m.get(2, 1), tol);
        assertEquals(0.63149f, m.get(2, 767), tol);
    }

    // ---- block ----

    // Expected values computed with numpy for layer 0: x = embed([15496, 995, 11]),
    // x = x + attention(ln_1(x)), then x = x + mlp(ln_2(x)), layer norm eps 1e-5.
    @Test void blockMatchesReferenceValues() {
        Tensor x = model.embed(new int[] {15496, 995, 11});
        Tensor b = model.block(x, weights.layers()[0]);
        float tol = 1e-3f;
        assertArrayEquals(new int[] {3, 768}, b.shape());
        assertEquals(0.67837f, b.get(0, 0), tol);
        assertEquals(-1.45964f, b.get(0, 1), tol);
        assertEquals(-1.61765f, b.get(0, 767), tol);
        assertEquals(-1.81112f, b.get(1, 0), tol);
        assertEquals(0.25370f, b.get(1, 1), tol);
        assertEquals(0.43149f, b.get(1, 767), tol);
        assertEquals(-0.29746f, b.get(2, 0), tol);
        assertEquals(-0.30936f, b.get(2, 1), tol);
        assertEquals(0.93643f, b.get(2, 767), tol);
    }

    // The block keeps the shape [T, 768], so blocks can be chained.
    @Test void blockOutputCanFeedTheNextBlock() {
        Tensor x = model.embed(new int[] {15496, 995, 11});
        Tensor b = model.block(model.block(x, weights.layers()[0]), weights.layers()[1]);
        assertArrayEquals(new int[] {3, 768}, b.shape());
    }

    // Causality survives the whole block: a later token must not change earlier rows.
    @Test void blockIgnoresFutureTokens() {
        Tensor two = model.block(model.embed(new int[] {15496, 995}), weights.layers()[0]);
        Tensor three = model.block(model.embed(new int[] {15496, 995, 11}), weights.layers()[0]);
        for (int t = 0; t < 2; t++) {
            for (int j = 0; j < 768; j++) {
                assertEquals(two.get(t, j), three.get(t, j), 1e-4f, "t=" + t + " j=" + j);
            }
        }
    }

    // ---- forward ----

    private static int argmax(Tensor logits, int row) {
        int best = 0;
        for (int j = 1; j < logits.shape()[1]; j++) {
            if (logits.get(row, j) > logits.get(row, best)) {
                best = j;
            }
        }
        return best;
    }

    // Expected values computed with numpy from model.safetensors: embed, all 12 blocks,
    // ln_f, then x times wte transposed. Tolerance is looser because float error adds up over 12 layers.
    @Test void forwardMatchesReferenceLogits() {
        Tensor logits = model.forward(new int[] {15496, 995, 11});
        float tol = 0.05f;
        assertArrayEquals(new int[] {3, 50257}, logits.shape());
        assertEquals(-35.2362f, logits.get(0, 0), tol);
        assertEquals(-35.3265f, logits.get(0, 1), tol);
        assertEquals(-36.4580f, logits.get(0, 50256), tol);
        assertEquals(-68.7190f, logits.get(1, 0), tol);
        assertEquals(-75.0033f, logits.get(1, 50256), tol);
        assertEquals(-107.0183f, logits.get(2, 0), tol);
        assertEquals(-106.3205f, logits.get(2, 50256), tol);
    }

    // The top-scoring token per row: "Hello" -> "," (11), "Hello world" -> "," (11),
    // "Hello world," -> " I" (314).
    @Test void forwardPredictsTheReferenceNextTokens() {
        Tensor logits = model.forward(new int[] {15496, 995, 11});
        assertEquals(11, argmax(logits, 0));
        assertEquals(11, argmax(logits, 1));
        assertEquals(314, argmax(logits, 2));
    }
}
