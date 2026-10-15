package openllm.generate;

import static org.junit.jupiter.api.Assertions.*;

import java.util.HashSet;
import java.util.Set;

import org.junit.jupiter.api.Test;

import openllm.backend.CpuBackend;
import openllm.tensors.Tensor;

public class SamplerTest {
    private final CpuBackend backend = new CpuBackend();

    // Max is at index 1, second highest at index 3, deliberately not in sorted order.
    private Tensor logits() {
        return Tensor.of(new float[] { 1f, 5f, 2f, 4f, 0f }, 1, 5);
    }

    @Test
    void temperatureZeroIsGreedy() {
        Sampler sampler = new Sampler(0f, 1, 42, backend);
        assertEquals(1, sampler.sample(logits()));
    }

    @Test
    void sameSeedGivesSameSequence() {
        Sampler a = new Sampler(1f, 5, 1234, backend);
        Sampler b = new Sampler(1f, 5, 1234, backend);
        for (int i = 0; i < 100; i++) {
            assertEquals(a.sample(logits()), b.sample(logits()), "draw " + i);
        }
    }

    @Test
    void topKOneMatchesGreedy() {
        Sampler greedy = new Sampler(0f, 1, 0, backend);
        Sampler topOne = new Sampler(1.5f, 1, 7, backend);
        int expected = greedy.sample(logits());
        for (int i = 0; i < 200; i++) {
            assertEquals(expected, topOne.sample(logits()), "draw " + i);
        }
    }

    @Test
    void topKOnlyReturnsTheKHighest() {
        Sampler sampler = new Sampler(1f, 2, 99, backend);
        Set<Integer> seen = new HashSet<>();
        for (int i = 0; i < 500; i++) {
            seen.add(sampler.sample(logits()));
        }
        assertEquals(Set.of(1, 3), seen);
    }

    @Test
    void drawFrequenciesFollowSoftmax() {
        // softmax([0, ln 3]) = [0.25, 0.75]
        Tensor twoWay = Tensor.of(new float[] { 0f, (float) Math.log(3) }, 1, 2);
        Sampler sampler = new Sampler(1f, 2, 2024, backend);
        int draws = 20000;
        int ones = 0;
        for (int i = 0; i < draws; i++) {
            if (sampler.sample(twoWay) == 1) {
                ones++;
            }
        }
        assertEquals(0.75, (double) ones / draws, 0.02);
    }

    @Test
    void doesNotModifyCallersLogits() {
        Tensor x = logits();
        new Sampler(1f, 2, 5, backend).sample(x);
        assertTrue(x.allClose(logits(), 0f));
    }

    @Test
    void rejectsBadArguments() {
        assertThrows(IllegalArgumentException.class, () -> new Sampler(-0.1f, 1, 0, backend));
        assertThrows(IllegalArgumentException.class, () -> new Sampler(1f, 0, 0, backend));
        Tensor twoRows = Tensor.of(new float[] { 1f, 2f, 3f, 4f }, 2, 2);
        assertThrows(IllegalArgumentException.class, () -> new Sampler(1f, 1, 0, backend).sample(twoRows));
    }
}
