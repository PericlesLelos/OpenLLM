package openllm.tensors;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorTest {
    @Test void readsRowMajorOrder() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        assertEquals(3f, t.get(0, 2));
        assertEquals(4f, t.get(1, 0));
        assertEquals(6f, t.get(1, 2));
    }

    @Test void rejectsDataThatDoesNotFitShape() {
        assertThrows(IllegalArgumentException.class, 
            () -> Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 2));        
    }

    @Test void reportsShapeRankAndSize() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        assertArrayEquals(new int[] {2, 3}, t.shape());
        assertEquals(6, t.size());
        assertEquals(2, t.rank());
    }

    @Test void setThenGet() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        t.set(10f, 0, 0);
        assertEquals(10f, t.get(0, 0));
        assertEquals(5f, t.get(1, 1));
    }

    @Test void rejectGetWithBadDimensions() {
        assertThrows(IllegalArgumentException.class,
            () -> {
                Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
                t.get(1, 2, 3);
            });
    }

    @Test void rejectOutOfRangeIndex() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        assertThrows(IndexOutOfBoundsException.class, () -> t.get(2, 0));   
        assertThrows(IndexOutOfBoundsException.class, () -> t.get(0, 3));   
        assertThrows(IndexOutOfBoundsException.class, () -> t.get(-1, 0));  
    }

    @Test void rejectBadZerosDimensions() {
        assertThrows(IllegalArgumentException.class, () -> Tensor.zeros(2, 0));
        assertThrows(IllegalArgumentException.class, () -> Tensor.zeros(2, -1));
    }
}
