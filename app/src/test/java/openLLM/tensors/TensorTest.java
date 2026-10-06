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

    @Test void tensorReshapesToNewShape() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = t.reshape(3, 2);
        t1.set(0,1, 1);
        assertArrayEquals(new int[] {3, 2}, t1.shape());
        assertEquals(3f, t1.get(1, 0));
        assertEquals(0f, t.get(1, 0));

    }

    @Test void rejectsReshapeWithWrongSize() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        assertThrows(IllegalArgumentException.class, 
            () -> t.reshape(3, 3)
        );
    }

    @Test void tensorTransformsToNewDimensions() {
        Tensor t = Tensor.of(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        t = t.transpose(0, 1);
        assertEquals(4f, t.get(0, 1));
        assertEquals(3f, t.get(2, 0));
    }

    @Test void tensorPreservesEditWhenTransformed() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = t.transpose(0, 1);
        t1.set(0,2, 1);
        assertEquals(0, t.get(1, 2));
    }

    @Test void rejectTransposeWithBadDimensions() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);

        assertThrows(IllegalArgumentException.class,
            () -> t.transpose(0, 2));
        assertThrows(IllegalArgumentException.class,
            () -> t.transpose(3, 0));
    }

    @Test void newTensorsAreContiguous() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        assertTrue(t.isContiguous());
    }

    @Test void transposedTensorsAreNotContiguous() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = t.transpose(0, 1);
        assertFalse(t1.isContiguous());
    }

    @Test void rejectReshapeOnTransposedTensors() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = t.transpose(0, 1);
        assertThrows(IllegalStateException.class,
            () -> t1.reshape(3, 2));
    }

    @Test void contiguousCopiesTransposeInRowMajorOrder() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3).transpose(0, 1);
        assertArrayEquals(new float[] {1, 4, 2, 5, 3, 6}, t.contiguous().getData());

    }

    @Test void contiguousReturnsIndependentCopy() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor tT = t.transpose(0,1);
        Tensor tC = tT.contiguous();
        tC.set(0, 0, 0);
        assertEquals(1f, tT.get(0, 0));
    }

    @Test void rowReturnsSelectedRow() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        assertArrayEquals(new float[] {4, 5, 6}, t.row(1).getData());
    }

    @Test void rowWorksWithTransposedTensor() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = t.transpose(0, 1);
        assertArrayEquals(new float[] {1, 4}, t1.row(0).getData());
    }

    @Test void setOnRowDoesntChangeInitial() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = t.row(0);
        t1.set(0,0);
        assertEquals(1f, t.get(0, 0));
    }

    @Test void rejectRowOnOneDTensor() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 6);
        assertThrows(IllegalStateException.class,
            () -> t.row(0));
    }

    @Test void rejectRowWhenIndexOutOfBounds() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        assertThrows(IndexOutOfBoundsException.class,
            () -> t.row(2));
    }

    @Test void toStringReturnsStringOfTensor() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        assertEquals("Tensor: [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]", t.toString());
    }
    
    @Test void toStringReturnsStringOfTransposedTensor() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = t.transpose(0, 1);
        assertEquals("Tensor: [3, 2] [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]", t1.toString());
    }

    @Test void toStringReturnsTruncatedIfLongerThanTen() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 3, 4);
        assertEquals("Tensor: [3, 4] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, ...]", t.toString());
    }

    @Test void identicalTensorsAreClose() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        assertTrue(t.allClose(t1, 0.00001f));
    }

    @Test void smallDifferencesAreClose() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = Tensor.of(new float[] {1.00001f, 2.00001f, 3.00001f, 4.00001f, 5.00001f, 6.00001f}, 2, 3);
        assertTrue(t.allClose(t1, 0.001f));
    }

    @Test void bigDifferencesArentClose() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = Tensor.of(new float[] {1.001f, 2.001f, 3.001f, 4.001f, 5.001f, 6.001f}, 2, 3);
        assertFalse(t.allClose(t1, 0.00001f));
    }

    @Test void differentShapesArentClose() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 3, 2);
        assertFalse(t.allClose(t1, 0.001f));
    }

    @Test void transposeIsClose() {
        Tensor t = Tensor.of(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor t1 = t.transpose(0, 1);
        Tensor tT = Tensor.of(new float[] {1, 4, 2, 5, 3, 6}, 3, 2);
        assertTrue(t1.allClose(tT, 0.001f));
    }
}
