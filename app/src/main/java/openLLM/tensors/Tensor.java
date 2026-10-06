package openllm.tensors;

import java.util.Arrays;

/**
 * An n-dimensional array of floats, stored in one flat {@code float[]}.
 *
 * <p>Elements are laid out in row-major order: the last dimension changes
 * fastest. Each dimension has a stride, the number of slots to jump in the
 * flat array to move one step along that dimension. The element at index
 * {@code [i0, i1, ...]} lives at {@code i0*strides[0] + i1*strides[1] + ...}.
 *
 * <p>Some methods return <em>views</em> that share this tensor's data
 * ({@link #reshape}, {@link #transpose}), so writing through a view changes
 * the original. Others return independent <em>copies</em>
 * ({@link #contiguous} when a copy is needed, {@link #row}). Each method's
 * documentation says which.
 */
public final class Tensor {
    private final float[] data;
    private final int[] shape, strides;

    private Tensor(float[] data, int[] shape) {
        this(data, shape, rowMajorStrides(shape));
    }

    private Tensor(float[]data, int[] shape, int[] strides) {
        this.data = data;
        this.shape = shape.clone();
        this.strides = strides;
    }

    // Strides for a densely packed row-major layout: the last dimension has stride 1, and each earlier one is the product of the sizes after it.
    private static int[] rowMajorStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    /**
     * Returns the element at the given index.
     *
     * @param index one index per dimension, e.g. {@code get(1, 2)} for a 2-D tensor
     * @return the element at that position
     * @throws IllegalArgumentException if the number of indices does not match {@link #rank()}
     * @throws IndexOutOfBoundsException if any index is negative or past the end of its dimension
     */
    public float get(int... index) {
        return this.data[offset(index)];
    }

    /**
     * Returns a copy of the underlying storage array.
     *
     * <p>Values are in storage order, which matches logical row-major order only
     * when the tensor is {@linkplain #isContiguous() contiguous}. Call
     * {@code contiguous().getData()} to be sure of the order.
     *
     * @return a new array; changing it does not affect this tensor
     */
    public float[] getData() {
        return this.data.clone();
    }

    /**
     * Returns a copy of a range of the underlying storage array.
     *
     * <p>Like {@link #getData()}, values are in storage order.
     *
     * @param start first storage position to copy (inclusive)
     * @param end position to stop at (exclusive); should not exceed {@link #size()},
     *            or the result is padded with zeros
     * @return a new array of length {@code end - start}
     */
    public float[] getData(int start, int end) {
        return Arrays.copyOfRange(this.getData(), start, end);
    }

    /**
     * Writes a value at the given index. If this tensor is a view, the
     * original tensor sees the change too.
     *
     * @param value the value to store
     * @param index one index per dimension
     * @throws IllegalArgumentException if the number of indices does not match {@link #rank()}
     * @throws IndexOutOfBoundsException if any index is out of range
     */
    public void set(float value, int... index) {
        this.data[offset(index)] = value;
    }

    // Turns a multi-dimensional index into a position in data: the sum of index[d] * strides[d].
    private int offset(int[] index) {
        if(index.length != shape.length) {
            throw new IllegalArgumentException("supplied index dimensions don't match tensor shape:\n index: " + Arrays.toString(index) + "\n shape: " + Arrays.toString(shape));
        }

        int offset = 0;
        for (int i = 0; i < index.length; i++) {
            if (index[i] < 0 || index[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("supplied index < 0 or greater than shape[i]: i = " + i);
            }
            offset += index[i] * this.strides[i];
        }
        return offset;
    }

    private static int sizeOf(int[] shape) {
        for(int dimension : shape) {
            if (dimension <= 0) {
                throw new IllegalArgumentException("Tensor shape must contain positive integers: " + Arrays.toString(shape));
            }
        }

        int size = 1;
        for (int i = 0; i < shape.length; i++) {
            size *= shape[i];
        }
        return size;
    }

    /**
     * Creates a tensor of the given shape filled with zeros.
     *
     * @param shape the size of each dimension, e.g. {@code zeros(2, 3)}
     * @return a new contiguous tensor
     * @throws IllegalArgumentException if any dimension is zero or negative
     */
    public static Tensor zeros(int... shape) {
        float[] data = new float[sizeOf(shape)];
        return new Tensor(data, shape);
    }

    /**
     * Creates a tensor from values listed in row-major order.
     *
     * <p>The array is copied, so later changes to {@code data} do not affect the tensor.
     *
     * @param data the values, row by row
     * @param shape the size of each dimension
     * @return a new contiguous tensor
     * @throws IllegalArgumentException if any dimension is zero or negative, or if
     *         {@code data.length} is not the product of the dimensions
     */
    public static Tensor of(float[] data, int... shape) {
        int expected = sizeOf(shape);
        if (data.length != expected) {
            throw new IllegalArgumentException("data.length != expected: \n data.length = " + data.length + " \n expected = " + expected);
        }
        return new Tensor(data.clone(), shape);
    }

    /**
     * Returns the size of each dimension.
     *
     * @return a copy of the shape array, e.g. {@code [2, 3]}
     */
    public int[] shape() { return this.shape.clone(); }
    
    /**
     * Returns the number of dimensions.
     *
     * @return 1 for a vector, 2 for a matrix, and so on
     */
    public int rank() { return this.shape.length; }
    
    /**
     * Returns the total number of elements.
     *
     * @return the product of all dimensions
     */
    public int size() { return this.data.length; }

    /**
     * Returns a view of the same data with a new shape. No values are copied.
     *
     * <p>Only works on contiguous tensors. For a transposed tensor, call
     * {@code contiguous().reshape(...)} instead.
     *
     * @param newShape the new dimensions; must hold the same number of elements
     * @return a view that shares this tensor's data
     * @throws IllegalArgumentException if {@code newShape} holds a different number of elements
     * @throws IllegalStateException if this tensor is not contiguous
     */
    public Tensor reshape(int... newShape) {
        if(sizeOf(this.shape) != sizeOf(newShape)) {
            throw new IllegalArgumentException("newShape must be the same size as current shape");
        } 
        if(!isContiguous()) {
            throw new IllegalStateException("tensor must be contiguous to call reshape");
        }
        return new Tensor(this.data, newShape);
    }

    /**
     * Returns a view with two dimensions swapped. No values are copied: only
     * the shape and strides are swapped.
     *
     * <p>For a matrix, {@code transpose(0, 1)} is the usual transpose. For
     * multi-head attention, swapping dims 0 and 1 of a
     * {@code [tokens, heads, headDim]} tensor gives {@code [heads, tokens, headDim]}.
     *
     * @param dimA the first dimension to swap
     * @param dimB the second dimension to swap
     * @return a view that shares this tensor's data, usually not contiguous
     * @throws IllegalArgumentException if either dimension is outside {@code [0, rank - 1]}
     */
    public Tensor transpose(int dimA, int dimB) {
        if(dimA < 0 || dimB < 0 || dimA > this.rank() - 1 || dimB > this.rank() - 1) {
            throw new IllegalArgumentException("dimensions must be between [0, rank - 1]");
        }
        int[] newShape = this.shape.clone();
        int[] newStrides = this.strides.clone();
        swap(newShape, dimA, dimB);
        swap(newStrides, dimA, dimB);
        return new Tensor(this.data, newShape, newStrides);
        
    }

    private static void swap(int[] array, int i, int j) {
        int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }

    /**
     * Reports whether the data is laid out in plain row-major order, so that
     * walking the storage array visits elements in logical order.
     *
     * @return true for freshly created and reshaped tensors; usually false after {@link #transpose}
     */
    public boolean isContiguous() { return (Arrays.equals(this.strides, rowMajorStrides(this.shape))); }
    
    /**
     * Returns a tensor with the same shape and values, stored in row-major order.
     *
     * @return this tensor if it is already contiguous; otherwise a new
     *         independent copy with its values rearranged
     */
    public Tensor contiguous() {
        if(this.isContiguous()) {return this; }
        int[] rm = rowMajorStrides(this.shape);
        float[] out = new float[this.size()];
        int[] index = new int[this.rank()];

        for (int i = 0; i < this.size(); i++) {
            for (int j = 0; j < this.rank(); j++) {
                // turn flat position i into a row-major index, like seconds into h:m:s
                index[j] = (i / rm[j]) % shape[j];
            }
            out[i] = get(index);
        }
        return new Tensor(out, shape);
    }

    /**
     * Returns one row of a matrix as a 1-D tensor, for example a token's
     * embedding from an embedding table.
     *
     * @param i the row index
     * @return a new independent tensor of shape {@code [columns]}
     * @throws IllegalStateException if this tensor is not 2-D
     * @throws IndexOutOfBoundsException if {@code i} is outside {@code [0, rows - 1]}
     */
    public Tensor row(int i) {
        if(this.rank() != 2) {
            throw new IllegalStateException("Rank must be 2 to select a row");
        }
        if(i < 0 || i >= this.shape[0]) {
            throw new IndexOutOfBoundsException("Index must fall within number of rows");
        }
        float[] row = new float[this.shape[1]];
        for (int j = 0; j < this.shape[1]; j++) {
            row[j] = this.get(i, j);
        }
        return Tensor.of(row, this.shape[1]);
    }

    /**
     * Returns a readable summary: the shape, then up to the first 10 values in
     * logical order, with {@code ...} when there are more.
     * For example: {@code Tensor: [2, 3] [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}.
     */
    @Override public String toString() {
        Tensor c = this.contiguous();
        StringBuilder sb = new StringBuilder();
        sb.append("Tensor: ");
        sb.append(Arrays.toString(c.shape));
        sb.append(" [");
        for (int i = 0; i < Math.min(this.size(), 10); i++) {
            if(i > 0) {sb.append(", "); }
            sb.append(c.data[i]);
        }
        if (this.size() > 10) {sb.append(", ...");}
        sb.append("]");
        return sb.toString();
        
    }

    /**
     * Checks whether two tensors have the same shape and every pair of values
     * differs by at most {@code tolerance}. Use this instead of exact equality
     * when comparing float results.
     *
     * <p>A {@code NaN} anywhere makes the result false.
     *
     * @param other the tensor to compare against
     * @param tolerance the largest allowed absolute difference per element
     * @return true if the shapes match and every element is within tolerance
     */
    public boolean allClose(Tensor other, float tolerance) {
        if (!Arrays.equals(this.shape, other.shape)) { return false; }
        Tensor t1C = this.contiguous();
        Tensor t2C = other.contiguous();
        for (int i = 0; i < this.size(); i++) {
            if (!(Math.abs(t1C.data[i] - t2C.data[i]) <= tolerance)) { return false; }
        }
        return true;
    }
}
