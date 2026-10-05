package openllm.tensors;

import java.util.Arrays;

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

    private static int[] rowMajorStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    public float get(int... index) {
        return this.data[offset(index)];
    }

    public float[] getData() {
        return this.data.clone();
    }

    public void set(float value, int... index) {
        this.data[offset(index)] = value;
    }

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

    public static Tensor zeros(int... shape) {
        float[] data = new float[sizeOf(shape)];
        return new Tensor(data, shape);
    }

    public static Tensor of(float[] data, int... shape) {
        int expected = sizeOf(shape);
        if (data.length != expected) {
            throw new IllegalArgumentException("data.length != expected: \n data.length = " + data.length + " \n expected = " + expected);
        }
        return new Tensor(data.clone(), shape);
    }

    public int[] shape() { return this.shape.clone(); }
    public int rank() { return this.shape.length; }
    public int size() { return this.data.length; }

    public Tensor reshape(int... newShape) {
        if(sizeOf(this.shape) != sizeOf(newShape)) {
            throw new IllegalArgumentException("newShape must be the same size as current shape");
        } 
        if(!isContiguous()) {
            throw new IllegalStateException("tensor must be contiguous to call reshape");
        }
        return new Tensor(this.data, newShape);
    }

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

    public boolean isContiguous() { return (Arrays.equals(this.strides, rowMajorStrides(this.shape))); }
    
    public Tensor contiguous() {
        if(this.isContiguous()) {return this; }
        int[] rm = rowMajorStrides(this.shape);
        float[] out = new float[this.size()];
        int[] index = new int[this.rank()];

        for (int i = 0; i < this.size(); i++) {
            for (int j = 0; j < this.rank(); j++) {
                index[j] = (i / rm[j]) % shape[j];
            }
            out[i] = get(index);
        }

        return new Tensor(out, shape);

    }
}
