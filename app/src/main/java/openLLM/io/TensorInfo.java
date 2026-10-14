package openllm.io;

/**
 * One tensor's entry in a safetensors header: everything needed to find and
 * load its data, but not the data itself.
 *
 * @param name the tensor's name in the file, e.g. {@code "wte.weight"}
 * @param dtype the element type, e.g. {@code "F32"}
 * @param shape the size of each dimension
 * @param start first byte of the data, relative to the end of the header
 * @param end one past the last byte of the data, relative to the end of the header
 */
public record TensorInfo(String name, String dtype, int[] shape, long start, long end){}


