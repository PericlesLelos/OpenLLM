package openllm.io;

public record TensorInfo(String name, String dtype, int[] shape, long start, long end){}


