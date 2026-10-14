package openllm.io;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.io.IOException;
import com.google.gson.JsonParser;

import openllm.tensors.Tensor;

import com.google.gson.JsonObject;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;

import java.util.HashMap;
import java.util.Map;

/**
 * Reads tensors from a {@code .safetensors} file.
 *
 * <p>File layout: an 8-byte little-endian header length {@code N}, then
 * {@code N} bytes of UTF-8 JSON describing each tensor (dtype, shape and
 * byte offsets), then the raw tensor data. Offsets in the header are
 * relative to the start of the data section, i.e. to byte {@code 8 + N}.
 */
public class SafetensorReader {
    
    /**
     * Parses the JSON header and describes every tensor in the file, without
     * loading any tensor data. The {@code __metadata__} entry is skipped.
     *
     * @param p path to the {@code .safetensors} file
     * @return a map from tensor name (e.g. {@code "h.0.attn.c_attn.weight"}) to its {@link TensorInfo}
     * @throws IOException if the file can't be read
     */
    public static Map<String, TensorInfo> readHeader(Path p) throws IOException {
        try (InputStream in = Files.newInputStream(p)) {
            in.skipNBytes(8);
            Map<String, TensorInfo> tensorsByName = new HashMap<>();
            long headerLength = readHeaderLength(p);
            String header = new String(in.readNBytes((int) headerLength), StandardCharsets.UTF_8);
            JsonObject root = JsonParser.parseString(header).getAsJsonObject();
            for (Map.Entry<String, JsonElement> entry : root.entrySet()) {
                String name = entry.getKey();
                if (name.equals("__metadata__")) { continue; }
                JsonObject info = entry.getValue().getAsJsonObject();  

                JsonArray shapeJson = info.getAsJsonArray("shape");               
                int[] shape = new int[shapeJson.size()];
                for (int i = 0; i < shape.length; i++) { shape[i] = shapeJson.get(i).getAsInt(); }
                
                JsonArray dataOffsetJson = info.getAsJsonArray("data_offsets");
                long start = dataOffsetJson.get(0).getAsLong();
                long end = dataOffsetJson.get(1).getAsLong();

                String dtype = info.get("dtype").getAsString();

                TensorInfo t = new TensorInfo(name, dtype, shape, start, end);
                tensorsByName.put(name, t);
            }
            return tensorsByName;
        }
    }

    /**
     * Loads one tensor's data from the file.
     *
     * <p>Seeks to {@code 8 + headerLength + info.start()} and reads the tensor's
     * bytes as little-endian 32-bit floats. Only F32 data is supported.
     *
     * @param p path to the {@code .safetensors} file
     * @param info the tensor's entry from {@link #readHeader(Path)}
     * @return a new tensor with the shape given in {@code info}
     * @throws IOException if the file can't be read or ends early
     */
    public static Tensor loadTensor(Path p, TensorInfo info) throws IOException {
        try (FileChannel channel = FileChannel.open(p)) {
            long headerLength = readHeaderLength(p);
            long offset = 8 + headerLength + info.start();
            long numBytes = info.end() - info.start();
            channel.position(offset);
            ByteBuffer buf = ByteBuffer.allocate((int) numBytes).order(ByteOrder.LITTLE_ENDIAN);
            while (buf.hasRemaining()) {
                if (channel.read(buf) == -1) throw new IOException("unexpected end of file");
            }
            buf.flip();
            float[] data = new float[(int) numBytes / 4];
            buf.asFloatBuffer().get(data);
            return Tensor.of(data, info.shape());
        }
    }

    // Reads the first 8 bytes of the file: the header length as a little-endian long.
    private static long readHeaderLength(Path p) throws IOException {
        try (InputStream in = Files.newInputStream(p)) {       
            byte[] headerLengthBytes =  in.readNBytes(8);
            return ByteBuffer.wrap(headerLengthBytes).order(ByteOrder.LITTLE_ENDIAN).getLong();
        }          
    }
}
