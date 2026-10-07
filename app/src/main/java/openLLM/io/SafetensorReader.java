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

public class SafetensorReader {
    
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

    private static long readHeaderLength(Path p) throws IOException {
        try (InputStream in = Files.newInputStream(p)) {       
            byte[] headerLengthBytes =  in.readNBytes(8);
            return ByteBuffer.wrap(headerLengthBytes).order(ByteOrder.LITTLE_ENDIAN).getLong();
        }          
    }
}
