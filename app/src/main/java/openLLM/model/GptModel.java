package openllm.model;

import java.nio.file.Path;

public enum GptModel {

    SMALL(Gpt2Config.GPT2_SMALL, "gpt2-small"),
    MEDIUM(Gpt2Config.GPT2_MEDIUM, "gpt2-medium"),
    LARGE(Gpt2Config.GPT2_LARGE, "gpt2-large"),
    XL(Gpt2Config.GPT2_XL, "gpt2-xl");

    private final Gpt2Config config;
    private final String folder;
    private GptModel(Gpt2Config config, String folder) {
        this.config = config;
        this.folder = folder;
    }

    public Gpt2Config config() {
        return this.config;
    }

    public Path weightsPath() {
        return Path.of("../models/" + folder + "/model.safetensors");
    }
}
