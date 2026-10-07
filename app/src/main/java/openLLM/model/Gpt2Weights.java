package openllm.model;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import openllm.tensors.Tensor;
import openllm.io.SafetensorReader;
import openllm.io.TensorInfo;

public record Gpt2Weights(Gpt2Config config, Tensor wte, Tensor wpe, LayerWeights[] layers, Tensor lnFWeight, Tensor lnFBias) {
    public static Gpt2Weights load(Path p) throws IOException {
        Map<String, TensorInfo> map = SafetensorReader.readHeader(p);
        Gpt2Config config = Gpt2Config.GPT2_SMALL;
        
        LayerWeights[] layers = new LayerWeights[config.nLayer()];
        Tensor wte = load(p, map, "wte.weight");
        Tensor wpe = load(p, map, "wpe.weight");
        Tensor lnFWeight = load(p, map, "ln_f.weight");
        Tensor lnFBias = load(p, map, "ln_f.bias");
        
        for (int i = 0; i < config.nLayer(); i++) {
            String prefix = "h." + i + ".";
            
            Tensor ln1Weight = load(p, map, prefix + "ln_1.weight");
            Tensor ln1Bias = load(p, map, prefix + "ln_1.bias");
            Tensor attnQkvWeight = load(p, map, prefix + "attn.c_attn.weight");
            Tensor attnQkvBias = load(p, map, prefix + "attn.c_attn.bias");
            Tensor attnProjWeight = load(p, map, prefix + "attn.c_proj.weight");
            Tensor attnProjBias = load(p, map, prefix + "attn.c_proj.bias");
            Tensor ln2Weight = load(p, map, prefix + "ln_2.weight");
            Tensor ln2Bias = load(p, map, prefix + "ln_2.bias");
            Tensor mlpFcWeight = load(p, map, prefix + "mlp.c_fc.weight");
            Tensor mlpFcBias = load(p, map, prefix + "mlp.c_fc.bias");
            Tensor mlpProjWeight = load(p, map, prefix + "mlp.c_proj.weight");
            Tensor mlpProjBias = load(p, map, prefix + "mlp.c_proj.bias");

            layers[i] = new LayerWeights(
                ln1Weight, 
                ln1Bias, 
                attnQkvWeight, 
                attnQkvBias, 
                attnProjWeight, 
                attnProjBias,
                ln2Weight,
                ln2Bias,
                mlpFcWeight,
                mlpFcBias,
                mlpProjWeight,
                mlpProjBias
            );
            
        }
        return new Gpt2Weights(config, wte, wpe, layers, lnFWeight, lnFBias);    
    }

    private static Tensor load(Path p, Map<String, TensorInfo> map, String name) throws IOException {
        TensorInfo info = map.get(name);
        if (info == null) {
            throw new IllegalArgumentException("no tensor named " + name);
        }
        return SafetensorReader.loadTensor(p, info);
    }
}
