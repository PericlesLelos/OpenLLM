package openllm.model;

import openllm.backend.Backend;
import openllm.tensors.Tensor;

public class Gpt2Model {

    private final Gpt2Weights weights;
    private final Backend backend;
    
    public Gpt2Model(Gpt2Weights weights, Backend backend) {
        this.weights = weights;
        this.backend = backend;
    }

    Tensor embed(int[] ids) {
        int numTokens = ids.length;
        int nEmbd = weights.config().nEmbd();
        if (numTokens > weights.config().nCtx() || numTokens <= 0) {
            throw new IllegalArgumentException("Prompt must be at most " + nEmbd + " and larger than 0");
        }
        for (int id : ids) {
            if (id < 0 || id > weights.config().vocabSize() - 1) {
                throw new IllegalArgumentException("Ids must be between 0 and " + (weights.config().vocabSize() - 1)  + " (inclusive)");
            }
        }
        float[] out = new float[numTokens * nEmbd];
        for (int i = 0; i < numTokens; i++) {
            Tensor wteRow = weights.wte().row(ids[i]);
            Tensor wpeRow = weights.wpe().row(i);
            float[] wteRowData = wteRow.getData();
            float[] wpeRowData = wpeRow.getData();
            for (int j = 0; j < nEmbd; j++) {
                out[i * nEmbd + j] = wteRowData[j] + wpeRowData[j];
            }
        }
        return Tensor.of(out, numTokens, nEmbd);
    }

    public Qkv qKVProjection(Tensor x, LayerWeights layer) {
        x = x.contiguous();
        Tensor out = backend.matmul(x, layer.attnQkvWeight());
        out = backend.addBias(out, layer.attnQkvBias());
        int[] outShape = out.shape();
        int width = weights.config().nEmbd();
        float[] outData = out.getData();
        float[] qs = new float[outShape[0] * width];
        float[] ks = new float[outShape[0] * width];
        float[] vs = new float[outShape[0] * width];
        for (int i = 0; i < outShape[0]; i++) {
            for (int j = 0; j < width; j++) {
                qs[i * width + j] = outData[i * outShape[1] + j];
                ks[i * width + j] = outData[i * outShape[1] + j + width];
                vs[i * width + j] = outData[i * outShape[1] + j + width * 2];
            }
        }
        Tensor q = Tensor.of(qs, outShape[0], width);
        Tensor k = Tensor.of(ks, outShape[0], width);
        Tensor v = Tensor.of(vs, outShape[0], width);
        return new Qkv(q, k, v);
    }

    Tensor columnSlice(Tensor x, int startingCol, int width) {
        x = x.contiguous();
        float[] xData = x.getData();
        int[] xShape = x.shape();
        float[] outData = new float[xShape[0] * width];
        for (int i = 0; i < xShape[0]; i++) {
            for (int j = startingCol; j < startingCol + width; j++) {
                outData[i * width + j - startingCol] = xData[i * xShape[1] + j];
            }
        }
        
        return Tensor.of(outData, xShape[0], width);
    }

    Tensor causalMask(Tensor scores) {
        float[] scoresData = scores.getData();
        int[] scoresShape = scores.shape();
        for (int i = 0; i < scoresShape[0]; i++) {
            for (int j = i + 1; j < scoresShape[1]; j++) {
                scoresData[i * scoresShape[1] + j] = Float.NEGATIVE_INFINITY; 
            }
        }
        return Tensor.of(scoresData, scoresShape);
    }

    Tensor attention(Tensor x, LayerWeights layer) {
        Qkv qkv = qKVProjection(x, layer);
        int nHead = weights.config().nHead();
        int nEmbd = weights.config().nEmbd();
        int headDim = nEmbd / nHead;
        float[] attentionScores = new float[x.shape()[0] * nHead * headDim]; 
        int[] xShape = x.shape();

        for (int i = 0; i < nHead; i++) {
            Tensor qSlice = columnSlice(qkv.q(), i * headDim, headDim);
            Tensor kSliceT = columnSlice(qkv.k(), i * headDim, headDim).transpose(0, 1);
            Tensor vSlice = columnSlice(qkv.v(), i * headDim, headDim);
            Tensor scores = backend.matmul(qSlice, kSliceT);
            scores = backend.scale(scores, (float) (1 / Math.sqrt(headDim)));
            Tensor maskedScores = causalMask(scores);
            Tensor softScores = backend.softmax(maskedScores);
            Tensor fullColScores = backend.matmul(softScores, vSlice);
            float[] fullColScoresData = fullColScores.getData();
            for (int j = 0; j < xShape[0]; j++) {
                for (int k = 0; k < headDim; k++) {
                    attentionScores[j * nEmbd + i * headDim + k] = fullColScoresData[j * headDim + k];
                }
            }
        }
        Tensor out = backend.matmul(Tensor.of(attentionScores, x.shape()), layer.attnProjWeight());
        return backend.addBias(out, layer.attnProjBias());
    }

    Tensor mlp(Tensor x, LayerWeights layer) {
        x = backend.matmul(x, layer.mlpFcWeight());
        x = backend.addBias(x, layer.mlpFcBias());
        x = backend.gelu(x);
        x = backend.matmul(x, layer.mlpProjWeight());
        x = backend.addBias(x, layer.mlpProjBias());
        return x;
    }

    Tensor block(Tensor x, LayerWeights layer) {
        Tensor xNorm1 = backend.layerNorm(x, layer.ln1Weight(), layer.ln1Bias(), 1e-5f);
        x = backend.add(x, attention(xNorm1, layer));
        Tensor xNorm2 = backend.layerNorm(x, layer.ln2Weight(), layer.ln2Bias(), 1e-5f);
        x = backend.add(x, mlp(xNorm2, layer));
        return x;
    }

    public Tensor forward(int[] ids) {
        Tensor x = embed(ids);
        for (LayerWeights layer : weights.layers()) {
            x = block(x, layer);
        }
        x = backend.layerNorm(x, weights.lnFWeight(), weights.lnFBias(), 1e-5f);
        x = backend.matmul(x, weights.wte().transpose(0, 1));
        return x;
    }
}
