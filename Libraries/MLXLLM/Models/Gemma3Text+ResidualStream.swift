// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Interpret) import MLXLMCommon
import MLXNN

// Residual-stream access for Gemma 3 text models.
//
// Two things make this conformance less mechanical than Llama/Qwen3, and both are
// reproduced faithfully here rather than approximated:
//
//  1. `embed` must scale by `sqrt(hiddenSize)` computed in bfloat16. Skipping it
//     leaves every readout wrong by a constant factor that varies with dtype.
//  2. Masking alternates per layer. Every `slidingWindowPattern`-th block attends
//     globally; the rest use a windowed mask. `Tests/MLXLMTests/Gemma3EncoderAccessTests`
//     deliberately hardcodes `.causal` for all layers, which is fine for its
//     purpose but would break split-path equivalence, so it is not the model here.

@_spi(Interpret)
extension Gemma3TextModel: ResidualStreamReentry {

    public var layerCount: Int { config.hiddenLayers }

    public var hiddenSize: Int { config.hiddenSize }

    public func embed(_ tokens: MLXArray) -> MLXArray {
        let h = model.embedTokens(tokens)
        // Matches Gemma3Model.callAsFunction: the scale is built in bfloat16 and
        // only then cast, which is not the same as scaling in the activation dtype.
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        return h * scale.asType(h.dtype)
    }

    public func runLayers(
        _ hidden: MLXArray, range: Range<Int>, cache: [KVCache]?
    ) -> MLXArray {
        guard !range.isEmpty else { return hidden }
        precondition(
            range.lowerBound >= 0 && range.upperBound <= layerCount,
            "layer range \(range) out of bounds 0..<\(layerCount)")

        let pattern = config.slidingWindowPattern
        let globalMask = createAttentionMask(h: hidden, cache: cache?[pattern - 1])
        let slidingWindowMask =
            if pattern > 1 {
                createAttentionMask(
                    h: hidden, cache: cache?[0], windowSize: config.slidingWindow)
            } else {
                MLXFast.ScaledDotProductAttentionMaskMode.none
            }

        var h = hidden
        for i in range {
            let isGlobal = (i % pattern == pattern - 1)
            h = model.layers[i](
                h, mask: isGlobal ? globalMask : slidingWindowMask, cache: cache?[i])
        }
        return h
    }

    public func finalNorm(_ hidden: MLXArray) -> MLXArray {
        model.norm(hidden)
    }

    public func unembed(_ hidden: MLXArray) -> MLXArray {
        // Gemma 3 always has an untied head; `sanitize` copies the embedding
        // weights into `lm_head` when the checkpoint omits it.
        lmHead(hidden)
    }
}
