// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Interpret) import MLXLMCommon
import MLXNN

// Residual-stream access for the Llama/Mistral family.
//
// Nothing in `Llama.swift` needs widening: this extension is inside `MLXLLM`, so
// it reads `configuration`, `model.layers`, and `lmHead` at their existing
// internal access level, and no protocol requirement mentions a non-public type.

@_spi(Interpret)
extension LlamaModel: ResidualStreamReentry {

    public var layerCount: Int { configuration.hiddenLayers }

    public var hiddenSize: Int { configuration.hiddenSize }

    public func embed(_ tokens: MLXArray) -> MLXArray {
        // Llama applies no post-lookup scaling.
        model.embedTokens(tokens)
    }

    public func runLayers(
        _ hidden: MLXArray, range: Range<Int>, cache: [KVCache]?
    ) -> MLXArray {
        guard !range.isEmpty else { return hidden }
        precondition(
            range.lowerBound >= 0 && range.upperBound <= layerCount,
            "layer range \(range) out of bounds 0..<\(layerCount)")

        // Matches `LlamaModelInner.callAsFunction`: one mask for the whole pass,
        // derived from the first cache entry. With `cache: nil` and more than one
        // position this is `.causal`.
        let mask = createAttentionMask(h: hidden, cache: cache?.first)

        var h = hidden
        for i in range {
            h = model.layers[i](h, mask: mask, cache: cache?[i])
        }
        return h
    }

    public func finalNorm(_ hidden: MLXArray) -> MLXArray {
        model.norm(hidden)
    }

    public func unembed(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        } else {
            return model.embedTokens.asLinear(hidden)
        }
    }
}
