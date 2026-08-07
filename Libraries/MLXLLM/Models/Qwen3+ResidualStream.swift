// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Interpret) import MLXLMCommon
import MLXNN

// Residual-stream access for the Qwen3 family. Structurally identical to Llama:
// pre-norm blocks, one mask for the whole pass, tied-or-untied unembedding.

@_spi(Interpret)
extension Qwen3Model: ResidualStreamReentry {

    public var layerCount: Int { configuration.hiddenLayers }

    public var hiddenSize: Int { configuration.hiddenSize }

    public func embed(_ tokens: MLXArray) -> MLXArray {
        // Qwen3 applies no post-lookup scaling.
        model.embedTokens(tokens)
    }

    public func runLayers(
        _ hidden: MLXArray, range: Range<Int>, cache: [KVCache]?
    ) -> MLXArray {
        guard !range.isEmpty else { return hidden }
        precondition(
            range.lowerBound >= 0 && range.upperBound <= layerCount,
            "layer range \(range) out of bounds 0..<\(layerCount)")

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
