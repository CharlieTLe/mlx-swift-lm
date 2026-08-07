// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Interpret) import MLXLMCommon

/// The logit lens: read an intermediate activation by pushing it straight through
/// the final norm and the unembedding, as if the remaining blocks were the
/// identity.
///
/// This is the cheap baseline, and it serves two purposes here beyond being a
/// lens in its own right:
///
/// - **Oracle.** At the last layer boundary the remaining stack *is* the identity,
///   so the logit lens and the Jacobian lens must agree exactly there. That
///   pins down the Jacobian lens implementation.
/// - **Contrast.** The paper's claim that the logit lens is unreliable early in
///   the stack is directly checkable by comparing ``Readout/entropy`` between the
///   two lenses at the same depth.
@_spi(Interpret)
public struct LogitLens {

    public let model: any ResidualStreamReentry
    public let tokenizer: any Tokenizer

    public init(model: any ResidualStreamReentry, tokenizer: any Tokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    /// Read out a single activation vector.
    ///
    /// - Parameters:
    ///   - hidden: shape `[hiddenSize]`.
    ///   - layer: the layer boundary `hidden` was taken from. Used only to label
    ///     the result.
    public func readout(of hidden: MLXArray, at layer: Int, topK: Int = 10) -> Readout {
        precondition(
            hidden.ndim == 1 && hidden.dim(0) == model.hiddenSize,
            "expected a [\(model.hiddenSize)] activation, got \(hidden.shape)")

        // Restore the [batch, position, hidden] shape the unembedding expects,
        // then drop it again.
        let scores = model.decode(hidden[.newAxis, .newAxis, 0...]).reshaped([-1])

        return ReadoutBuilder.build(
            scores: scores,
            topK: topK,
            layer: layer,
            depthPercent: model.depthPercent(ofLayer: layer),
            calibrationPromptCount: nil,
            tokenizer: tokenizer
        )
    }

    /// Read out every layer boundary for `tokens` at one sequence position.
    ///
    /// - Parameters:
    ///   - position: sequence index, negative counting from the end. `-1` is the
    ///     last prompt token, which is the position that produces the next token.
    public func sweep(
        tokens: MLXArray, position: Int = -1, topK: Int = 10
    ) -> [Readout] {
        let stream = model.hiddenStates(of: tokens.ndim == 1 ? tokens[.newAxis, 0...] : tokens)
        return stream.enumerated().map { layer, hidden in
            readout(of: Activations.vector(from: hidden, position: position), at: layer, topK: topK)
        }
    }
}

/// Shape wrangling shared by the lenses.
@_spi(Interpret)
public enum Activations {

    /// Extract a single `[hiddenSize]` vector from a `[batch, position, hidden]`
    /// activation.
    ///
    /// - Parameter position: negative counts from the end; `-1` is the last token.
    public static func vector(from hidden: MLXArray, position: Int = -1, batch: Int = 0) -> MLXArray
    {
        switch hidden.ndim {
        case 1:
            return hidden
        case 2:
            let p = position < 0 ? hidden.dim(0) + position : position
            return hidden[p, 0...]
        case 3:
            let p = position < 0 ? hidden.dim(1) + position : position
            return hidden[batch, p, 0...]
        default:
            preconditionFailure("unsupported activation rank \(hidden.ndim)")
        }
    }
}
