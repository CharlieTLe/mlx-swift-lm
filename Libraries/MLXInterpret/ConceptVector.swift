// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Interpret) import MLXLMCommon

/// Concept directions derived from contrasting activations, independently of any
/// lens.
///
/// This exists so the J-space decomposition can be *tested* rather than assumed.
/// A ``JacobianLens/lensVector(forToken:at:)`` is by construction an element of
/// J-space, so decomposing one and asking how much lies in J-space is circular. A
/// difference-of-means direction is built only from the model's own activations, so
/// the fraction of it that turns out to be J-space expressible is a real
/// measurement.
@_spi(Interpret)
public enum ConceptVector {

    /// The direction that separates activations of `positive` prompts from
    /// `negative` ones: `mean(positive) − mean(negative)`, L2-normalized.
    ///
    /// The oldest and most robust way to get a concept direction. It makes no
    /// reference to the unembedding, the vocabulary, or any lens, which is exactly
    /// what makes it a fair input to a J-space test.
    ///
    /// - Parameters:
    ///   - positive: prompts in which the concept is present.
    ///   - negative: matched prompts in which it is not. Keep these structurally
    ///     parallel to `positive`, or the direction picks up sentence shape rather
    ///     than the concept.
    ///   - layer: layer boundary to read at.
    ///   - position: sequence position, negative counting from the end.
    public static func differenceOfMeans(
        model: any ResidualStreamReentry,
        tokenizer: any Tokenizer,
        positive: [String],
        negative: [String],
        layer: Int,
        position: Int = -1
    ) -> MLXArray {
        let positiveMean = meanActivation(
            model: model, tokenizer: tokenizer, prompts: positive, layer: layer,
            position: position)
        let negativeMean = meanActivation(
            model: model, tokenizer: tokenizer, prompts: negative, layer: layer,
            position: position)

        let difference = positiveMean - negativeMean
        let norm = sqrt((difference * difference).sum())
        return difference / maximum(norm, MLXArray(Float(1e-12)))
    }

    /// Mean activation across `prompts` at one layer and position.
    public static func meanActivation(
        model: any ResidualStreamReentry,
        tokenizer: any Tokenizer,
        prompts: [String],
        layer: Int,
        position: Int = -1
    ) -> MLXArray {
        precondition(!prompts.isEmpty, "need at least one prompt")

        var accumulator = MLXArray.zeros([model.hiddenSize], dtype: .float32)
        var counted = 0

        for prompt in prompts {
            let ids = tokenizer.encode(text: prompt, addSpecialTokens: false).map { Int32($0) }
            guard !ids.isEmpty else { continue }
            let tokens = MLXArray(ids)[.newAxis, .ellipsis]
            let hidden = model.hiddenState(of: tokens, at: layer)
            accumulator =
                accumulator + Activations.vector(from: hidden, position: position).asType(.float32)
            counted += 1
            eval(accumulator)
        }

        precondition(counted > 0, "every prompt tokenized to nothing")
        return accumulator / Float(counted)
    }
}
