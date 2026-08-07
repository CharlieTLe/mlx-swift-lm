// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Interpret) import MLXLMCommon

/// A ranked list of vocabulary tokens read out of a single residual-stream
/// activation — the output of every lens in this library.
@_spi(Interpret)
public struct Readout: Sendable {

    /// One ranked token.
    public struct Entry: Sendable {
        /// Vocabulary id.
        public let tokenId: Int
        /// Display form, via ``TokenDisplay``.
        public let token: String
        /// Pre-softmax score.
        public let score: Float
        /// Softmax probability over the whole vocabulary.
        public let probability: Float
    }

    /// Highest-scoring tokens first.
    public let entries: [Entry]

    /// Layer boundary this was read at.
    public let layer: Int

    /// ``layer`` expressed on the paper's 0–100 depth scale.
    public let depthPercent: Double

    /// How many calibration prompts the Jacobian was averaged over.
    ///
    /// `nil` for the logit lens, which needs no averaging. For the Jacobian lens
    /// this is load-bearing context, not trivia: the paper's ~1000-prompt average
    /// is what separates "generally poised to be spoken about" from "happened to
    /// be verbalized in this one context," so a readout is not interpretable
    /// without knowing it.
    public let calibrationPromptCount: Int?

    /// The top-scoring token, or `nil` for an empty readout.
    public var top: Entry? { entries.first }

    /// Whether any of `tokens` appears in the readout, compared case- and
    /// whitespace-insensitively.
    public func contains(anyOf tokens: [String]) -> Bool {
        let wanted = Set(
            tokens.map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() })
        return entries.contains { entry in
            wanted.contains(
                entry.token.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
        }
    }

    /// Rank of `token` in the readout, or `nil` if absent. 0 is the top slot.
    public func rank(of token: String) -> Int? {
        let wanted = token.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return entries.firstIndex {
            $0.token.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() == wanted
        }
    }

    /// Shannon entropy of the readout's probabilities, in nats.
    ///
    /// A proxy for how committed the readout is. The paper's claim that the logit
    /// lens is unreliable early in the stack shows up here as high entropy where
    /// the Jacobian lens is already sharp.
    public var entropy: Float {
        -entries.reduce(Float(0)) { acc, entry in
            entry.probability > 0 ? acc + entry.probability * log(entry.probability) : acc
        }
    }

    public var description: String {
        let body =
            entries
            .map { String(format: "%@ (%.3f)", $0.token, $0.probability) }
            .joined(separator: ", ")
        let suffix = calibrationPromptCount.map { " [n=\($0)]" } ?? ""
        return String(format: "L%d (%.0f%%)%@: ", layer, depthPercent, suffix) + body
    }
}

/// Builds a ``Readout`` from a vocabulary-sized score vector.
enum ReadoutBuilder {

    /// - Parameters:
    ///   - scores: shape `[vocabularySize]`.
    ///   - topK: how many entries to keep.
    static func build(
        scores: MLXArray,
        topK: Int,
        layer: Int,
        depthPercent: Double,
        calibrationPromptCount: Int?,
        tokenizer: any Tokenizer
    ) -> Readout {
        precondition(
            scores.ndim == 1, "expected a [vocabularySize] score vector, got \(scores.shape)")

        let probabilities = softmax(scores.asType(.float32), axis: -1)

        // MLX Swift has no `topK`; a full argSort is the clearest correct option
        // and costs milliseconds even on a 150k vocabulary.
        let ascending = argSort(scores, axis: -1)
        let k = min(topK, scores.dim(0))
        let topIndices = ascending[(scores.dim(0) - k)...].asArray(Int32.self).reversed()

        let scoreValues = scores.asType(.float32).asArray(Float.self)
        let probabilityValues = probabilities.asArray(Float.self)

        let entries = topIndices.map { index -> Readout.Entry in
            let id = Int(index)
            return Readout.Entry(
                tokenId: id,
                token: TokenDisplay.string(for: id, tokenizer: tokenizer),
                score: scoreValues[id],
                probability: probabilityValues[id]
            )
        }

        return Readout(
            entries: entries,
            layer: layer,
            depthPercent: depthPercent,
            calibrationPromptCount: calibrationPromptCount
        )
    }
}
