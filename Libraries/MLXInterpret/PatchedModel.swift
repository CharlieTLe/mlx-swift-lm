// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Interpret) import MLXLMCommon
import MLXNN

/// A ``LanguageModel`` that applies an edit to the residual stream partway
/// through every forward pass.
///
/// This is the piece that turns a static readout into an experiment. Because it
/// conforms to `LanguageModel`, it drops into `TokenIterator`, `generate(...)`,
/// and `ChatSession` unchanged — so "swap `spider` for `ant` at layer 12 and see
/// what the model says" is an ordinary generation call against a patched model,
/// with no special-cased generation path to maintain.
///
/// ```swift
/// let patched = PatchedModel(wrapping: model, at: 12) { hidden in
///     Interventions.coordinateSwap(hidden, from: spiderVector, to: antVector)
/// }
/// // generate with `patched` exactly as with `model`
/// ```
///
/// ### Behavior during generation
///
/// The patch runs on every forward pass, which during decoding means every token.
/// That is deliberate: it models "hold this concept in mind throughout," not "nudge
/// the first token." The KV cache is indexed absolutely and shared with the
/// unpatched layer ranges, so cached keys and values below the patch layer stay
/// valid.
///
/// Note the patch sees the activation for the *current* step only, whose position
/// axis has length 1 during decoding and the full prompt length during prefill.
/// A patch that depends on sequence position must handle both.
@_spi(Interpret)
public final class PatchedModel: Module, LanguageModel {

    /// The model being wrapped. Weights are shared, not copied.
    public let base: any ResidualStreamReentry

    /// Layer boundaries at which the patch is applied, ascending. Each in
    /// `0...layerCount`.
    ///
    /// Patching a *band* of layers rather than one is usually far more effective
    /// than a single boundary: a single-layer edit is partly undone by the layers
    /// that follow, whereas re-applying it at each boundary holds the concept in
    /// place. The paper does not specify which it uses, and the difference is large
    /// enough that it is worth calibrating rather than assuming.
    public let layers: [Int]

    /// The single patch layer, for the common case.
    public var layer: Int { layers[0] }

    /// The edit. Receives and returns `[batch, position, hiddenSize]`.
    public let patch: (MLXArray) -> MLXArray

    public convenience init(
        wrapping base: any ResidualStreamReentry,
        at layer: Int,
        patch: @escaping (MLXArray) -> MLXArray
    ) {
        self.init(wrapping: base, atLayers: [layer], patch: patch)
    }

    /// Apply the patch at several layer boundaries.
    public init(
        wrapping base: any ResidualStreamReentry,
        atLayers layers: [Int],
        patch: @escaping (MLXArray) -> MLXArray
    ) {
        precondition(!layers.isEmpty, "need at least one patch layer")
        let sorted = layers.sorted()
        precondition(
            sorted.allSatisfy { (0 ... base.layerCount).contains($0) },
            "layers \(sorted) out of range 0...\(base.layerCount)")
        self.base = base
        self.layers = sorted
        self.patch = patch
        super.init()
    }

    /// Convenience for the common case: exchange one concept direction for another.
    public convenience init(
        wrapping base: any ResidualStreamReentry,
        at layer: Int,
        swapping source: MLXArray,
        for target: MLXArray,
        coefficient: Float? = nil
    ) {
        self.init(wrapping: base, at: layer) { hidden in
            Interventions.coordinateSwap(
                hidden, from: source, to: target, coefficient: coefficient)
        }
    }

    /// Convenience for the ablation experiment: project directions out.
    public convenience init(
        wrapping base: any ResidualStreamReentry,
        at layer: Int,
        ablating directions: [MLXArray]
    ) {
        self.init(wrapping: base, at: layer) { hidden in
            Interventions.ablate(hidden, directions: directions)
        }
    }

    // MARK: - LanguageModel

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = base.embed(inputs)
        var cursor = 0
        for boundary in layers {
            hidden = base.runLayers(hidden, range: cursor ..< boundary, cache: cache)
            hidden = patch(hidden)
            cursor = boundary
        }
        hidden = base.runLayers(hidden, range: cursor ..< base.layerCount, cache: cache)
        return base.unembed(base.finalNorm(hidden))
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        base.newCache(parameters: parameters)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        base.sanitize(weights: weights)
    }

    /// Chunked prefill, mirroring `LLMModel`'s default.
    ///
    /// Reimplemented rather than inherited because that default lives on
    /// `MLXLLM.LLMModel`, and this target deliberately does not depend on MLXLLM —
    /// it works against any conforming architecture.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, windowSize: Int?
    ) throws -> PrepareResult {
        let prefillStepSize = windowSize ?? 512
        var y = input.text

        try withPreparedCache(cache, lengths: y.sequenceLengths) {
            while y.tokens.size > prefillStepSize {
                try Task.checkCancellation()
                autoreleasepool {
                    let chunk = y[.newAxis, ..<prefillStepSize]
                    _ = self(chunk.tokens, cache: cache.isEmpty ? nil : cache)
                    asyncEval(cache)
                    y = y[prefillStepSize...]
                }
            }
            eval(cache)
        }

        return .tokens(y)
    }

    // MARK: - Chat conventions

    // Forwarded so a patched model renders prompts and parses tool calls and
    // reasoning tags exactly as the model it wraps. Defaulting these instead
    // would silently change generation behavior for e.g. Qwen3's thinking tags,
    // and the resulting difference would look like an effect of the patch.

    public var toolCallFormat: ToolCallFormat? { base.toolCallFormat }

    public var reasoningConfig: ReasoningConfig? { base.reasoningConfig }
}
