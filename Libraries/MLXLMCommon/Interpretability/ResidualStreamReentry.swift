// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// A model whose stack can be run in pieces, so a caller can resume the forward
/// pass from a hidden state the model would not itself have produced.
///
/// This is deliberately *not* the way to read the residual stream. Reading goes
/// through the normal call path: set ``collectHiddenStatesKey`` on the incoming
/// ``LMOutput/State`` and read ``hiddenStatesKey`` back off the output. That
/// observes what the model actually did, with the real cache and the real mask
/// mode, which a separate reconstruction cannot.
///
/// What re-entry is for is the two things an output channel cannot express:
///
/// - **Differentiation.** A Jacobian-vector product needs a re-runnable function
///   `h_ℓ → h_L` that can be evaluated at perturbed inputs and traced by `vjp`.
///   An output channel yields values after the fact. Synthesizing the function by
///   re-running the whole model with a replacement patch works but recomputes
///   layers `0..<ℓ` every evaluation — at layer 28 of 36 that is 36 layers of work
///   where ``tail(from:)`` does 8.
/// - **Patching mid-generation.** ``PatchedModel`` wraps a conforming model and
///   applies an edit at a layer boundary on every forward pass, which is what lets
///   an activation-patching experiment run through `TokenIterator` and
///   `ChatSession` unchanged.
///
/// ### The conformance contract
///
/// Two obligations, both checked by `ResidualStreamReentryTests`:
///
/// 1. The four primitives compose to reproduce the model's own forward pass:
///    ```
///    unembed(finalNorm(runLayers(embed(t), 0..<layerCount, nil))) == model(t)
///    ```
/// 2. The model honors ``collectHiddenStatesKey``, and the state it reports agrees
///    with re-entry: re-entering at `hiddenStates[ℓ]` and running `ℓ..<layerCount`
///    reproduces the same logits.
///
/// Two things are easy to get wrong and silent when you do. ``embed(_:)`` must
/// include any architecture-specific post-lookup scaling — Gemma multiplies by
/// `sqrt(hiddenSize)` computed in bfloat16. ``runLayers(_:range:cache:)`` must
/// reproduce per-layer mask selection — Gemma 3 alternates windowed and global
/// attention on a `slidingWindowPattern` cycle.
@_spi(Interpret)
public protocol ResidualStreamReentry: LanguageModel {

    /// Number of transformer blocks.
    var layerCount: Int { get }

    /// Residual stream width.
    var hiddenSize: Int { get }

    /// Number of vocabulary logits ``unembed(_:)`` produces.
    ///
    /// Not available on ``LanguageModel``, but every conforming model already has
    /// it, and the lenses need it to shape cotangents for reverse-mode work.
    var vocabularySize: Int { get }

    /// Token embeddings, including any architecture-specific post-lookup scaling.
    func embed(_ tokens: MLXArray) -> MLXArray

    /// Run the blocks in `range` over an arbitrary hidden state.
    ///
    /// Pass `cache: nil` for analysis: a full-sequence pass mutates no state and is
    /// the only form safe to use inside an autodiff trace. When `cache` is non-nil
    /// it must have one entry per block in the *whole* model, indexed absolutely,
    /// so `range` selects a slice of it.
    func runLayers(_ hidden: MLXArray, range: Range<Int>, cache: [KVCache]?) -> MLXArray

    /// The norm applied after the final block, before unembedding.
    func finalNorm(_ hidden: MLXArray) -> MLXArray

    /// Project a post-norm hidden state to vocabulary logits, resolving tied
    /// embeddings and applying any softcap or scale the architecture defines.
    func unembed(_ hidden: MLXArray) -> MLXArray
}

@_spi(Interpret)
extension ResidualStreamReentry {

    // MARK: - Reading, via the normal call path

    /// The residual stream at every layer boundary, `layerCount + 1` entries.
    ///
    /// Goes through the model's own `callAsFunction(_:cache:state:)` with
    /// ``collectHiddenStatesKey`` set, so the activations are the ones the model
    /// really produced rather than a reconstruction. Index `i` is the state
    /// *entering* block `i`; index 0 is the embedding output.
    ///
    /// - Parameters:
    ///   - tokens: `[batch, position]`, or 1-D for a single sequence.
    ///   - cache: pass a live cache to observe a decode step in context. `nil` runs
    ///     the span as one uncached pass.
    public func hiddenStates(of tokens: MLXArray, cache: [KVCache]? = nil) -> [MLXArray] {
        let batched = tokens.ndim == 1 ? tokens[.newAxis, 0...] : tokens
        let output = self(
            LMInput.Text(tokens: batched), cache: cache,
            state: .collectingHiddenStates())

        guard let states = output.state?[hiddenStatesKey] else {
            preconditionFailure(
                "\(type(of: self)) conforms to ResidualStreamReentry but did not populate "
                    + "hiddenStatesKey when asked. Its callAsFunction(_:cache:state:) must "
                    + "honor collectHiddenStatesKey — see Llama.swift for the pattern.")
        }
        precondition(
            states.count == layerCount + 1,
            "expected \(layerCount + 1) hidden states, got \(states.count)")
        return states
    }

    /// The residual stream entering block `layer`.
    public func hiddenState(of tokens: MLXArray, at layer: Int, cache: [KVCache]? = nil)
        -> MLXArray
    {
        precondition(
            (0 ... layerCount).contains(layer),
            "layer \(layer) out of range 0...\(layerCount)")
        return hiddenStates(of: tokens, cache: cache)[layer]
    }

    // MARK: - Re-entry

    /// The remainder of the stack from `layer` onward, **excluding**
    /// ``finalNorm(_:)`` and ``unembed(_:)``.
    ///
    /// This is the function whose Jacobian the Jacobian lens is defined over. The
    /// norm is deliberately left out: the lens readout is
    /// `softmax(W_U · norm(J h))`, so the norm applies to the Jacobian's output
    /// rather than being folded into `J`.
    ///
    /// Safe to hand to `jvp` / `vjp`: it passes `cache: nil` and never calls `eval`.
    public func tail(from layer: Int) -> (MLXArray) -> MLXArray {
        precondition(
            (0 ... layerCount).contains(layer),
            "layer \(layer) out of range 0...\(layerCount)")
        return { [self] hidden in
            runLayers(hidden, range: layer ..< layerCount, cache: nil)
        }
    }

    /// Vocabulary logits for a hidden state taken from the *end* of the stack,
    /// i.e. `unembed(finalNorm(hidden))`. This is the logit-lens readout.
    public func decode(_ hidden: MLXArray) -> MLXArray {
        unembed(finalNorm(hidden))
    }

    // MARK: - Depth

    /// The paper's depth convention: layers reindexed onto 0–100 so results are
    /// comparable across models of different depth.
    public func layer(atDepthPercent percent: Double) -> Int {
        let clamped = min(max(percent, 0), 100)
        return Int((clamped / 100 * Double(layerCount)).rounded())
    }

    /// Inverse of ``layer(atDepthPercent:)``, for labelling results.
    public func depthPercent(ofLayer layer: Int) -> Double {
        guard layerCount > 0 else { return 0 }
        return Double(layer) / Double(layerCount) * 100
    }
}
