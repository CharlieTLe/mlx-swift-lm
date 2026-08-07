// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - Hidden-state collection keys
//
// A model opts into reporting its per-layer residual stream by honoring
// `collectHiddenStatesKey` on the incoming ``LMOutput/State`` and writing
// `hiddenStatesKey` on the outgoing one. This mirrors the MTP keys in
// `MTPDrafterModel.swift`, which established the pattern: a flag in, tensors out.
//
// Reading through the normal call path rather than a separate reconstruction
// matters. A re-run with `cache: nil` sees a different mask mode and a different
// SDPA kernel than a cached decode step does, so its activations are a
// reconstruction of what the model *would* do. These keys report what it actually
// did, mid-generation, with the real cache.

/// Per-layer residual stream, `layerCount + 1` entries.
///
/// Index `i` is the state *entering* block `i`: index 0 is the embedding output
/// (including any architecture-specific scaling) and the last entry is the output
/// of the final block, before the final norm. Each has shape
/// `[batch, position, hiddenSize]` for whatever span the call covered — the whole
/// prompt during prefill, a single position during decode.
///
/// Only populated when the caller sets ``collectHiddenStatesKey``.
public let hiddenStatesKey = LMOutput.Key<[MLXArray]>("interpret.hiddenStates")

/// Set this on the ``LMOutput/State`` passed *into* a model to opt into
/// ``hiddenStatesKey``. An absent key reads as `false`, so callers that do not ask
/// are unaffected.
///
/// ### Cost
///
/// Collection adds no arithmetic: every entry is an activation the forward pass
/// computes anyway, and MLX is lazy, so building the array only extends the graph.
/// What it does add is *retention*. Normally MLX can release each layer's output
/// once the next layer has consumed it; holding references keeps all of them alive
/// until evaluation. On a 36-layer model with `hiddenSize` 2560 in bfloat16 that is
/// negligible for a single decode step and roughly 360 MB across a 2048-token
/// prefill. Opt in per call rather than for a whole generation when prompts are long.
public let collectHiddenStatesKey = LMOutput.Key<Bool>("interpret.collectHiddenStates")

extension LMOutput.State {

    /// Whether the caller asked for per-layer hidden states.
    ///
    /// Models read this instead of the key directly so the absent-means-false
    /// default lives in one place.
    public var collectsHiddenStates: Bool {
        self[collectHiddenStatesKey] ?? false
    }

    /// A state that opts into hidden-state collection.
    public static func collectingHiddenStates() -> LMOutput.State {
        var state = LMOutput.State()
        state[collectHiddenStatesKey] = true
        return state
    }
}
