// Copyright © 2026 Apple Inc.

import MLX
@_spi(Interpret) import MLXInterpret
@_spi(Interpret) import MLXLLM
@_spi(Interpret) import MLXLMCommon
import MLXNN
import XCTest

/// Pins down exactly which MLX autodiff modes work through a transformer stack.
///
/// These are not tests of `MLXInterpret` — they are tests of the platform it
/// stands on, and they exist because the answer is surprising and load-bearing.
/// `JacobianLens` defaults to central differences rather than forward-mode
/// autodiff solely because of the two failures recorded here.
///
/// One of these tests asserts that something *fails*. If MLX gains the missing
/// forward-mode rules it will start failing, which is the signal to revisit the
/// central-difference default. Measured against mlx-swift 0.31.6, which vendors
/// MLX 0.31.1.
final class AutodiffCapabilityTests: XCTestCase {

    private func makeLlama(layers: Int = 1, kvHeads: Int = 4) -> LlamaModel {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: layers, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 1e-5, vocabularySize: 100, kvHeads: kvHeads)
        let model = LlamaModel(config)
        eval(model)
        return model
    }

    private let tokens = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]

    /// Runs `body` and reports whether MLX raised an error, without letting an
    /// abort take the process down.
    private func mlxErrorMessage(_ body: () -> Void) -> String? {
        do {
            try withError { error in
                body()
                try error.check()
            }
            return nil
        } catch {
            return "\(error)"
        }
    }

    // MARK: - Forward mode: what works

    func testForwardModeWorksThroughRMSNorm() {
        let model = makeLlama()
        let h = model.embed(tokens).asType(.float32)
        let error = mlxErrorMessage {
            _ = jvp(
                { i in [model.finalNorm(i[0])] }, primals: [h], tangents: [MLXArray.ones(like: h)])
        }
        XCTAssertNil(error, "forward mode through RMSNorm regressed")
    }

    func testForwardModeWorksThroughUnembedding() {
        let model = makeLlama()
        let h = model.embed(tokens).asType(.float32)
        let error = mlxErrorMessage {
            _ = jvp(
                { i in [model.unembed(i[0])] }, primals: [h], tangents: [MLXArray.ones(like: h)])
        }
        XCTAssertNil(error, "forward mode through the unembedding regressed")
    }

    func testForwardModeWorksThroughRoPE() {
        let rope = initializeRope(
            dims: 8, base: 10_000, traditional: false, scalingConfig: nil,
            maxPositionEmbeddings: 32768)
        let x = MLXRandom.normal([1, 8, 5, 8], key: MLXRandom.key(0)).asType(.float32)
        let error = mlxErrorMessage {
            _ = jvp(
                { i in [applyRotaryPosition(rope, to: i[0], offset: nil)] },
                primals: [x], tangents: [MLXArray.ones(like: x)])
        }
        XCTAssertNil(error, "forward mode through RoPE regressed")
    }

    /// Attention with matching query and KV head counts, and no symbolic mask.
    func testForwardModeWorksThroughPlainAttention() {
        let x = MLXRandom.normal([1, 8, 5, 8], key: MLXRandom.key(0)).asType(.float32)
        let error = mlxErrorMessage {
            _ = jvp(
                { i in
                    [
                        MLXFast.scaledDotProductAttention(
                            queries: i[0], keys: i[0], values: i[0], scale: 0.35, mask: .none)
                    ]
                }, primals: [x], tangents: [MLXArray.ones(like: x)])
        }
        XCTAssertNil(error, "forward mode through unmasked non-GQA attention regressed")
    }

    /// An explicit additive mask array works where the symbolic `.causal` mode
    /// does not — so the blocker is the fast causal path, not causal masking.
    func testForwardModeWorksWithAnExplicitMaskArray() {
        let x = MLXRandom.normal([1, 8, 5, 8], key: MLXRandom.key(0)).asType(.float32)
        let mask = createCausalMask(n: 5, offset: 0).asType(.float32)
        let error = mlxErrorMessage {
            _ = jvp(
                { i in
                    [
                        MLXFast.scaledDotProductAttention(
                            queries: i[0], keys: i[0], values: i[0], scale: 0.35, mask: .array(mask)
                        )
                    ]
                }, primals: [x], tangents: [MLXArray.ones(like: x)])
        }
        XCTAssertNil(error, "forward mode with an explicit mask array regressed")
    }

    // MARK: - Forward mode: the two blockers

    /// Blocker 1: grouped-query attention. This is the decisive one for real
    /// models — every modern LLM here uses fewer KV heads than query heads, so
    /// there is no real attention block forward mode can traverse.
    ///
    /// This failure is at least *catchable*: it surfaces as an MLX error
    /// (`[addmm] Got 0 dimension input`) that `withError` converts to a Swift
    /// throw.
    func testForwardModeFailsOnGroupedQueryAttention() {
        let x = MLXRandom.normal([1, 5, 64], key: MLXRandom.key(0)).asType(.float32)
        let error = mlxErrorMessage {
            _ = jvp(
                { i in
                    let q = i[0].reshaped(1, 5, 8, 8).transposed(0, 2, 1, 3)
                    let kv = i[0][0..., 0..., 0 ..< 32].reshaped(1, 5, 4, 8).transposed(0, 2, 1, 3)
                    return [
                        MLXFast.scaledDotProductAttention(
                            queries: q, keys: kv, values: kv, scale: 0.35, mask: .none)
                    ]
                }, primals: [x], tangents: [MLXArray.ones(like: x)])
        }
        XCTAssertNotNil(
            error,
            "forward mode now handles grouped-query attention — revisit the central-difference default"
        )
    }

    /// Blocker 2 is **not tested here, deliberately, because it cannot be.**
    ///
    /// Forward mode through the symbolic `.causal` mask trips a raw C++
    /// `assert(tangents.size() == 3)` in `Select::jvp` (`primitives.cpp:3077`),
    /// which calls `abort()` directly rather than routing through MLX's error
    /// handler. Unlike the GQA failure above, `withError` cannot intercept it — a
    /// test exercising it takes down the whole test process with signal 6.
    ///
    /// The root cause is not attention. `Select` is `where(cond, a, b)`, which the
    /// SDPA fallback uses to apply the mask, and `Select::jvp` assumes all three
    /// inputs carry tangents. MLX sizes `tangents` to `argnums`, and a boolean
    /// condition can never carry one, so any `where` under `jvp` fails the assert.
    /// Four lines reproduce it with no model involved:
    ///
    /// ```swift
    /// jvp({ i in [which(condition, i[0], zeros)] },
    ///     primals: [x], tangents: [MLXArray.ones(like: x)])
    /// ```
    ///
    /// **This one is already fixed upstream.** `Select::jvp` on `ml-explore/mlx`
    /// main asserts `tangents.size() == argnums.size()` and indexes `tangents[i]`
    /// positionally. mlx-swift pins the `mlx` submodule at 0.31.1, which predates
    /// the fix, so this is a stale pin rather than an open defect and it should
    /// resolve on a submodule bump. Blocker 1 above is the one that persists.
    ///
    /// This is the reason `JacobianLens` never attempts forward-mode autodiff on a
    /// real stack even as a "try it and fall back" optimization: the attempt
    /// itself is unrecoverable. To re-check by hand:
    ///
    /// ```swift
    /// jvp({ i in [MLXFast.scaledDotProductAttention(
    ///         queries: i[0], keys: i[0], values: i[0], scale: 0.35, mask: .causal)] },
    ///     primals: [x], tangents: [MLXArray.ones(like: x)])
    /// ```
    ///
    /// Passing the same causal semantics as an explicit mask array works fine —
    /// see ``testForwardModeWorksWithAnExplicitMaskArray``.

    /// The consequence for a real model: forward mode cannot differentiate a
    /// block. Here the GQA error surfaces first, so it is catchable.
    func testForwardModeFailureThroughARealLayerIsCatchable() {
        let model = makeLlama()
        let h = model.embed(tokens).asType(.float32)
        let error = mlxErrorMessage {
            _ = jvp(
                { i in [model.runLayers(i[0], range: 0 ..< 1, cache: nil)] },
                primals: [h], tangents: [MLXArray.ones(like: h)])
        }
        XCTAssertNotNil(error, "forward mode now works through a real layer")
        XCTAssertTrue(
            error?.contains("addmm") ?? false,
            "unexpected forward-mode failure mode: \(error ?? "none")")
    }

    // MARK: - Reverse mode works throughout

    func testReverseModeWorksThroughTheFullStack() {
        let model = makeLlama(layers: 4)
        let h = model.embed(tokens).asType(.float32)
        let error = mlxErrorMessage {
            _ = vjp(
                { i in [model.runLayers(i[0], range: 0 ..< 4, cache: nil)] },
                primals: [h], cotangents: [MLXArray.ones(like: h)])
        }
        XCTAssertNil(error, "reverse mode through the full stack regressed")
    }

    func testReverseModeGradientIsNonTrivial() {
        let model = makeLlama(layers: 4)
        let h = model.embed(tokens).asType(.float32)
        let (_, gradients) = vjp(
            { i in [model.runLayers(i[0], range: 0 ..< 4, cache: nil)] },
            primals: [h], cotangents: [MLXArray.ones(like: h)])
        XCTAssertEqual(gradients[0].shape, h.shape)
        let magnitude = gradients[0].abs().max().item(Float.self)
        XCTAssertTrue(magnitude.isFinite)
        XCTAssertGreaterThan(magnitude, 0)
    }
}
