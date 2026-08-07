// Copyright © 2026 Apple Inc.

import MLX
@_spi(Interpret) import MLXInterpret
@_spi(Interpret) import MLXLLM
@_spi(Interpret) import MLXLMCommon
import MLXNN
import XCTest

/// Foundational tests for the residual-stream tap.
///
/// Everything else in `MLXInterpret` is built on two assumptions these tests
/// check directly: that decomposing the forward pass into
/// `embed → runLayers → finalNorm → unembed` reproduces the model exactly, and
/// that MLX can differentiate the tail of the stack with respect to a hidden
/// state. If either fails, no readout downstream means anything.
final class ResidualStreamReentryTests: XCTestCase {

    /// Small, unquantized, deterministic-shaped Llama. Random weights are fine:
    /// these tests check structural identities, not learned behavior.
    private func makeLlama(layers: Int = 4) -> LlamaModel {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: layers, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 1e-5, vocabularySize: 100, kvHeads: 4)
        let model = LlamaModel(config)
        eval(model)
        return model
    }

    private func makeQwen3(layers: Int = 4) -> Qwen3Model {
        let config = Qwen3Configuration(
            hiddenSize: 64, hiddenLayers: layers, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 1e-5, vocabularySize: 100, kvHeads: 4, headDim: 8)
        let model = Qwen3Model(config)
        eval(model)
        return model
    }

    private let tokens = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]

    // MARK: - Split-path equivalence

    /// The load-bearing oracle: the four tap primitives, composed in order, must
    /// reproduce the model's own forward pass. No autodiff involved, so this can
    /// assert tightly.
    func testLlamaSplitPathMatchesForwardPass() {
        let model = makeLlama()

        let reference = model(tokens, cache: nil)
        let split = model.unembed(
            model.finalNorm(
                model.runLayers(model.embed(tokens), range: 0 ..< model.layerCount, cache: nil)))

        XCTAssertEqual(reference.shape, split.shape)
        XCTAssertTrue(
            allClose(reference, split, atol: 1e-5).item(Bool.self),
            "split path diverged from the monolithic forward pass")
    }

    func testQwen3SplitPathMatchesForwardPass() {
        let model = makeQwen3()

        let reference = model(tokens, cache: nil)
        let split = model.unembed(
            model.finalNorm(
                model.runLayers(model.embed(tokens), range: 0 ..< model.layerCount, cache: nil)))

        XCTAssertTrue(
            allClose(reference, split, atol: 1e-5).item(Bool.self),
            "split path diverged from the monolithic forward pass")
    }

    /// Running the stack in two pieces must equal running it in one.
    func testRunLayersIsComposable() {
        let model = makeLlama(layers: 6)
        let embedded = model.embed(tokens)

        let whole = model.runLayers(embedded, range: 0 ..< 6, cache: nil)
        let firstHalf = model.runLayers(embedded, range: 0 ..< 3, cache: nil)
        let pieced = model.runLayers(firstHalf, range: 3 ..< 6, cache: nil)

        XCTAssertTrue(
            allClose(whole, pieced, atol: 1e-5).item(Bool.self),
            "runLayers is not composable across a split")
    }

    func testEmptyRangeIsIdentity() {
        let model = makeLlama()
        let embedded = model.embed(tokens)
        let out = model.runLayers(embedded, range: 2 ..< 2, cache: nil)
        XCTAssertTrue(allClose(embedded, out, atol: 0).item(Bool.self))
    }

    // MARK: - Residual stream

    func testResidualStreamShapeAndEndpoints() {
        let model = makeLlama(layers: 5)
        let stream = model.hiddenStates(of: tokens)

        XCTAssertEqual(stream.count, model.layerCount + 1, "expected one state per layer boundary")
        for state in stream {
            XCTAssertEqual(state.shape, [1, 5, model.hiddenSize])
        }

        // First state is the embedding output.
        XCTAssertTrue(allClose(stream[0], model.embed(tokens), atol: 1e-6).item(Bool.self))

        // Last state, decoded, is the model's logits.
        XCTAssertTrue(
            allClose(model.decode(stream[stream.count - 1]), model(tokens, cache: nil), atol: 1e-5)
                .item(Bool.self))

        // The layers must actually transform the stream.
        XCTAssertFalse(
            allClose(stream[0], stream[stream.count - 1], atol: 1e-3).item(Bool.self),
            "residual stream is unchanged across the whole stack")
    }

    func testResidualStreamAtLayerMatchesFullSweep() {
        let model = makeLlama(layers: 4)
        let stream = model.hiddenStates(of: tokens)
        for layer in 0 ... model.layerCount {
            let direct = model.hiddenState(of: tokens, at: layer)
            XCTAssertTrue(
                allClose(stream[layer], direct, atol: 1e-5).item(Bool.self),
                "single-layer extraction disagrees with the full sweep at layer \(layer)")
        }
    }

    // MARK: - The state channel agrees with re-entry

    /// The second half of the conformance contract, and the reason the hybrid design
    /// is safe: the hidden states a model *reports* must be the same ones re-entry
    /// *operates on*.
    ///
    /// Checked at every layer boundary. Take the reported state entering block L,
    /// re-enter there, run the rest of the stack, and the logits must match the
    /// model's own. If the two paths ever diverge — a mask selected differently, a
    /// scale applied on one path but not the other — a readout would describe one
    /// computation while an intervention edited another, and nothing downstream
    /// would be trustworthy.
    func testReportedStatesAgreeWithReentryAtEveryLayer() {
        let model = makeLlama(layers: 5)
        let reference = model(tokens, cache: nil)
        let states = model.hiddenStates(of: tokens)

        for layer in 0 ... model.layerCount {
            let reentered = model.unembed(
                model.finalNorm(
                    model.runLayers(states[layer], range: layer ..< model.layerCount, cache: nil)))
            XCTAssertTrue(
                allClose(reference, reentered, atol: 1e-4).item(Bool.self),
                "re-entering at the reported state for layer \(layer) did not reproduce the logits")
        }
    }

    /// Same contract for Qwen3, whose per-head RMSNorm makes it the likelier of the
    /// two to break on a shape mistake.
    func testReportedStatesAgreeWithReentryForQwen3() {
        let model = makeQwen3(layers: 4)
        let reference = model(tokens, cache: nil)
        let states = model.hiddenStates(of: tokens)

        for layer in [0, 2, model.layerCount] {
            let reentered = model.unembed(
                model.finalNorm(
                    model.runLayers(states[layer], range: layer ..< model.layerCount, cache: nil)))
            XCTAssertTrue(
                allClose(reference, reentered, atol: 1e-4).item(Bool.self),
                "Qwen3 re-entry at layer \(layer) did not reproduce the logits")
        }
    }

    /// Asking for hidden states must not change the answer. If collection perturbed
    /// the forward pass, every measurement taken with it would be of a different
    /// model than the one that runs in production.
    func testCollectingHiddenStatesDoesNotChangeTheLogits() {
        let model = makeLlama(layers: 4)

        let plain = model(tokens, cache: nil)
        let collected = model(
            LMInput.Text(tokens: tokens), cache: nil, state: .collectingHiddenStates())

        XCTAssertTrue(
            allClose(plain, collected.logits, atol: 0).item(Bool.self),
            "opting into hidden-state collection changed the logits")
    }

    /// Callers who do not ask must not pay, and must not receive.
    func testStateIsAbsentWhenNotRequested() {
        let model = makeLlama(layers: 3)

        let withoutState = model(LMInput.Text(tokens: tokens), cache: nil, state: nil)
        XCTAssertNil(withoutState.state?[hiddenStatesKey])

        // An explicitly-false flag is the same as absent.
        var off = LMOutput.State()
        off[collectHiddenStatesKey] = false
        XCTAssertNil(
            model(LMInput.Text(tokens: tokens), cache: nil, state: off).state?[hiddenStatesKey])
    }

    func testReportedStateShapesAndCount() {
        let model = makeLlama(layers: 5)
        let states = model.hiddenStates(of: tokens)

        XCTAssertEqual(states.count, model.layerCount + 1)
        for state in states {
            XCTAssertEqual(state.shape, [1, tokens.dim(1), model.hiddenSize])
        }
        // The layers must actually transform the stream.
        XCTAssertFalse(
            allClose(states[0], states[states.count - 1], atol: 1e-3).item(Bool.self))
    }

    /// The advantage of reading through the normal path: it observes a real decode
    /// step in cache context, which a cache-free reconstruction cannot.
    func testCollectionWorksDuringCachedDecode() {
        let model = makeLlama(layers: 4)
        let cache = model.newCache(parameters: nil)

        // Warm the cache on the prompt, then collect for a single next token.
        _ = model(tokens, cache: cache)
        let step = MLXArray([7])[.newAxis, .ellipsis]
        let output = model(
            LMInput.Text(tokens: step), cache: cache, state: .collectingHiddenStates())

        guard let states = output.state?[hiddenStatesKey] else {
            return XCTFail("no hidden states reported during a cached decode step")
        }
        XCTAssertEqual(states.count, model.layerCount + 1)
        for state in states {
            // One position, because that is what this step computed.
            XCTAssertEqual(state.shape, [1, 1, model.hiddenSize])
        }
    }

    // MARK: - Depth reindexing

    func testDepthPercentRoundTrip() {
        let model = makeLlama(layers: 8)
        XCTAssertEqual(model.layer(atDepthPercent: 0), 0)
        XCTAssertEqual(model.layer(atDepthPercent: 100), 8)
        XCTAssertEqual(model.layer(atDepthPercent: 50), 4)
        XCTAssertEqual(model.depthPercent(ofLayer: 4), 50, accuracy: 1e-9)
        // Out-of-range input clamps rather than trapping.
        XCTAssertEqual(model.layer(atDepthPercent: -10), 0)
        XCTAssertEqual(model.layer(atDepthPercent: 250), 8)
    }

    // MARK: - Differentiability of the tail

    /// The directional derivative of the tail must be finite, non-zero, and
    /// correctly shaped.
    ///
    /// This uses central differences rather than forward-mode autodiff because
    /// `jvp` cannot traverse a real attention block in this MLX version — see
    /// ``AutodiffCapabilityTests`` for the two specific blockers.
    func testDirectionalDerivativeThroughTailIsFinite() {
        let model = makeLlama(layers: 4)
        let hidden = model.hiddenState(of: tokens, at: 1).asType(.float32)
        let tail = model.tail(from: 1)

        var direction = MLXRandom.normal(hidden.shape, key: MLXRandom.key(0)).asType(.float32)
        direction = direction / sqrt((direction * direction).sum())

        let epsilon: Float = 1e-3
        let derivative =
            (tail(hidden + epsilon * direction) - tail(hidden - epsilon * direction))
            / (2 * epsilon)

        XCTAssertEqual(derivative.shape, hidden.shape)
        let magnitude = derivative.abs().max().item(Float.self)
        XCTAssertTrue(magnitude.isFinite, "directional derivative is not finite")
        XCTAssertGreaterThan(magnitude, 0, "directional derivative is identically zero")
    }

    /// Central differences must be consistent with reverse mode, which is the
    /// autodiff direction MLX does support here: `⟨c, Jv⟩ == ⟨Jᵀc, v⟩`.
    ///
    /// This is the strongest available check that the numeric derivative is the
    /// real Jacobian and not an artifact of the step size, since the two are
    /// computed by completely different means.
    func testCentralDifferenceIsAdjointToReverseMode() {
        let model = makeLlama(layers: 3)
        let hidden = model.hiddenState(of: tokens, at: 1).asType(.float32)
        let tail = model.tail(from: 1)

        var v = MLXRandom.normal(hidden.shape, key: MLXRandom.key(11)).asType(.float32)
        v = v / sqrt((v * v).sum())
        let c = MLXRandom.normal(hidden.shape, key: MLXRandom.key(13)).asType(.float32)

        let epsilon: Float = 1e-3
        let jv = (tail(hidden + epsilon * v) - tail(hidden - epsilon * v)) / (2 * epsilon)
        let (_, jtc) = vjp(
            { inputs in [tail(inputs[0])] }, primals: [hidden], cotangents: [c])

        let lhs = (c * jv).sum().item(Float.self)
        let rhs = (jtc[0] * v).sum().item(Float.self)

        let scale = max(abs(lhs), abs(rhs), 1e-6)
        XCTAssertLessThan(
            abs(lhs - rhs) / scale, 5e-2,
            "central difference and reverse mode disagree: \(lhs) vs \(rhs)")
    }

    /// Reverse mode is what the Jacobian lens uses to build dictionary vectors.
    func testVJPThroughTailProducesFiniteResult() {
        let model = makeLlama(layers: 3)
        let hidden = model.hiddenState(of: tokens, at: 1).asType(.float32)
        let tail = model.tail(from: 1)

        let cotangent = MLXRandom.normal(hidden.shape, key: MLXRandom.key(3)).asType(.float32)
        let (_, gradients) = vjp(
            { inputs in [tail(inputs[0])] }, primals: [hidden], cotangents: [cotangent])

        XCTAssertEqual(gradients.count, 1)
        XCTAssertEqual(gradients[0].shape, hidden.shape)
        let magnitude = gradients[0].abs().max().item(Float.self)
        XCTAssertTrue(magnitude.isFinite, "VJP produced a non-finite gradient")
        XCTAssertGreaterThan(magnitude, 0, "VJP gradient is identically zero")
    }
}
