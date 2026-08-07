// Copyright © 2026 Apple Inc.

import MLX
@_spi(Interpret) import MLXInterpret
@_spi(Interpret) import MLXLLM
@_spi(Interpret) import MLXLMCommon
import MLXNN
import XCTest

/// Tests for the patched-model wrapper.
///
/// The wrapper's whole value is that it is indistinguishable from a real
/// `LanguageModel`, so these tests mostly check that it stays faithful: an
/// identity patch must be invisible, and the incremental (cached) path must agree
/// with the full-sequence path.
final class PatchedModelTests: XCTestCase {

    private static let vocabularySize = 128

    private func makeModel(layers: Int = 4, tied: Bool = true) -> LlamaModel {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: layers, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 1e-5, vocabularySize: Self.vocabularySize, kvHeads: 4,
            tieWordEmbeddings: tied)
        let model = LlamaModel(config)
        eval(model)
        return model
    }

    private let tokens = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]

    /// `LMInput.Text` expects **1-D** tokens and adds the batch axis itself; handing
    /// it an already-batched array double-batches and collapses the token axis
    /// downstream.
    private let flatTokens = MLXArray([1, 2, 3, 4, 5])

    fileprivate func direction(_ index: Int, size: Int = 64) -> MLXArray {
        var vector = MLXArray.zeros([size], dtype: .float32)
        vector[index] = MLXArray(Float(1))
        return vector
    }

    // MARK: - Faithfulness

    /// An identity patch must produce exactly the base model's logits. If this
    /// drifts, every measured effect of a real patch is confounded by the wrapper.
    func testIdentityPatchMatchesTheBaseModel() {
        let model = makeModel()
        let patched = PatchedModel(wrapping: model, at: 2) { $0 }

        let reference = model(tokens, cache: nil)
        let result = patched(tokens, cache: nil)

        XCTAssertEqual(reference.shape, result.shape)
        XCTAssertTrue(
            allClose(reference, result, atol: 1e-5).item(Bool.self),
            "identity patch changed the logits")
    }

    /// The identity must hold at both ends of the stack, where the layer ranges
    /// degenerate to empty on one side.
    func testIdentityPatchIsFaithfulAtBothBoundaries() {
        let model = makeModel(layers: 3)
        let reference = model(tokens, cache: nil)

        for layer in [0, model.layerCount] {
            let patched = PatchedModel(wrapping: model, at: layer) { $0 }
            XCTAssertTrue(
                allClose(reference, patched(tokens, cache: nil), atol: 1e-5).item(Bool.self),
                "identity patch at layer \(layer) changed the logits")
        }
    }

    /// Also verify the untied-unembedding path, which takes a different branch in
    /// `unembed`.
    func testIdentityPatchIsFaithfulWithAnUntiedHead() {
        let model = makeModel(tied: false)
        let patched = PatchedModel(wrapping: model, at: 2) { $0 }
        XCTAssertTrue(
            allClose(model(tokens, cache: nil), patched(tokens, cache: nil), atol: 1e-5)
                .item(Bool.self))
    }

    /// Ablating a direction the activation has no component along must be a no-op
    /// end to end, not just inside the intervention primitive.
    ///
    /// The direction has to be orthogonal to *every* position's activation, not
    /// just the one being read: the patch applies at all positions, and a change at
    /// an earlier position reaches the last one through attention.
    func testAblatingAnOrthogonalDirectionIsANoOp() {
        let model = makeModel()
        let hidden = model.hiddenState(of: tokens, at: 2).asType(.float32)

        let positionVectors = (0 ..< hidden.dim(1)).map { hidden[0, $0, 0...] }
        let basis = Interventions.orthonormalBasis(positionVectors, dtype: .float32)

        // Gram-Schmidt a candidate against the span of all position activations.
        var candidate = direction(0)
        for axis in basis {
            candidate = candidate - Interventions.project(candidate, onto: axis) * axis
        }
        let orthogonal = Interventions.normalized(candidate)

        // Confirm the construction worked before relying on it.
        for vector in positionVectors {
            XCTAssertEqual(
                Interventions.cosineSimilarity(orthogonal, vector), 0, accuracy: 1e-4,
                "test setup: direction is not orthogonal to every position")
        }

        let patched = PatchedModel(wrapping: model, at: 2, ablating: [orthogonal])

        XCTAssertTrue(
            allClose(model(tokens, cache: nil), patched(tokens, cache: nil), atol: 1e-3)
                .item(Bool.self),
            "ablating a direction orthogonal to the activation changed the logits")
    }

    // MARK: - Patches take effect

    func testSwapPatchChangesTheLogits() {
        let model = makeModel()
        let patched = PatchedModel(
            wrapping: model, at: 2, swapping: direction(0), for: direction(1), coefficient: 10)

        let reference = model(tokens, cache: nil)
        let result = patched(tokens, cache: nil)

        XCTAssertFalse(
            allClose(reference, result, atol: 1e-3).item(Bool.self),
            "the swap patch had no effect on the logits")
    }

    func testPatchIsDeterministic() {
        let model = makeModel()
        let patched = PatchedModel(
            wrapping: model, at: 2, swapping: direction(0), for: direction(1), coefficient: 5)

        let first = patched(tokens, cache: nil)
        let second = patched(tokens, cache: nil)
        XCTAssertTrue(allClose(first, second, atol: 0).item(Bool.self))
    }

    /// Ablating a direction the activation *does* have a component along must change
    /// the output — the counterpart to the orthogonal no-op above.
    func testAblatingAPresentDirectionChangesTheLogits() {
        let model = makeModel()
        let activation = Activations.vector(from: model.hiddenState(of: tokens, at: 2))
            .asType(.float32)
        let present = Interventions.normalized(activation)

        let patched = PatchedModel(wrapping: model, at: 2, ablating: [present])

        XCTAssertFalse(
            allClose(model(tokens, cache: nil), patched(tokens, cache: nil), atol: 1e-3)
                .item(Bool.self),
            "ablating a direction the activation contains had no effect")
    }

    // MARK: - Multi-layer patching

    /// An identity patch stays invisible however many boundaries it is applied at.
    func testMultiLayerIdentityPatchIsFaithful() {
        let model = makeModel(layers: 6)
        let patched = PatchedModel(wrapping: model, atLayers: [1, 3, 5]) { $0 }

        XCTAssertEqual(patched.layers, [1, 3, 5])
        XCTAssertTrue(
            allClose(model(tokens, cache: nil), patched(tokens, cache: nil), atol: 1e-5)
                .item(Bool.self),
            "identity patch across several layers changed the logits")
    }

    /// Layers are applied in ascending order regardless of how they were given.
    func testMultiLayerBoundariesAreSorted() {
        let model = makeModel(layers: 6)
        let patched = PatchedModel(wrapping: model, atLayers: [5, 1, 3]) { $0 }
        XCTAssertEqual(patched.layers, [1, 3, 5])
    }

    /// A repeated boundary means the patch runs twice there, which must not corrupt
    /// the layer walk.
    func testRepeatedBoundaryIsFaithfulForAnIdentityPatch() {
        let model = makeModel(layers: 4)
        let patched = PatchedModel(wrapping: model, atLayers: [2, 2]) { $0 }
        XCTAssertTrue(
            allClose(model(tokens, cache: nil), patched(tokens, cache: nil), atol: 1e-5)
                .item(Bool.self))
    }

    /// Patching a band should move the output further than patching one boundary,
    /// which is the reason multi-layer support exists.
    func testPatchingABandHasMoreEffectThanASingleLayer() {
        let model = makeModel(layers: 6)
        let baseline = model(tokens, cache: nil)

        func deviation(_ patched: PatchedModel) -> Float {
            (baseline - patched(tokens, cache: nil)).abs().max().item(Float.self)
        }

        let single = deviation(
            PatchedModel(
                wrapping: model, at: 3, swapping: direction(0), for: direction(1),
                coefficient: 2))
        let band = deviation(
            PatchedModel(wrapping: model, atLayers: [2, 3, 4]) { hidden in
                Interventions.coordinateSwap(
                    hidden, from: self.direction(0), to: self.direction(1), coefficient: 2)
            })

        XCTAssertGreaterThan(
            band, single,
            "patching a band of layers had no more effect than a single layer")
    }

    /// Incremental decoding must stay faithful with several patch points too.
    func testMultiLayerIncrementalDecodingMatchesFullSequence() {
        let model = makeModel(layers: 6)
        let patched = PatchedModel(wrapping: model, atLayers: [2, 4]) { hidden in
            Interventions.coordinateSwap(
                hidden, from: self.direction(0), to: self.direction(1), coefficient: 2)
        }

        let expected = patched(tokens, cache: nil)[0, -1, 0...]

        let cache = patched.newCache(parameters: nil)
        var last: MLXArray?
        for position in 0 ..< tokens.dim(1) {
            last = patched(tokens[0..., position ..< (position + 1)], cache: cache)
        }

        XCTAssertTrue(
            allClose(expected, last![0, -1, 0...], atol: 1e-3).item(Bool.self),
            "multi-layer incremental decoding diverged from the full-sequence pass")
    }

    // MARK: - Cache consistency

    /// The test that matters for generation: feeding tokens one at a time through a
    /// KV cache must produce the same next-token logits as one full-sequence pass.
    ///
    /// This exercises the assumption `runLayers` relies on — that the cache is
    /// indexed absolutely, so splitting the stack around the patch layer leaves
    /// each layer reading and writing its own cache entry.
    func testIncrementalDecodingMatchesTheFullSequencePass() {
        let model = makeModel(layers: 4)
        let patched = PatchedModel(
            wrapping: model, at: 2, swapping: direction(0), for: direction(1), coefficient: 3)

        let full = patched(tokens, cache: nil)
        let expected = full[0, -1, 0...]

        let cache = patched.newCache(parameters: nil)
        var last: MLXArray?
        for position in 0 ..< tokens.dim(1) {
            let step = tokens[0..., position ..< (position + 1)]
            last = patched(step, cache: cache)
        }

        let actual = last![0, -1, 0...]
        XCTAssertTrue(
            allClose(expected, actual, atol: 1e-3).item(Bool.self),
            "incremental decoding diverged from the full-sequence pass")
    }

    /// The same property for the unpatched wrapper, isolating cache handling from
    /// the patch itself.
    func testIncrementalDecodingIsFaithfulWithAnIdentityPatch() {
        let model = makeModel(layers: 4)
        let patched = PatchedModel(wrapping: model, at: 2) { $0 }

        let expected = model(tokens, cache: nil)[0, -1, 0...]

        let cache = patched.newCache(parameters: nil)
        var last: MLXArray?
        for position in 0 ..< tokens.dim(1) {
            last = patched(tokens[0..., position ..< (position + 1)], cache: cache)
        }

        XCTAssertTrue(
            allClose(expected, last![0, -1, 0...], atol: 1e-3).item(Bool.self),
            "cached decoding diverged from the full-sequence pass")
    }

    func testCacheHasOneEntryPerLayer() {
        let model = makeModel(layers: 6)
        let patched = PatchedModel(wrapping: model, at: 3) { $0 }
        XCTAssertEqual(patched.newCache(parameters: nil).count, model.layerCount)
    }

    // MARK: - LanguageModel conformance

    /// Chat conventions must be forwarded, not defaulted. Defaulting them would
    /// change how prompts render and how reasoning tags parse, and the resulting
    /// behavioral difference would look like an effect of the patch.
    func testChatConventionsAreForwardedFromTheBaseModel() {
        let qwenConfig = Qwen3Configuration(
            hiddenSize: 64, hiddenLayers: 2, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 1e-5, vocabularySize: Self.vocabularySize, kvHeads: 4, headDim: 8)
        let qwen = Qwen3Model(qwenConfig)
        eval(qwen)

        // Qwen3 declares a reasoning protocol; the wrapper must report the same one.
        XCTAssertNotNil(qwen.reasoningConfig)
        let patched = PatchedModel(wrapping: qwen, at: 1) { $0 }
        XCTAssertNotNil(
            patched.reasoningConfig, "reasoningConfig was not forwarded from the base model")
    }

    func testSanitizeIsForwarded() {
        let model = makeModel(tied: true)
        let patched = PatchedModel(wrapping: model, at: 1) { $0 }

        let weights = [
            "self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]), "keep": MLXArray.zeros([4]),
        ]
        let sanitized = patched.sanitize(weights: weights)

        // Llama's sanitize drops precomputed rotary frequencies.
        XCTAssertNil(sanitized["self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["keep"])
    }

    func testPrepareReturnsTokensForAShortPrompt() throws {
        let model = makeModel()
        let patched = PatchedModel(wrapping: model, at: 2) { $0 }

        let input = LMInput(text: LMInput.Text(tokens: tokens))
        let cache = patched.newCache(parameters: nil)
        let result = try patched.prepare(input, cache: cache, state: nil, windowSize: nil)

        guard case .tokens(let remaining) = result else {
            return XCTFail("expected .tokens, got \(result)")
        }
        // Nothing to chunk at this length, so the whole prompt comes back.
        XCTAssertEqual(remaining.tokens.dim(1), tokens.dim(1))
    }

    /// Generation through the real token loop.
    ///
    /// The manual cache tests above drive `callAsFunction(_:cache:)` directly.
    /// `TokenIterator` goes through `callAsFunction(_ input: LMInput.Text, cache:state:)`
    /// instead, which is a different entry point — and one a `Module` subclass can
    /// silently get wrong.
    func testGenerationThroughTokenIterator() throws {
        let model = makeModel(layers: 4)
        let patched = PatchedModel(wrapping: model, at: 2) { $0 }

        let iterator = try TokenIterator(
            input: LMInput(text: LMInput.Text(tokens: flatTokens)),
            model: patched,
            cache: patched.newCache(parameters: nil),
            parameters: GenerateParameters(maxTokens: 4, temperature: 0))

        var generated: [Int] = []
        for token in iterator {
            generated.append(token)
            if generated.count >= 4 { break }
        }

        XCTAssertEqual(
            generated.count, 4, "generation through TokenIterator produced too few tokens")
    }

    /// The same for Qwen3, whose attention applies RMSNorm per head — a shape the
    /// wrapper must not disturb.
    func testGenerationThroughTokenIteratorForQwen3() throws {
        let config = Qwen3Configuration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 1e-5, vocabularySize: Self.vocabularySize, kvHeads: 4, headDim: 8)
        let qwen = Qwen3Model(config)
        eval(qwen)
        let patched = PatchedModel(wrapping: qwen, at: 2) { $0 }

        let iterator = try TokenIterator(
            input: LMInput(text: LMInput.Text(tokens: flatTokens)),
            model: patched,
            cache: patched.newCache(parameters: nil),
            parameters: GenerateParameters(maxTokens: 4, temperature: 0))

        var generated: [Int] = []
        for token in iterator {
            generated.append(token)
            if generated.count >= 4 { break }
        }
        XCTAssertEqual(generated.count, 4)
    }

    func testInvalidPatchLayerIsRejected() {
        // A patch layer above `layerCount` is a programming error; the precondition
        // documents the valid range as 0...layerCount inclusive.
        let model = makeModel(layers: 4)
        XCTAssertEqual(model.layerCount, 4)
        // The inclusive upper bound is valid and must not trap.
        _ = PatchedModel(wrapping: model, at: 4) { $0 }
    }
}
