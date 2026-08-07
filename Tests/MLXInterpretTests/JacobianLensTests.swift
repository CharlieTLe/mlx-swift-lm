// Copyright © 2026 Apple Inc.

import MLX
@_spi(Interpret) import MLXInterpret
@_spi(Interpret) import MLXLLM
@_spi(Interpret) import MLXLMCommon
import MLXNN
import XCTest

/// Calibration tests for the Jacobian lens.
///
/// The central identity exploited throughout: at the *last* layer boundary the
/// remaining stack is empty, so `J` is the identity and the Jacobian lens must
/// reduce exactly to the logit lens. That gives a closed-form expected answer for
/// an otherwise numerical procedure.
final class JacobianLensTests: XCTestCase {

    private static let vocabularySize = 256

    private func makeModel(layers: Int = 4) -> LlamaModel {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: layers, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 1e-5, vocabularySize: Self.vocabularySize, kvHeads: 4)
        let model = LlamaModel(config)
        eval(model)
        return model
    }

    private func makeLens(
        model: LlamaModel, tokenizer: StubTokenizer, prompts: Int = 4
    ) -> JacobianLens {
        JacobianLens(
            model: model,
            tokenizer: tokenizer,
            corpus: CalibrationCorpus.standard(tokenizer: tokenizer, count: prompts, maxTokens: 10)
        )
    }

    private func makeTokenizer() -> StubTokenizer {
        StubTokenizer(vocabularySize: Self.vocabularySize)
    }

    // MARK: - Identity at the top of the stack

    /// With no blocks left to traverse, `J̄h` must be `h` itself.
    func testJacobianVectorProductIsIdentityAtLastLayer() {
        let model = makeModel()
        let lens = makeLens(model: model, tokenizer: makeTokenizer())

        let direction = MLXRandom.normal([model.hiddenSize], key: MLXRandom.key(1)).asType(.float32)
        let mapped = lens.jacobianVectorProduct(of: direction, at: model.layerCount)

        XCTAssertEqual(mapped.shape, [model.hiddenSize])
        let error = (mapped - direction).abs().max().item(Float.self)
        let scale = max(direction.abs().max().item(Float.self), 1e-6)
        XCTAssertLessThan(
            error / scale, 1e-3,
            "J is not the identity at the last layer boundary (relative error \(error / scale))")
    }

    /// The consequence: the two lenses must agree there.
    func testJacobianLensMatchesLogitLensAtLastLayer() {
        let model = makeModel()
        let tokenizer = makeTokenizer()
        let jacobian = makeLens(model: model, tokenizer: tokenizer)
        let logit = LogitLens(model: model, tokenizer: tokenizer)

        let hidden = MLXRandom.normal([model.hiddenSize], key: MLXRandom.key(2)).asType(.float32)
        let layer = model.layerCount

        let jacobianReadout = jacobian.readout(of: hidden, at: layer, topK: 5)
        let logitReadout = logit.readout(of: hidden, at: layer, topK: 5)

        XCTAssertEqual(
            jacobianReadout.entries.map(\.tokenId),
            logitReadout.entries.map(\.tokenId),
            "Jacobian and logit lens rankings diverge where J is the identity")
    }

    /// Below the top, the lenses should genuinely differ — otherwise the Jacobian
    /// is being ignored and the whole exercise is a no-op.
    func testJacobianLensDiffersFromLogitLensMidStack() {
        let model = makeModel(layers: 6)
        let tokenizer = makeTokenizer()
        let jacobian = makeLens(model: model, tokenizer: tokenizer)
        let logit = LogitLens(model: model, tokenizer: tokenizer)

        let tokens = MLXArray(
            tokenizer.encode(text: "the committee published its findings").map { Int32($0) })[
                .newAxis, .ellipsis]
        let hidden = Activations.vector(from: model.hiddenState(of: tokens, at: 2)).asType(
            .float32)

        let jacobianReadout = jacobian.readout(of: hidden, at: 2, topK: 5)
        let logitReadout = logit.readout(of: hidden, at: 2, topK: 5)

        XCTAssertNotEqual(
            jacobianReadout.entries.map(\.tokenId),
            logitReadout.entries.map(\.tokenId),
            "the Jacobian had no effect on the readout mid-stack")
    }

    // MARK: - Linearity

    /// A Jacobian-vector product is linear in the vector, so doubling the input
    /// must double the output. This catches step-size bugs, because a
    /// central-difference estimate that failed to rescale by the input magnitude
    /// would come out identical for both.
    func testJacobianVectorProductIsLinearInTheDirection() {
        let model = makeModel(layers: 3)
        let lens = makeLens(model: model, tokenizer: makeTokenizer(), prompts: 3)

        let direction = MLXRandom.normal([model.hiddenSize], key: MLXRandom.key(4)).asType(.float32)
        let single = lens.jacobianVectorProduct(of: direction, at: 1)
        let doubled = lens.jacobianVectorProduct(of: 2 * direction, at: 1)

        let error = (doubled - 2 * single).abs().max().item(Float.self)
        let scale = max(doubled.abs().max().item(Float.self), 1e-6)
        XCTAssertLessThan(
            error / scale, 5e-2,
            "J̄ is not linear in the direction (relative error \(error / scale))")
    }

    func testZeroDirectionMapsToZero() {
        let model = makeModel(layers: 2)
        let lens = makeLens(model: model, tokenizer: makeTokenizer(), prompts: 2)
        let mapped = lens.jacobianVectorProduct(
            of: MLXArray.zeros([model.hiddenSize], dtype: .float32), at: 1)
        XCTAssertEqual(mapped.abs().max().item(Float.self), 0)
    }

    // MARK: - Lens vectors

    func testLensVectorIsNormalizedAndFinite() {
        let model = makeModel(layers: 3)
        let lens = makeLens(model: model, tokenizer: makeTokenizer(), prompts: 3)

        let vector = lens.lensVector(forToken: 17, at: 1)
        XCTAssertEqual(vector.shape, [model.hiddenSize])

        let norm = sqrt((vector * vector).sum()).item(Float.self)
        XCTAssertTrue(norm.isFinite)
        XCTAssertEqual(norm, 1.0, accuracy: 1e-4, "lens vector is not L2-normalized")
    }

    /// The round trip that gives lens vectors their meaning: the direction built
    /// to raise a token's logit must, when read back through the lens, actually
    /// rank that token near the top.
    ///
    /// This ties the reverse-mode path (which builds the vector) to the
    /// central-difference path (which reads it out). A sign error or a basis
    /// mismatch between the two would show up here and almost nowhere else.
    func testLensVectorReadsBackItsOwnToken() {
        let model = makeModel(layers: 3)
        let tokenizer = makeTokenizer()
        let lens = makeLens(model: model, tokenizer: tokenizer, prompts: 4)

        let layer = 1
        // A handful of arbitrary but valid ids, avoiding the special-token block.
        for token in [11, 29, 63] {
            let vector = lens.lensVector(forToken: token, at: layer)
            let readout = lens.readout(of: vector, at: layer, topK: 10)
            let rank = readout.entries.firstIndex { $0.tokenId == token }
            XCTAssertNotNil(
                rank,
                "token \(token) absent from the readout of its own lens vector; "
                    + "top was \(readout.entries.prefix(3).map(\.tokenId))")
        }
    }

    /// Different tokens must produce different directions.
    func testLensVectorsAreTokenSpecific() {
        let model = makeModel(layers: 3)
        let lens = makeLens(model: model, tokenizer: makeTokenizer(), prompts: 3)

        let a = lens.lensVector(forToken: 11, at: 1)
        let b = lens.lensVector(forToken: 42, at: 1)
        let similarity = (a * b).sum().item(Float.self)

        XCTAssertLessThan(
            abs(similarity), 0.99,
            "lens vectors for distinct tokens are nearly identical (cosine \(similarity))")
    }

    // MARK: - Readout bookkeeping

    /// A readout has to carry its calibration count, since the number is part of
    /// interpreting it rather than metadata about it.
    func testReadoutRecordsCalibrationPromptCount() {
        let model = makeModel(layers: 2)
        let tokenizer = makeTokenizer()
        let lens = makeLens(model: model, tokenizer: tokenizer, prompts: 5)

        let hidden = MLXRandom.normal([model.hiddenSize], key: MLXRandom.key(9)).asType(.float32)
        let readout = lens.readout(of: hidden, at: 1, topK: 3)

        XCTAssertEqual(readout.calibrationPromptCount, 5)
        XCTAssertEqual(readout.entries.count, 3)
        XCTAssertEqual(readout.layer, 1)
        XCTAssertEqual(readout.depthPercent, 50, accuracy: 1e-9)

        // The logit lens has nothing to average, so it reports no count.
        let logitReadout = LogitLens(model: model, tokenizer: tokenizer)
            .readout(of: hidden, at: 1, topK: 3)
        XCTAssertNil(logitReadout.calibrationPromptCount)
    }

    func testReadoutEntriesAreSortedByDescendingScore() {
        let model = makeModel(layers: 2)
        let lens = makeLens(model: model, tokenizer: makeTokenizer(), prompts: 2)
        let hidden = MLXRandom.normal([model.hiddenSize], key: MLXRandom.key(21)).asType(.float32)
        let readout = lens.readout(of: hidden, at: 1, topK: 8)

        let scores = readout.entries.map(\.score)
        XCTAssertEqual(scores, scores.sorted(by: >), "readout is not sorted by score")

        let probabilities = readout.entries.map(\.probability)
        for probability in probabilities {
            XCTAssertGreaterThanOrEqual(probability, 0)
            XCTAssertLessThanOrEqual(probability, 1)
        }
    }

    // MARK: - Sweeps

    func testSweepCoversEveryLayerBoundary() {
        let model = makeModel(layers: 4)
        let tokenizer = makeTokenizer()
        let lens = makeLens(model: model, tokenizer: tokenizer, prompts: 2)

        let tokens = MLXArray(tokenizer.encode(text: "snow covered the fields").map { Int32($0) })
        let readouts = lens.sweep(tokens: tokens, topK: 3)

        XCTAssertEqual(readouts.count, model.layerCount + 1)
        XCTAssertEqual(readouts.map(\.layer), Array(0 ... model.layerCount))
        for readout in readouts {
            XCTAssertEqual(readout.entries.count, 3)
            XCTAssertTrue(readout.entropy.isFinite)
        }
    }
}
