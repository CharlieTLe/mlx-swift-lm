// Copyright © 2026 Apple Inc.

import MLX
@_spi(Interpret) import MLXInterpret
@_spi(Interpret) import MLXLLM
@_spi(Interpret) import MLXLMCommon
import MLXNN
import XCTest

/// Tests for difference-of-means concept directions.
///
/// These matter more than their size suggests. A `JacobianLens` lens vector is by
/// construction an element of J-space, so decomposing one and asking how much of it
/// is J-space expressible is circular. `ConceptVector` exists to break that circle
/// by producing a direction from activations alone, which means any J-space
/// measurement built on it inherits these tests' correctness.
final class ConceptVectorTests: XCTestCase {

    private static let vocabularySize = 256

    private func makeModel(layers: Int = 4) -> LlamaModel {
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: layers, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 1e-5, vocabularySize: Self.vocabularySize, kvHeads: 4)
        let model = LlamaModel(config)
        eval(model)
        return model
    }

    private func makeTokenizer() -> StubTokenizer {
        StubTokenizer(vocabularySize: Self.vocabularySize)
    }

    /// Two structurally parallel prompt sets differing in one word, drawn from the
    /// calibration corpus vocabulary so the stub tokenizer resolves every token.
    private let positive = [
        "snow covered the fields for most of the winter",
        "snow covered the bridge before dark",
        "snow covered the village that year",
    ]
    private let negative = [
        "water covered the fields for most of the winter",
        "water covered the bridge before dark",
        "water covered the village that year",
    ]

    // MARK: - Shape and normalization

    func testDirectionIsUnitNormAndFinite() {
        let model = makeModel()
        let tokenizer = makeTokenizer()

        let direction = ConceptVector.differenceOfMeans(
            model: model, tokenizer: tokenizer,
            positive: positive, negative: negative, layer: 2)

        XCTAssertEqual(direction.shape, [model.hiddenSize])
        let norm = sqrt((direction * direction).sum()).item(Float.self)
        XCTAssertTrue(norm.isFinite, "direction is not finite")
        XCTAssertEqual(norm, 1.0, accuracy: 1e-4, "direction is not L2-normalized")
    }

    // MARK: - The direction actually separates the sets

    /// The load-bearing test: positive prompts must project *higher* onto the
    /// direction than negative ones.
    ///
    /// This is what catches a sign flip, which is otherwise invisible — a flipped
    /// direction is still unit-norm, still finite, and still the right shape, and
    /// every intervention built on it would push the opposite way.
    func testPositivePromptsProjectHigherThanNegative() {
        let model = makeModel()
        let tokenizer = makeTokenizer()
        let layer = 2

        let direction = ConceptVector.differenceOfMeans(
            model: model, tokenizer: tokenizer,
            positive: positive, negative: negative, layer: layer)

        let positiveMean = ConceptVector.meanActivation(
            model: model, tokenizer: tokenizer, prompts: positive, layer: layer)
        let negativeMean = ConceptVector.meanActivation(
            model: model, tokenizer: tokenizer, prompts: negative, layer: layer)

        let positiveProjection = (positiveMean * direction).sum().item(Float.self)
        let negativeProjection = (negativeMean * direction).sum().item(Float.self)

        XCTAssertGreaterThan(
            positiveProjection, negativeProjection,
            "the direction does not separate the contrast sets; a sign error would look like this")
    }

    /// Swapping the sets must negate the direction exactly.
    func testSwappingTheSetsNegatesTheDirection() {
        let model = makeModel()
        let tokenizer = makeTokenizer()

        let forward = ConceptVector.differenceOfMeans(
            model: model, tokenizer: tokenizer,
            positive: positive, negative: negative, layer: 2)
        let reversed = ConceptVector.differenceOfMeans(
            model: model, tokenizer: tokenizer,
            positive: negative, negative: positive, layer: 2)

        XCTAssertTrue(
            allClose(forward, -reversed, atol: 1e-5).item(Bool.self),
            "reversing the contrast sets did not negate the direction")
    }

    /// Identical sets have no direction to find, and the zero-norm guard must keep
    /// the result finite rather than dividing by zero.
    func testIdenticalSetsProduceAFiniteResult() {
        let model = makeModel()
        let tokenizer = makeTokenizer()

        let direction = ConceptVector.differenceOfMeans(
            model: model, tokenizer: tokenizer,
            positive: positive, negative: positive, layer: 2)

        XCTAssertEqual(direction.shape, [model.hiddenSize])
        let magnitude = direction.abs().max().item(Float.self)
        XCTAssertTrue(magnitude.isFinite, "identical sets produced NaN or infinity")
        XCTAssertLessThan(magnitude, 1e-3, "identical sets produced a non-zero direction")
    }

    // MARK: - meanActivation

    /// With one prompt there is nothing to average, so the mean must be that
    /// prompt's activation exactly. This pins the layer and position plumbing.
    func testMeanOverOnePromptEqualsThatActivation() {
        let model = makeModel()
        let tokenizer = makeTokenizer()
        let layer = 2
        let prompt = positive[0]

        let mean = ConceptVector.meanActivation(
            model: model, tokenizer: tokenizer, prompts: [prompt], layer: layer)

        let tokens = MLXArray(
            tokenizer.encode(text: prompt, addSpecialTokens: false).map { Int32($0) })[
                .newAxis, .ellipsis]
        let expected = Activations.vector(from: model.hiddenState(of: tokens, at: layer))
            .asType(.float32)

        XCTAssertTrue(
            allClose(mean, expected, atol: 1e-5).item(Bool.self),
            "mean over a single prompt diverged from that prompt's activation")
    }

    /// The mean of two prompts is the midpoint of their activations.
    func testMeanOfTwoPromptsIsTheirMidpoint() {
        let model = makeModel()
        let tokenizer = makeTokenizer()
        let layer = 2

        let mean = ConceptVector.meanActivation(
            model: model, tokenizer: tokenizer, prompts: [positive[0], positive[1]], layer: layer)

        func activation(_ prompt: String) -> MLXArray {
            let tokens = MLXArray(
                tokenizer.encode(text: prompt, addSpecialTokens: false).map { Int32($0) })[
                    .newAxis, .ellipsis]
            return Activations.vector(from: model.hiddenState(of: tokens, at: layer))
                .asType(.float32)
        }
        let expected = (activation(positive[0]) + activation(positive[1])) / 2

        XCTAssertTrue(allClose(mean, expected, atol: 1e-5).item(Bool.self))
    }

    /// Reading at a different depth must give a different direction, or the `layer`
    /// argument is being ignored.
    func testLayerArgumentChangesTheResult() {
        let model = makeModel(layers: 6)
        let tokenizer = makeTokenizer()

        let shallow = ConceptVector.differenceOfMeans(
            model: model, tokenizer: tokenizer,
            positive: positive, negative: negative, layer: 1)
        let deep = ConceptVector.differenceOfMeans(
            model: model, tokenizer: tokenizer,
            positive: positive, negative: negative, layer: 5)

        let similarity = Interventions.cosineSimilarity(shallow, deep)
        XCTAssertLessThan(
            abs(similarity), 0.999,
            "the layer argument had no effect on the direction (cosine \(similarity))")
    }

    /// Likewise the position argument, on a prompt long enough for positions to differ.
    func testPositionArgumentChangesTheResult() {
        let model = makeModel()
        let tokenizer = makeTokenizer()

        let last = ConceptVector.meanActivation(
            model: model, tokenizer: tokenizer, prompts: positive, layer: 2, position: -1)
        let first = ConceptVector.meanActivation(
            model: model, tokenizer: tokenizer, prompts: positive, layer: 2, position: 0)

        XCTAssertFalse(
            allClose(last, first, atol: 1e-4).item(Bool.self),
            "the position argument had no effect")
    }
}
