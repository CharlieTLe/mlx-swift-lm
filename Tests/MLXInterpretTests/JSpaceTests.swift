// Copyright © 2026 Apple Inc.

import MLX
@_spi(Interpret) import MLXInterpret
@_spi(Interpret) import MLXLLM
@_spi(Interpret) import MLXLMCommon
import MLXNN
import XCTest

/// Tests for sparse nonnegative decomposition.
///
/// Most of these use a synthetic dictionary rather than real lens vectors. That is
/// deliberate: with orthonormal atoms and a known combination there is an exact
/// right answer, so the pursuit algorithm can be tested for correctness rather
/// than merely for plausibility.
final class JSpaceTests: XCTestCase {

    private static let dimension = 16

    /// Dictionary of distinct one-hot atoms, so coefficients are recoverable exactly.
    private func orthonormalSpace(count: Int) -> JSpace {
        let atoms = (0 ..< count).map { index -> (tokenId: Int, token: String, vector: MLXArray) in
            var vector = MLXArray.zeros([Self.dimension], dtype: .float32)
            vector[index] = MLXArray(Float(1))
            return (tokenId: index, token: "t\(index)", vector: vector)
        }
        return JSpace(atoms: atoms, layer: 0)
    }

    private func vector(_ values: [Int: Float]) -> MLXArray {
        var result = MLXArray.zeros([Self.dimension], dtype: .float32)
        for (index, value) in values {
            result[index] = MLXArray(value)
        }
        return result
    }

    // MARK: - Recovery

    func testRecoversAKnownSparseNonnegativeCombination() {
        let space = orthonormalSpace(count: 8)
        let target = vector([1: 3.0, 4: 1.5, 6: 0.5])

        let decomposition = space.decompose(target, maxTerms: 8)

        let recovered = Dictionary(
            uniqueKeysWithValues: decomposition.terms.map { ($0.tokenId, $0.coefficient) })
        XCTAssertEqual(recovered[1] ?? 0, 3.0, accuracy: 1e-2)
        XCTAssertEqual(recovered[4] ?? 0, 1.5, accuracy: 1e-2)
        XCTAssertEqual(recovered[6] ?? 0, 0.5, accuracy: 1e-2)

        XCTAssertEqual(decomposition.explainedFraction, 1.0, accuracy: 1e-2)
        XCTAssertLessThan(
            decomposition.remainder.abs().max().item(Float.self), 1e-2,
            "an exactly representable target left a remainder")
    }

    /// Terms are reported largest-first, which is the order a reader will assume.
    func testTermsAreOrderedByDescendingCoefficient() {
        let space = orthonormalSpace(count: 8)
        let target = vector([0: 0.5, 3: 4.0, 5: 2.0])

        let coefficients = space.decompose(target, maxTerms: 8).terms.map(\.coefficient)
        XCTAssertEqual(coefficients, coefficients.sorted(by: >))
    }

    /// A component pointing *against* an atom must not produce a negative
    /// coefficient — the whole point of the nonnegativity constraint is that a
    /// concept is present or absent, never anti-present.
    func testCoefficientsAreNeverNegative() {
        let space = orthonormalSpace(count: 8)
        let target = vector([1: 2.0, 2: -3.0])

        let decomposition = space.decompose(target, maxTerms: 8)

        for term in decomposition.terms {
            XCTAssertGreaterThanOrEqual(
                term.coefficient, 0, "negative coefficient for \(term.token)")
        }
        // The negative direction cannot be represented, so it stays in the remainder.
        XCTAssertEqual(decomposition.remainder[2].item(Float.self), -3.0, accuracy: 1e-2)
    }

    // MARK: - Sparsity

    func testRespectsTheSparsityCap() {
        let space = orthonormalSpace(count: 12)
        let target = vector(Dictionary(uniqueKeysWithValues: (0 ..< 12).map { ($0, Float(1)) }))

        for cap in [1, 3, 7] {
            let decomposition = space.decompose(target, maxTerms: cap)
            XCTAssertLessThanOrEqual(decomposition.terms.count, cap)
        }
    }

    /// More atoms must explain at least as much as fewer.
    func testExplainedFractionIsMonotoneInTheSparsityCap() {
        let space = orthonormalSpace(count: 12)
        let target = vector([0: 4.0, 3: 3.0, 5: 2.0, 9: 1.0])

        var previous: Float = -1
        for cap in [1, 2, 3, 4] {
            let explained = space.decompose(target, maxTerms: cap).explainedFraction
            XCTAssertGreaterThanOrEqual(
                explained, previous - 1e-4,
                "explained fraction fell when the cap rose to \(cap)")
            previous = explained
        }
        XCTAssertEqual(previous, 1.0, accuracy: 1e-2)
    }

    /// Pursuit must take the largest component first.
    func testSelectsTheStrongestComponentFirst() {
        let space = orthonormalSpace(count: 8)
        let target = vector([2: 0.4, 5: 9.0])

        let decomposition = space.decompose(target, maxTerms: 1)
        XCTAssertEqual(decomposition.terms.count, 1)
        XCTAssertEqual(decomposition.terms[0].tokenId, 5)
    }

    // MARK: - Structural invariants

    /// The split has to be exhaustive, or "swap the jPart, leave the remainder"
    /// would silently drop part of the activation.
    func testJPartAndRemainderSumToTheTarget() {
        let space = orthonormalSpace(count: 8)
        let target = vector([0: 1.0, 1: -2.0, 3: 4.0, 7: 0.25])

        let decomposition = space.decompose(target, maxTerms: 4)
        let reconstructed = decomposition.jPart + decomposition.remainder

        XCTAssertTrue(
            allClose(reconstructed, target, atol: 1e-4).item(Bool.self),
            "jPart + remainder does not reconstruct the target")
    }

    func testProjectAgreesWithDecompose() {
        let space = orthonormalSpace(count: 8)
        let target = vector([1: 2.0, 4: 1.0])

        let (jPart, remainder) = space.project(target, maxTerms: 4)
        let decomposition = space.decompose(target, maxTerms: 4)

        XCTAssertTrue(allClose(jPart, decomposition.jPart, atol: 1e-5).item(Bool.self))
        XCTAssertTrue(allClose(remainder, decomposition.remainder, atol: 1e-5).item(Bool.self))
    }

    // MARK: - Degenerate inputs

    func testEmptyDictionaryExplainsNothing() {
        let space = JSpace(atoms: [], layer: 0)
        let target = vector([0: 1.0])

        let decomposition = space.decompose(target)

        XCTAssertTrue(decomposition.terms.isEmpty)
        XCTAssertEqual(decomposition.explainedFraction, 0)
        XCTAssertTrue(allClose(decomposition.remainder, target, atol: 1e-6).item(Bool.self))
    }

    func testZeroTargetDecomposesToNothing() {
        let space = orthonormalSpace(count: 4)
        let decomposition = space.decompose(MLXArray.zeros([Self.dimension], dtype: .float32))

        XCTAssertTrue(decomposition.terms.isEmpty)
        XCTAssertEqual(decomposition.explainedFraction, 0)
    }

    /// A target orthogonal to every atom is entirely outside the span.
    func testTargetOutsideTheSpanIsAllRemainder() {
        let space = orthonormalSpace(count: 4)
        let target = vector([10: 5.0])

        let decomposition = space.decompose(target, maxTerms: 4)

        XCTAssertTrue(decomposition.terms.isEmpty)
        XCTAssertEqual(decomposition.explainedFraction, 0, accuracy: 1e-4)
    }

    // MARK: - Overlapping atoms

    /// Real lens vectors are not orthogonal. With overlapping atoms, plain matching
    /// pursuit overshoots; the projected-gradient refit is what keeps the
    /// reconstruction accurate, so this is the test that justifies it.
    func testReconstructsAccuratelyWithOverlappingAtoms() {
        var atoms: [(tokenId: Int, token: String, vector: MLXArray)] = []
        for index in 0 ..< 6 {
            var raw = MLXArray.zeros([Self.dimension], dtype: .float32)
            raw[index] = MLXArray(Float(1))
            // Substantial overlap with a shared direction.
            raw[15] = MLXArray(Float(0.6))
            atoms.append(
                (tokenId: index, token: "t\(index)", vector: Interventions.normalized(raw)))
        }
        let space = JSpace(atoms: atoms, layer: 0)

        // A genuine nonnegative combination of three overlapping atoms.
        let target = 2.0 * atoms[0].vector + 1.0 * atoms[2].vector + 0.5 * atoms[4].vector

        let decomposition = space.decompose(target, maxTerms: 6)

        XCTAssertGreaterThan(
            decomposition.explainedFraction, 0.95,
            "overlapping atoms reconstructed poorly (\(decomposition.explainedFraction))")
        for term in decomposition.terms {
            XCTAssertGreaterThanOrEqual(term.coefficient, 0)
        }
    }

    // MARK: - Integration with a real model

    /// End-to-end against real lens vectors: a dictionary built around an
    /// activation should explain a non-trivial share of it.
    func testDecomposesARealActivation() {
        let vocabularySize = 256
        let config = LlamaConfiguration(
            hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 1e-5, vocabularySize: vocabularySize, kvHeads: 4)
        let model = LlamaModel(config)
        eval(model)

        let tokenizer = StubTokenizer(vocabularySize: vocabularySize)
        let lens = JacobianLens(
            model: model, tokenizer: tokenizer,
            corpus: CalibrationCorpus.standard(tokenizer: tokenizer, count: 3, maxTokens: 8))

        let tokens = MLXArray(
            tokenizer.encode(text: "the doctor recommended rest").map { Int32($0) })[
                .newAxis, .ellipsis]
        let layer = 2
        let hidden = Activations.vector(from: model.hiddenState(of: tokens, at: layer))
            .asType(.float32)

        let space = JSpace.build(lens: lens, at: layer, around: hidden, candidateCount: 8)
        XCTAssertEqual(space.atoms.count, 8)

        let decomposition = space.decompose(hidden, maxTerms: 8)

        XCTAssertGreaterThan(decomposition.explainedFraction, 0)
        XCTAssertLessThanOrEqual(decomposition.explainedFraction, 1.0 + 1e-4)
        XCTAssertTrue(
            allClose(decomposition.jPart + decomposition.remainder, hidden, atol: 1e-3)
                .item(Bool.self))
        XCTAssertFalse(decomposition.description.isEmpty)
    }
}
