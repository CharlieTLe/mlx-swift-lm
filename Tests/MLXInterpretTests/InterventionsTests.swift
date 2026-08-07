// Copyright © 2026 Apple Inc.

import MLX
@_spi(Interpret) import MLXInterpret
@_spi(Interpret) import MLXLLM
@_spi(Interpret) import MLXLMCommon
import MLXNN
import XCTest

/// Tests for the intervention primitives.
///
/// These are pure linear algebra, so unlike the lens tests they have exact
/// expected answers and can assert tightly.
final class InterventionsTests: XCTestCase {

    private func basis(_ index: Int, _ size: Int = 8) -> MLXArray {
        var vector = MLXArray.zeros([size], dtype: .float32)
        vector[index] = MLXArray(Float(1))
        return vector
    }

    // MARK: - Steering

    func testSteerAddsAlongTheNormalizedDirection() {
        let hidden = MLXArray([1, 0, 0, 0, 0, 0, 0, 0] as [Float])
        // Deliberately unnormalized: steering must normalize so `alpha` means the
        // same thing regardless of how the caller scaled the direction.
        let direction = 5 * basis(1)

        let result = Interventions.steer(hidden, along: direction, alpha: 2)

        XCTAssertEqual(result[0].item(Float.self), 1, accuracy: 1e-6)
        XCTAssertEqual(result[1].item(Float.self), 2, accuracy: 1e-6)
    }

    // MARK: - Ablation

    func testAblateRemovesTheComponentAlongOneDirection() {
        let hidden = MLXArray([3, 4, 0, 0, 0, 0, 0, 0] as [Float])
        let result = Interventions.ablate(hidden, directions: [basis(0)])

        XCTAssertEqual(result[0].item(Float.self), 0, accuracy: 1e-6)
        XCTAssertEqual(result[1].item(Float.self), 4, accuracy: 1e-6)
    }

    func testAblateIsANoOpForOrthogonalDirections() {
        let hidden = MLXArray([0, 5, 0, 0, 0, 0, 0, 0] as [Float])
        let result = Interventions.ablate(hidden, directions: [basis(0), basis(2)])
        XCTAssertTrue(allClose(result, hidden, atol: 1e-6).item(Bool.self))
    }

    /// Overlapping directions must not over-subtract. Ablating two nearly parallel
    /// directions naively would remove the shared component twice and push the
    /// result past zero into the opposite sign.
    func testAblateHandlesOverlappingDirections() {
        let hidden = MLXArray([1, 0, 0, 0, 0, 0, 0, 0] as [Float])
        let nearlyParallel = Interventions.normalized(basis(0) + 0.01 * basis(1))

        let result = Interventions.ablate(hidden, directions: [basis(0), nearlyParallel])

        // The component along basis(0) is gone, and nothing was over-subtracted.
        XCTAssertEqual(result[0].item(Float.self), 0, accuracy: 1e-5)
        let magnitude = sqrt((result * result).sum()).item(Float.self)
        XCTAssertLessThan(magnitude, 1e-4, "ablation over-subtracted (residual \(magnitude))")
    }

    /// Ablating a direction twice must equal ablating it once.
    func testAblateIsIdempotent() {
        let hidden = MLXArray([3, 4, 5, 0, 0, 0, 0, 0] as [Float])
        let once = Interventions.ablate(hidden, directions: [basis(0)])
        let twice = Interventions.ablate(once, directions: [basis(0)])
        XCTAssertTrue(allClose(once, twice, atol: 1e-6).item(Bool.self))
    }

    func testAblateDropsDependentDirections() {
        let hidden = MLXArray([1, 1, 0, 0, 0, 0, 0, 0] as [Float])
        // The second is an exact multiple of the first and contributes no new axis.
        let result = Interventions.ablate(hidden, directions: [basis(0), 2 * basis(0)])
        XCTAssertEqual(result[0].item(Float.self), 0, accuracy: 1e-6)
        XCTAssertEqual(result[1].item(Float.self), 1, accuracy: 1e-6)
    }

    // MARK: - Coordinate swap

    /// The property the paper's result depends on: the orthogonal remainder is
    /// untouched, so a behavioral change is attributable to the swapped coordinate.
    func testCoordinateSwapPreservesTheOrthogonalRemainder() {
        let hidden = MLXArray([2, 0, 7, 0, 0, 0, 0, 0] as [Float])
        let result = Interventions.coordinateSwap(hidden, from: basis(0), to: basis(1))

        // The `from` coordinate is emptied and its magnitude moved to `to`.
        XCTAssertEqual(result[0].item(Float.self), 0, accuracy: 1e-6)
        XCTAssertEqual(result[1].item(Float.self), 2, accuracy: 1e-6)
        // Everything orthogonal to both is exactly as it was.
        XCTAssertEqual(result[2].item(Float.self), 7, accuracy: 1e-6)
    }

    func testCoordinateSwapWithExplicitCoefficient() {
        let hidden = MLXArray([2, 0, 0, 0, 0, 0, 0, 0] as [Float])
        let result = Interventions.coordinateSwap(
            hidden, from: basis(0), to: basis(1), coefficient: 5)

        XCTAssertEqual(result[0].item(Float.self), -3, accuracy: 1e-6)
        XCTAssertEqual(result[1].item(Float.self), 5, accuracy: 1e-6)
    }

    /// Swapping a direction for itself must change nothing.
    func testCoordinateSwapToItselfIsIdentity() {
        let hidden = MLXArray([2, 3, 4, 0, 0, 0, 0, 0] as [Float])
        let result = Interventions.coordinateSwap(hidden, from: basis(0), to: basis(0))
        XCTAssertTrue(allClose(result, hidden, atol: 1e-6).item(Bool.self))
    }

    /// Swapping an absent concept is a no-op, because the coefficient defaults to
    /// the activation's own projection, which is zero.
    func testCoordinateSwapOfAnAbsentConceptIsANoOp() {
        let hidden = MLXArray([0, 0, 9, 0, 0, 0, 0, 0] as [Float])
        let result = Interventions.coordinateSwap(hidden, from: basis(0), to: basis(1))
        XCTAssertTrue(allClose(result, hidden, atol: 1e-6).item(Bool.self))
    }

    // MARK: - Batched activations

    /// Interventions have to work on live `[batch, position, hidden]` activations,
    /// not just bare vectors, because that is the shape `PatchedModel` hands them.
    func testInterventionsBroadcastOverBatchedActivations() {
        let hidden = MLXArray.zeros([1, 3, 8], dtype: .float32)
        hidden[0, 0, 0] = MLXArray(Float(1))
        hidden[0, 1, 0] = MLXArray(Float(2))
        hidden[0, 2, 2] = MLXArray(Float(5))

        let result = Interventions.coordinateSwap(hidden, from: basis(0), to: basis(1))

        XCTAssertEqual(result.shape, [1, 3, 8])
        // Per-position projection: each position swaps its own amount.
        XCTAssertEqual(result[0, 0, 1].item(Float.self), 1, accuracy: 1e-6)
        XCTAssertEqual(result[0, 1, 1].item(Float.self), 2, accuracy: 1e-6)
        XCTAssertEqual(result[0, 0, 0].item(Float.self), 0, accuracy: 1e-6)
        // A position with nothing along `from` is untouched.
        XCTAssertEqual(result[0, 2, 2].item(Float.self), 5, accuracy: 1e-6)
    }

    // MARK: - Clamping

    /// Clamping must restore the in-span component exactly while leaving the
    /// orthogonal part of the perturbed activation untouched.
    func testClampRestoresTheInSpanComponent() {
        let reference = MLXArray([5, 7, 0, 0, 0, 0, 0, 0] as [Float])
        // Perturbed both inside the clamped span (axis 0) and outside it (axis 2).
        let perturbed = MLXArray([99, 7, 3, 0, 0, 0, 0, 0] as [Float])

        let result = Interventions.clamp(perturbed, toSpanOf: [basis(0)], matching: reference)

        // Inside the span: back to the reference value.
        XCTAssertEqual(result[0].item(Float.self), 5, accuracy: 1e-5)
        // Outside it: the perturbation survives.
        XCTAssertEqual(result[2].item(Float.self), 3, accuracy: 1e-5)
    }

    func testClampWithAnEmptyBasisIsANoOp() {
        let hidden = MLXArray([1, 2, 3, 0, 0, 0, 0, 0] as [Float])
        let result = Interventions.clamp(
            hidden, toSpanOf: [], matching: MLXArray.zeros(like: hidden))
        XCTAssertTrue(allClose(result, hidden, atol: 1e-6).item(Bool.self))
    }

    /// Clamping against the activation itself changes nothing.
    func testClampAgainstItselfIsIdentity() {
        let hidden = MLXArray([1, 2, 3, 4, 0, 0, 0, 0] as [Float])
        let result = Interventions.clamp(
            hidden, toSpanOf: [basis(0), basis(2)], matching: hidden)
        XCTAssertTrue(allClose(result, hidden, atol: 1e-5).item(Bool.self))
    }

    /// After clamping, the activation must agree with the reference on every
    /// clamped axis even when the basis is not axis-aligned.
    func testClampHandlesANonAxisAlignedBasis() {
        let axis = Interventions.normalized(basis(0) + basis(1))
        let reference = MLXArray([2, 2, 9, 0, 0, 0, 0, 0] as [Float])
        let perturbed = MLXArray([-8, 4, 9, 0, 0, 0, 0, 0] as [Float])

        let result = Interventions.clamp(perturbed, toSpanOf: [axis], matching: reference)

        XCTAssertEqual(
            Interventions.project(result, onto: axis).item(Float.self),
            Interventions.project(reference, onto: axis).item(Float.self),
            accuracy: 1e-4)
    }

    // MARK: - Similarity

    func testCosineSimilarity() {
        XCTAssertEqual(Interventions.cosineSimilarity(basis(0), basis(0)), 1, accuracy: 1e-6)
        XCTAssertEqual(Interventions.cosineSimilarity(basis(0), basis(1)), 0, accuracy: 1e-6)
        XCTAssertEqual(Interventions.cosineSimilarity(basis(0), -basis(0)), -1, accuracy: 1e-6)
        // Magnitude must not matter.
        XCTAssertEqual(Interventions.cosineSimilarity(basis(0), 100 * basis(0)), 1, accuracy: 1e-6)
    }

    func testNormalizedHandlesTheZeroVector() {
        let zero = MLXArray.zeros([8], dtype: .float32)
        let result = Interventions.normalized(zero)
        XCTAssertTrue(result.abs().max().item(Float.self).isFinite, "zero vector produced NaN")
    }
}
