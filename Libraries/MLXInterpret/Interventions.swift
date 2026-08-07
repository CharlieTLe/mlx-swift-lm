// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Interpret) import MLXLMCommon

/// Edits to a residual-stream activation.
///
/// All three of these operate in the same space and share one design rule: touch
/// the component the intervention is *about* and leave the orthogonal remainder
/// alone. That rule is what makes the results interpretable. An intervention that
/// also perturbs unrelated directions cannot distinguish "changing this concept
/// changed the answer" from "perturbing the activation at all changed the answer."
@_spi(Interpret)
public enum Interventions {

    /// Add `alpha · direction` to the activation.
    ///
    /// The bluntest intervention, and the least controlled: it changes the
    /// activation's magnitude along `direction` without removing whatever was
    /// there before. Prefer ``coordinateSwap(_:from:to:coefficient:)`` when the
    /// question is "what if it were thinking about B instead of A."
    public static func steer(
        _ hidden: MLXArray, along direction: MLXArray, alpha: Float
    ) -> MLXArray {
        hidden + alpha * normalized(direction).asType(hidden.dtype)
    }

    /// Project `directions` out of the activation.
    ///
    /// This is the paper's ablation experiment: zero the activation's components
    /// along a set of lens directions and see which downstream tasks survive. The
    /// directions are orthonormalized first via Gram-Schmidt, because projecting
    /// out overlapping directions one at a time would over-subtract the shared
    /// component and remove more than asked.
    public static func ablate(_ hidden: MLXArray, directions: [MLXArray]) -> MLXArray {
        let basis = orthonormalBasis(directions, dtype: hidden.dtype)
        var result = hidden
        for direction in basis {
            let coefficient = project(result, onto: direction)
            result = result - coefficient * direction
        }
        return result
    }

    /// Replace the activation's component along `from` with the same amount along
    /// `to`, leaving everything orthogonal to both untouched.
    ///
    /// This is the paper's coordinate swap, and the reason its result is
    /// meaningful: `h − c·v_a + c·v_b` preserves the activation's entire
    /// orthogonal remainder, so a change in the model's answer can be attributed
    /// to the exchanged coordinate rather than to generic disruption.
    ///
    /// - Parameters:
    ///   - coefficient: the amount to exchange. Defaults to the activation's own
    ///     projection onto `from`, which swaps exactly what was there.
    public static func coordinateSwap(
        _ hidden: MLXArray, from source: MLXArray, to target: MLXArray,
        coefficient: Float? = nil
    ) -> MLXArray {
        let sourceUnit = normalized(source).asType(hidden.dtype)
        let targetUnit = normalized(target).asType(hidden.dtype)
        let amount =
            coefficient.map { MLXArray($0).asType(hidden.dtype) }
            ?? project(hidden, onto: sourceUnit)
        return hidden - amount * sourceUnit + amount * targetUnit
    }

    /// How much of `direction` is present in `hidden`, as a scalar.
    ///
    /// Handles both a bare `[hiddenSize]` vector and a batched
    /// `[batch, position, hiddenSize]` activation, in which case the projection is
    /// per-position and broadcasts back correctly.
    public static func project(_ hidden: MLXArray, onto direction: MLXArray) -> MLXArray {
        (hidden * direction).sum(axis: -1, keepDims: true)
    }

    /// Restore `hidden`'s component inside the span of `basis` to whatever it was
    /// in `reference`, leaving the orthogonal part of `hidden` alone.
    ///
    /// This is the paper's *clamping* control, and it is what makes a
    /// remainder-only intervention interpretable. Perturbing an activation along a
    /// direction that is mostly outside J-space still nudges the J-space
    /// coordinates a little, because the direction is not exactly orthogonal to
    /// them. Any behavioral change could then be attributed to J-space after all.
    /// Clamping pins those coordinates so whatever effect remains must come from
    /// outside the span.
    ///
    /// The paper's remainder-swap success rate falls from 28% to 6% once clamped,
    /// so the difference this makes is most of the result.
    ///
    /// - Parameters:
    ///   - basis: spanning directions. Orthonormalized internally.
    ///   - reference: the activation whose in-span component should be preserved.
    public static func clamp(
        _ hidden: MLXArray, toSpanOf basis: [MLXArray], matching reference: MLXArray
    ) -> MLXArray {
        let axes = orthonormalBasis(basis, dtype: hidden.dtype)
        var result = hidden
        for axis in axes {
            let current = project(result, onto: axis)
            let wanted = project(reference.asType(hidden.dtype), onto: axis)
            result = result - current * axis + wanted * axis
        }
        return result
    }

    /// Cosine similarity between two vectors.
    public static func cosineSimilarity(_ a: MLXArray, _ b: MLXArray) -> Float {
        let similarity = (normalized(a) * normalized(b)).sum()
        return similarity.item(Float.self)
    }

    public static func normalized(_ vector: MLXArray) -> MLXArray {
        let norm = sqrt((vector * vector).sum(axis: -1, keepDims: true))
        return vector / maximum(norm, MLXArray(Float(1e-12)).asType(vector.dtype))
    }

    /// Gram-Schmidt, dropping directions that are linearly dependent on those
    /// already accepted.
    public static func orthonormalBasis(_ directions: [MLXArray], dtype: DType) -> [MLXArray] {
        var basis: [MLXArray] = []
        for direction in directions {
            var candidate = direction.asType(dtype)
            for existing in basis {
                candidate = candidate - project(candidate, onto: existing) * existing
            }
            let norm = sqrt((candidate * candidate).sum()).item(Float.self)
            // Dependent within numerical tolerance: it contributes nothing and
            // normalizing it would amplify rounding error into a spurious axis.
            guard norm > 1e-6 else { continue }
            basis.append(candidate / MLXArray(norm).asType(dtype))
        }
        return basis
    }
}
