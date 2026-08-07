// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Interpret) import MLXLMCommon

/// The result of decomposing an activation onto a dictionary of lens vectors.
@_spi(Interpret)
public struct JSpaceDecomposition {

    /// One selected atom and its weight.
    public struct Term {
        public let tokenId: Int
        public let token: String
        public let coefficient: Float
    }

    /// Selected atoms, in the order they were chosen (largest contribution first).
    public let terms: [Term]

    /// The reconstruction: the part of the activation that lies in the span of the
    /// selected lens vectors.
    public let jPart: MLXArray

    /// What the dictionary could not explain, `target − jPart`.
    ///
    /// This is the half of the decomposition the paper's headline result rests on:
    /// swapping the ``jPart`` changes the model's stated answer most of the time,
    /// while swapping the remainder barely does, even though the remainder holds
    /// the overwhelming majority of the variance.
    public let remainder: MLXArray

    /// Fraction of the target's squared norm captured by ``jPart``.
    public let explainedFraction: Float

    public var description: String {
        let body =
            terms
            .map { String(format: "%@×%.3f", $0.token, $0.coefficient) }
            .joined(separator: " + ")
        return String(format: "[%.1f%% explained] ", explainedFraction * 100) + body
    }
}

/// A dictionary of Jacobian-lens vectors at one layer, and sparse nonnegative
/// decomposition onto it.
///
/// The paper defines its "J-space" as the set of points expressible as sparse
/// nonnegative combinations of J-lens vectors. Because those vectors are
/// overcomplete they form a frame rather than a basis, so there is no projection
/// formula — membership and coordinates have to be found by a pursuit algorithm.
///
/// ### Why the dictionary is a candidate set, not the vocabulary
///
/// A lens vector costs one VJP per calibration prompt, so a full-vocabulary
/// dictionary is out of reach: 150k tokens × 12 prompts is 1.8M backward passes.
/// The dictionary is therefore built from a candidate token set, normally the top
/// entries of a readout at the same layer. That is a real limitation, not a
/// detail: the decomposition can only find concepts the candidate set contains.
@_spi(Interpret)
public struct JSpace {

    /// Atoms, each an L2-normalized lens vector paired with its token.
    public let atoms: [(tokenId: Int, token: String, vector: MLXArray)]

    /// Layer these vectors were computed at.
    public let layer: Int

    public init(atoms: [(tokenId: Int, token: String, vector: MLXArray)], layer: Int) {
        self.atoms = atoms
        self.layer = layer
    }

    /// Build a dictionary from an explicit candidate token list.
    public static func build(
        lens: JacobianLens, at layer: Int, candidateTokens: [Int]
    ) -> JSpace {
        let atoms = candidateTokens.map { token in
            (
                tokenId: token,
                token: TokenDisplay.string(for: token, tokenizer: lens.tokenizer),
                vector: lens.lensVector(forToken: token, at: layer)
            )
        }
        return JSpace(atoms: atoms, layer: layer)
    }

    /// Build a dictionary from the tokens a readout of `hidden` surfaces.
    ///
    /// The natural pairing: read the activation first to find out which concepts
    /// are even plausibly present, then build a dictionary from those and
    /// decompose. Costs `candidateCount` VJPs per calibration prompt.
    public static func build(
        lens: JacobianLens, at layer: Int, around hidden: MLXArray, candidateCount: Int = 32
    ) -> JSpace {
        let readout = lens.readout(of: hidden, at: layer, topK: candidateCount)
        return build(lens: lens, at: layer, candidateTokens: readout.entries.map(\.tokenId))
    }

    /// Decompose `target` into a sparse nonnegative combination of atoms.
    ///
    /// Nonnegative orthogonal matching pursuit: repeatedly select the atom with
    /// the largest positive correlation against the current residual, then refit
    /// all selected coefficients by projected gradient descent so earlier choices
    /// are corrected as later ones are added. Plain matching pursuit without the
    /// refit systematically overshoots when atoms overlap, which lens vectors
    /// generally do.
    ///
    /// - Parameters:
    ///   - maxTerms: sparsity cap. The paper uses 25.
    ///   - tolerance: stop when no atom correlates with the residual by more than
    ///     this, relative to the target's norm.
    public func decompose(
        _ target: MLXArray, maxTerms: Int = 25, tolerance: Float = 1e-3
    ) -> JSpaceDecomposition {
        let goal = target.asType(.float32).reshaped([-1])
        let goalNorm = sqrt((goal * goal).sum()).item(Float.self)

        guard !atoms.isEmpty, goalNorm > 0 else {
            return JSpaceDecomposition(
                terms: [],
                jPart: MLXArray.zeros(like: goal),
                remainder: goal,
                explainedFraction: 0)
        }

        // [atomCount, hiddenSize]
        let dictionary = stacked(atoms.map { $0.vector.asType(.float32).reshaped([-1]) })

        var selected: [Int] = []
        var coefficients: [Float] = []
        var residual = goal

        for _ in 0 ..< min(maxTerms, atoms.count) {
            // Correlation of every atom with the residual.
            let correlations = matmul(dictionary, residual)

            // Nonnegativity means only positive correlations can help, and an atom
            // already chosen must not be chosen twice.
            var best = -Float.greatestFiniteMagnitude
            var bestIndex = -1
            let values = correlations.asArray(Float.self)
            for (index, value) in values.enumerated() where !selected.contains(index) {
                if value > best {
                    best = value
                    bestIndex = index
                }
            }

            guard bestIndex >= 0, best > tolerance * goalNorm else { break }

            selected.append(bestIndex)
            coefficients = Self.refit(
                dictionary: dictionary, selected: selected, target: goal,
                initial: coefficients + [best])

            residual = goal - Self.reconstruct(dictionary, selected, coefficients)
        }

        let jPart = Self.reconstruct(dictionary, selected, coefficients)
        let remainder = goal - jPart
        let explained =
            goalNorm > 0
            ? (jPart * jPart).sum().item(Float.self) / (goalNorm * goalNorm)
            : 0

        // Report largest contribution first, which is the order a reader expects
        // even though pursuit does not always select in that order after refitting.
        let ordered = zip(selected, coefficients)
            .sorted { $0.1 > $1.1 }
            .map { index, coefficient in
                JSpaceDecomposition.Term(
                    tokenId: atoms[index].tokenId,
                    token: atoms[index].token,
                    coefficient: coefficient)
            }

        return JSpaceDecomposition(
            terms: ordered,
            jPart: jPart,
            remainder: remainder,
            explainedFraction: explained)
    }

    /// Split `target` into its in-dictionary and out-of-dictionary parts.
    ///
    /// Convenience for the swap experiment, which needs both halves so each can be
    /// perturbed independently.
    public func project(_ target: MLXArray, maxTerms: Int = 25) -> (
        jPart: MLXArray, remainder: MLXArray
    ) {
        let decomposition = decompose(target, maxTerms: maxTerms)
        return (decomposition.jPart, decomposition.remainder)
    }

    // MARK: - Pursuit internals

    private static func reconstruct(
        _ dictionary: MLXArray, _ selected: [Int], _ coefficients: [Float]
    ) -> MLXArray {
        guard !selected.isEmpty else {
            return MLXArray.zeros([dictionary.dim(1)], dtype: .float32)
        }
        let subset = dictionary[MLXArray(selected.map { Int32($0) })]
        return matmul(MLXArray(coefficients), subset)
    }

    /// Nonnegative least squares over the selected atoms, by projected gradient
    /// descent.
    ///
    /// A closed-form solve would ignore the nonnegativity constraint and can
    /// return negative weights, which are meaningless here — a concept is present
    /// or absent, not anti-present. Projected gradient keeps every coefficient at
    /// or above zero. The step size is bounded by the largest eigenvalue of the
    /// Gram matrix, estimated cheaply by its trace.
    private static func refit(
        dictionary: MLXArray, selected: [Int], target: MLXArray, initial: [Float],
        iterations: Int = 64
    ) -> [Float] {
        let subset = dictionary[MLXArray(selected.map { Int32($0) })]  // [k, d]
        let gram = matmul(subset, subset.transposed())  // [k, k]
        let projections = matmul(subset, target)  // [k]

        // Every atom is unit-norm, so the Gram matrix's trace is exactly the number
        // of selected atoms — a safe Lipschitz bound with no computation needed.
        let stepSize = 1.0 / max(Float(selected.count), 1e-6)

        var x = MLXArray(initial.map { max($0, 0) })
        for _ in 0 ..< iterations {
            let gradient = matmul(gram, x) - projections
            x = maximum(x - stepSize * gradient, MLXArray(Float(0)))
        }
        eval(x)
        return x.asArray(Float.self)
    }
}
