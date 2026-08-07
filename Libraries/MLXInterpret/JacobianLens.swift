// Copyright © 2026 Apple Inc.

import Foundation
import MLX
@_spi(Interpret) import MLXLMCommon

/// The prompt set a ``JacobianLens`` averages its Jacobian over.
///
/// The paper averages over roughly a thousand pretraining-like prompts, and that
/// averaging is not incidental: it is what separates a concept the model is
/// *generally poised to speak about* from one that merely happens to be
/// verbalizable in a single context. A lens calibrated on one prompt is a
/// statement about that prompt, not about the model.
///
/// The built-in corpus is far smaller than the paper's. Treat
/// ``Readout/calibrationPromptCount`` as part of every result.
@_spi(Interpret)
public struct CalibrationCorpus {

    /// Tokenized prompts, each already truncated to the configured length.
    public let prompts: [[Int]]

    public init(prompts: [[Int]]) {
        self.prompts = prompts.filter { $0.count >= 2 }
    }

    /// Generic declarative English prose, deliberately spread across registers
    /// and topics so the average is not dominated by one domain.
    ///
    /// These stand in for a pretraining sample. They are intentionally short:
    /// a directional derivative runs two forward passes over the tail of the
    /// stack per prompt, and attention cost grows with the square of length.
    public static let defaultPrompts: [String] = [
        "The capital city grew quickly during the last century.",
        "She opened the letter and read it twice before speaking.",
        "Water boils at a lower temperature at high altitude.",
        "The committee published its findings the following spring.",
        "He kept a notebook of every bird he saw that year.",
        "Manufacturing output fell sharply in the second quarter.",
        "The novel begins with a description of the harbour.",
        "Most of the students had already finished the assignment.",
        "Iron rusts when it is left exposed to damp air.",
        "The train was delayed again because of the weather.",
        "Her argument rested on a single disputed assumption.",
        "They built the bridge out of local stone and timber.",
        "The recipe calls for butter, flour, sugar, and salt.",
        "Astronomers measured the distance using parallax.",
        "The election results surprised nearly every commentator.",
        "A small crowd had gathered outside the courthouse.",
        "The software update fixed several longstanding problems.",
        "Fishing boats returned to the village before dark.",
        "Historians disagree about the causes of the collapse.",
        "The doctor recommended rest and plenty of fluids.",
        "Snow covered the fields for most of the winter.",
        "He translated the poem without losing its rhythm.",
        "The museum acquired the collection in nineteen sixty.",
        "Prices rose faster than wages for several years.",
    ]

    /// Tokenize the built-in prompts.
    ///
    /// - Parameters:
    ///   - count: how many prompts to use. Lower is faster and noisier.
    ///   - maxTokens: truncation length per prompt.
    public static func standard(
        tokenizer: any Tokenizer, count: Int = 12, maxTokens: Int = 24
    ) -> CalibrationCorpus {
        let selected = defaultPrompts.prefix(max(1, count))
        let tokenized = selected.map { prompt in
            Array(tokenizer.encode(text: prompt, addSpecialTokens: false).prefix(maxTokens))
        }
        return CalibrationCorpus(prompts: tokenized)
    }
}

/// The Jacobian lens: describe a residual-stream activation by its first-order
/// causal effect on the model's output.
///
/// Where the logit lens asks *"what would this activation say if the rest of the
/// network did nothing,"* the Jacobian lens asks *"what does this activation
/// actually push the output toward, given everything the rest of the network will
/// do to it."* Formally the readout is `softmax(W_U · norm(J_ℓ h))`, with `J_ℓ`
/// the Jacobian of the remaining blocks averaged over positions and a corpus of
/// prompts.
///
/// ### Why the full Jacobian is never built
///
/// `J_ℓ` is `hiddenSize × hiddenSize`, which is millions of entries and would
/// need one derivative evaluation per column. It is also never needed: a readout
/// only requires the single product `J̄h`, and because every calibration prompt
/// shares the same direction `h`, that product costs a fixed two forward passes
/// *per prompt* regardless of model width.
///
/// ### Why central differences instead of forward-mode autodiff
///
/// `J̄h` is exactly a Jacobian-vector product, which `MLX.jvp` computes directly —
/// but not through a transformer. As of mlx-swift 0.31.6, which vendors MLX
/// 0.31.1, forward mode fails on two things:
///
/// - **Grouped-query attention**, with `[addmm] Got 0 dimension input`. Every
///   modern LLM uses fewer KV heads than query heads, so this one is decisive.
/// - **The symbolic causal mask**, via a raw C++ `assert` in `Select::jvp` that
///   calls `abort()` where no Swift error handler can intervene. That failure is
///   already fixed on `ml-explore/mlx` main and should resolve on a submodule
///   bump; the GQA failure is the one that persists.
///
/// Because the second failure is an `abort()` rather than an error, attempting
/// forward mode is not merely slow or wrong, it is process-fatal — so this type
/// does not try it even as an optimization. `AutodiffCapabilityTests` records the
/// precise boundaries.
///
/// A central difference costs the same two forward passes, is second-order
/// accurate, and works through every op. `ResidualStreamReentryTests` validates
/// it against reverse mode through the adjoint identity `⟨c, Jv⟩ == ⟨Jᵀc, v⟩`.
/// Reverse mode *is* fully supported, and ``lensVector(forToken:at:)`` uses it.
@_spi(Interpret)
public struct JacobianLens {

    public let model: any ResidualStreamReentry
    public let tokenizer: any Tokenizer
    public let corpus: CalibrationCorpus

    /// Step size for the central difference, as a fraction of the activation's
    /// RMS magnitude.
    ///
    /// Scaling to the activation rather than using an absolute step is what keeps
    /// the estimate stable across layers, since residual-stream norms grow with
    /// depth. `1e-3` sits comfortably between truncation error, which grows as
    /// the step grows, and floating-point cancellation, which grows as it shrinks.
    public var epsilon: Float

    /// Compute in float32 regardless of the model's dtype.
    ///
    /// A central difference subtracts two nearly equal numbers, so it loses
    /// precision exactly where the dtype has least to spare. bfloat16 has roughly
    /// three decimal digits of mantissa and cannot support this at all.
    public var computeInFloat32: Bool

    public init(
        model: any ResidualStreamReentry,
        tokenizer: any Tokenizer,
        corpus: CalibrationCorpus? = nil,
        epsilon: Float = 1e-3,
        computeInFloat32: Bool = true
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.corpus = corpus ?? CalibrationCorpus.standard(tokenizer: tokenizer)
        self.epsilon = epsilon
        self.computeInFloat32 = computeInFloat32
    }

    // MARK: - Jacobian-vector product

    /// `J̄ · direction`: the averaged first-order effect of perturbing the
    /// residual stream at `layer` along `direction`.
    ///
    /// The perturbation is applied at a single source position and its effect is
    /// averaged over every target position that can see it — positions before the
    /// source are excluded rather than averaged in as structural zeros, which
    /// would otherwise dilute the result by an amount that varies with prompt
    /// length.
    ///
    /// - Parameters:
    ///   - direction: shape `[hiddenSize]`. Need not be normalized.
    ///   - layer: layer boundary, `0...layerCount`.
    /// - Returns: shape `[hiddenSize]`, in the pre-final-norm residual basis.
    public func jacobianVectorProduct(of direction: MLXArray, at layer: Int) -> MLXArray {
        precondition(
            direction.ndim == 1 && direction.dim(0) == model.hiddenSize,
            "expected a [\(model.hiddenSize)] direction, got \(direction.shape)")
        precondition(!corpus.prompts.isEmpty, "calibration corpus is empty")

        let dtype: DType = computeInFloat32 ? .float32 : direction.dtype

        // Split magnitude from direction so the step size is independent of how
        // the caller scaled the input, then restore linearity at the end.
        let magnitude = sqrt((direction * direction).sum()).asType(.float32)
        let magnitudeValue = magnitude.item(Float.self)
        guard magnitudeValue > 0 else {
            return MLXArray.zeros([model.hiddenSize], dtype: dtype)
        }
        let unit = (direction.asType(dtype) / magnitude.asType(dtype))

        let tail = model.tail(from: layer)
        var accumulator = MLXArray.zeros([model.hiddenSize], dtype: dtype)

        for prompt in corpus.prompts {
            let tokens = MLXArray(prompt.map { Int32($0) })[.newAxis, .ellipsis]
            let hidden = model.hiddenState(of: tokens, at: layer).asType(dtype)
            let positions = hidden.dim(1)
            let source = positions - 1

            // Step scaled to this activation's own magnitude.
            let rms = sqrt((hidden * hidden).mean()).item(Float.self)
            let step = epsilon * max(rms, 1e-6)

            let delta = MLXArray.zeros(like: hidden)
            delta[0, source, 0...] = unit * step

            let derivative = (tail(hidden + delta) - tail(hidden - delta)) / (2 * step)

            // Only positions at or after the source can be causally affected.
            let affected = derivative[0, source..., 0...].mean(axis: 0)
            accumulator = accumulator + affected
            eval(accumulator)
        }

        return accumulator / Float(corpus.prompts.count) * magnitudeValue
    }

    // MARK: - Readout

    /// Read out an activation through the Jacobian lens.
    ///
    /// - Parameters:
    ///   - hidden: shape `[hiddenSize]`, taken from layer boundary `layer`.
    public func readout(of hidden: MLXArray, at layer: Int, topK: Int = 10) -> Readout {
        let mapped = jacobianVectorProduct(of: hidden, at: layer)
        return readoutOfMappedVector(mapped, at: layer, topK: topK)
    }

    /// Turn an already-computed `J̄h` into a ``Readout``.
    ///
    /// The final norm is applied *after* the Jacobian, matching the paper's
    /// `softmax(W_U · norm(J h))`. This is why ``ResidualStreamReentry/tail(from:)``
    /// deliberately excludes the norm.
    public func readoutOfMappedVector(
        _ mapped: MLXArray, at layer: Int, topK: Int = 10
    ) -> Readout {
        let scores = model.decode(mapped[.newAxis, .newAxis, 0...]).reshaped([-1])
        return ReadoutBuilder.build(
            scores: scores,
            topK: topK,
            layer: layer,
            depthPercent: model.depthPercent(ofLayer: layer),
            calibrationPromptCount: corpus.prompts.count,
            tokenizer: tokenizer
        )
    }

    /// Read out one sequence position at every layer boundary.
    public func sweep(tokens: MLXArray, position: Int = -1, topK: Int = 10) -> [Readout] {
        let batched = tokens.ndim == 1 ? tokens[.newAxis, 0...] : tokens
        let stream = model.hiddenStates(of: batched)
        return stream.enumerated().map { layer, hidden in
            readout(
                of: Activations.vector(from: hidden, position: position).asType(.float32),
                at: layer, topK: topK)
        }
    }

    // MARK: - Lens vectors

    /// The direction at `layer` that most increases `token`'s output logit:
    /// `∂logit_token / ∂h_layer`, averaged over the calibration corpus.
    ///
    /// These are the dictionary atoms a ``JSpace`` decomposition is built from —
    /// the paper's "J-lens vectors". Computed in reverse mode, which MLX supports
    /// fully through the whole stack, and which is the natural direction here:
    /// one VJP yields the gradient of a single scalar logit with respect to every
    /// hidden dimension at once.
    ///
    /// Note this differentiates `decode(tail(·))`, so the final norm and any
    /// logit softcap are included rather than approximated — there is no need to
    /// reach for the unembedding matrix directly.
    ///
    /// - Returns: shape `[hiddenSize]`, L2-normalized.
    public func lensVector(forToken token: Int, at layer: Int) -> MLXArray {
        precondition(!corpus.prompts.isEmpty, "calibration corpus is empty")

        let dtype: DType = computeInFloat32 ? .float32 : .float32
        let tail = model.tail(from: layer)
        var accumulator = MLXArray.zeros([model.hiddenSize], dtype: dtype)

        for prompt in corpus.prompts {
            let tokens = MLXArray(prompt.map { Int32($0) })[.newAxis, .ellipsis]
            let hidden = model.hiddenState(of: tokens, at: layer).asType(dtype)
            let positions = hidden.dim(1)
            let source = positions - 1

            let (_, gradients) = vjp(
                { inputs in [model.decode(tail(inputs[0]))] },
                primals: [hidden],
                cotangents: [oneHotCotangent(token: token, positions: positions, source: source)]
            )

            accumulator = accumulator + gradients[0][0, source, 0...]
            eval(accumulator)
        }

        let averaged = accumulator / Float(corpus.prompts.count)
        let norm = sqrt((averaged * averaged).sum())
        return averaged / maximum(norm, MLXArray(Float(1e-12)))
    }

    /// ``lensVector(forToken:at:)`` for several tokens.
    ///
    /// Each token costs its own VJP per calibration prompt — reverse mode returns
    /// the gradient of one scalar at a time — so this is linear in the number of
    /// tokens. Keep candidate sets small.
    public func lensVectors(forTokens tokens: [Int], at layer: Int) -> [MLXArray] {
        tokens.map { lensVector(forToken: $0, at: layer) }
    }

    /// Cotangent selecting a single logit at the last position.
    ///
    /// The cotangent must match the *output* of the differentiated function, so it
    /// is shaped `[1, positions, vocabularySize]` and is zero everywhere except
    /// the one logit whose gradient is wanted.
    private func oneHotCotangent(token: Int, positions: Int, source: Int) -> MLXArray {
        let vocabularySize = model.vocabularySize
        let cotangent = MLXArray.zeros([1, positions, vocabularySize], dtype: .float32)
        cotangent[0, source, token] = MLXArray(Float(1))
        return cotangent
    }
}
