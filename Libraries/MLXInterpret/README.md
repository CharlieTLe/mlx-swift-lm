# MLXInterpret

Mechanistic-interpretability tooling for `mlx-swift-lm`: read what a language model
is representing partway through a forward pass, and change it.

This is an implementation of the core method from
[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
(Gurnee, Sofroniew, Lindsey et al., Transformer Circuits, July 2026).

The paper's claim is that a language model maintains a small privileged set of
internal representations it can report on, deliberately manipulate, and reason
over. Its central tool is the **Jacobian lens**: describe a residual-stream
activation by its first-order causal effect on the output rather than by what it
would say in isolation. Its central intervention is the **coordinate swap**:
exchange two concepts' lens coordinates while leaving everything orthogonal
untouched, and watch the model's stated answer change.

## Two ways in, for two different jobs

**Reading goes through the normal call path.** Ask for hidden states on the way in,
read them off the way out:

```swift
let output = model(LMInput.Text(tokens: tokens), cache: cache,
                   state: .collectingHiddenStates())
let states = output.state?[hiddenStatesKey]   // layerCount + 1 entries
```

This observes what the model *actually did* — real cache, real mask mode, real
kernel selection — including mid-generation. It costs no arithmetic, since every
entry is an activation the pass computes anyway; what it costs is retention, so see
`collectHiddenStatesKey` before turning it on across a long prefill.

**Re-entry is a separate, narrower protocol.** `ResidualStreamReentry` exists for
the two things an output channel cannot express: differentiating the tail of the
stack, and patching mid-generation. A Jacobian-vector product needs a re-runnable
function `h_ℓ → h_L`, not values after the fact; synthesizing one by re-running the
whole model would recompute layers `0..<ℓ` on every evaluation, which at layer 28
of 36 is 36 layers of work where `tail(from:)` does 8.

The two must agree, and `ResidualStreamReentryTests` checks it at every layer
boundary: take the reported state entering block ℓ, re-enter there, and the logits
must match the model's own.

## Getting started

Everything is behind one SPI group, so the whole surface is opt-in and none of it
is part of the package's stable API:

```swift
@_spi(Interpret) import MLXInterpret
@_spi(Interpret) import MLXLLM
@_spi(Interpret) import MLXLMCommon
```

### What is the model thinking about here?

```swift
guard let model = context.model as? any ResidualStreamReentry else { return }

let lens = JacobianLens(model: model, tokenizer: context.tokenizer)
// 78%, not the middle of the paper's 38-92% band — see "What it actually produces".
let layer = model.layer(atDepthPercent: 78)

let tokens = MLXArray(context.tokenizer.encode(text: "The capital of France is"))[.newAxis, .ellipsis]
let hidden = Activations.vector(from: model.residualStream(of: tokens, at: layer))

print(lens.readout(of: hidden, at: layer, topK: 8).description)
// L28 (78%) [n=4]:  France (0.728),  Paris (0.072),  located (0.037),  named (0.020), …
```

### What if it were thinking about something else?

```swift
let franceVector = lens.lensVector(forToken: franceToken, at: layer)
let chinaVector  = lens.lensVector(forToken: chinaToken,  at: layer)

let patched = PatchedModel(
    wrapping: model, at: layer, swapping: franceVector, for: chinaVector)

// `patched` is an ordinary LanguageModel: it works with TokenIterator,
// generate(...), and ChatSession with no special handling.
```

## What it actually produces

Measured on `mlx-community/Qwen3-4B-Instruct-2507-4bit` (36 layers, hidden 2560,
4-bit) with an 8-prompt calibration corpus.

These numbers come from a real-weight experiment suite that is **proposed
separately** — it is not in this change, so there is no file here that reproduces
them. They are recorded because they are the evidence that this library does what
it claims; the offline tests below verify correctness, not behavior on a real model.

**The tap is exact.** Split-path deviation from the model's own forward pass:
`0.0`.

**Verbal report.** For `"Q: Name a sport played with a round ball on grass. A:"`,
the Jacobian lens at increasing depth:

```
L22 (62%):  radix (0.05),  legacy (0.02),  ambush (0.01), …          ← noise
L27 (75%):  soccer (0.41),  cricket (0.26),  golf (0.16),  football (0.05), …
L32 (88%):  soccer (0.77),  cricket (0.15),  Soccer (0.04),  football (0.02), …
```

Note the transition: below roughly 70% depth this model's readout is not
informative, and the paper's 38-92% "workspace band" is too generous here. Pick a
layer by looking, not by assumption.

The clearest single view of the paper's logit-lens claim is the same activation
through both lenses at layer 28, from the playground app:

```
J-lens:   France (0.728),  Paris (0.072),  located (0.037),  named (0.020), …
logit :   ____ (0.768),  ____ (0.068),  __ (0.054),  Paris (0.019), …
```

**Coordinate swap.** Exchanging the `France` coordinate for `China` at layer 28:

```
"The capital of France is"                  →  Paris   | swapped →  Beijing
"The main language spoken in France is"     →  French  | swapped →  French
"France is located on the continent of"     →  Europe  | swapped →  Europe
"The currency used in France is the"        →  euro    | swapped →  euro
```

One of four templates propagates. The paper reports 76/192 across 16 templates, so
partial propagation is the expected shape of the result, but a single template is
not a rate.

**Lens vectors are concept-specific.** Every concept scores far higher under its own
lens vector than another's (12/12 pairwise comparisons), and the near-misses are
semantically sensible — `water` and `music` score closer to each other than either
does to `Paris` or `China`.

**Ablation is specific, not just disruptive.** Ablating the top ten lens directions
moves logits by 5.9; ablating a direction orthogonal to the activation moves them by
0.25. A ~23× ratio is what makes "the intervention did something" mean something.

**J-space.** Decomposing the France activation yields
`Paris×7.1 + France×6.9 + located×5.7 + named×3.5 + famous×3.0 + known×2.1 + Italy×0.9 + …`,
explaining 2.8% of squared norm in 12 terms. The paper reports ~6-7% — same order of
magnitude with a 16-token dictionary instead of a vocabulary.

**The J-space asymmetry reproduces.** This is the paper's central claim: a concept
vector's J-space component carries the causal effect while its orthogonal remainder
does not. Running their protocol (swap success rate over 12 trials, concept vectors
built by difference-of-means so the test isn't circular), at layer 28 with
intervention strength 2.0:

```
mean J-space share of the concept vector: 5.1%      (paper: 6-7%)

condition                 successes   rate    paper
lens vectors               7/12       58%      88%
full concept vector        2/12       17%       —
J-space component          6/12       50%      59%
remainder (clamped)        0/12        0%       5%
remainder (unclamped)      1/12        8%      28%
```

50% versus 0% is the asymmetry, and the clamped-versus-unclamped gap runs the same
direction the paper reports: an unclamped remainder swap drags the J-space
coordinates along with it, so clamping is what makes the control meaningful.

Two things this took, both of which are easy to get wrong:

- **Split the concept vector, not the activation.** An earlier attempt split the
  activation and reported the asymmetry running *backwards*. That was measuring a
  different object.
- **Calibrate the intervention first.** At strength 1.0 even pure lens vectors only
  flipped 1 of 12 answers, so every condition sat at zero and the null said nothing.
  A ceiling check is a prerequisite, not a nicety. Curiously, patching a *band* of
  layers was consistently worse than a single boundary, which I did not expect.

**Where the numbers fall short of the paper.** Pure lens vectors reach 58% against
88%, and the model answers a question about the swapped-in entity rather than always
naming it. Both are consistent with a 4B 4-bit model and an 8-prompt calibration
corpus rather than a frontier model and ~1000 prompts.

## The pieces

| Type | Purpose |
|---|---|
| `hiddenStatesKey` / `collectHiddenStatesKey` | Reading, through the normal call path. Lives in `MLXLMCommon`. |
| `ResidualStreamReentry` | Re-entry, for derivatives and patching. Lives in `MLXLMCommon`; architectures opt in by conforming. |
| `LogitLens` | Push an activation straight through the unembedding. Cheap baseline and correctness oracle. |
| `JacobianLens` | The paper's lens. `readout(of:at:)` and `lensVector(forToken:at:)`. |
| `JSpace` | Sparse nonnegative decomposition onto a dictionary of lens vectors. |
| `Interventions` | `steer`, `ablate`, `coordinateSwap`. |
| `PatchedModel` | A `LanguageModel` that applies an edit mid-stack on every forward pass. |
| `Readout` / `TokenDisplay` | Ranked results and legible token rendering. |

## Adding an architecture

`MLXInterpret` depends only on the protocol, never on `MLXLLM`, so a new
architecture opts in without touching this target. Conform it in `MLXLLM`
alongside the model — see `Libraries/MLXLLM/Models/Llama+ResidualStream.swift`
for the simplest case, which needs no changes to `Llama.swift` at all.

Four methods must compose to reproduce the model's own forward pass:

```
unembed(finalNorm(runLayers(embed(tokens), range: 0..<layerCount, cache: nil))) == model(tokens)
```

and the model must honor `collectHiddenStatesKey`, reporting states that agree with
re-entry. `Tests/MLXInterpretTests/ResidualStreamReentryTests.swift` asserts both.
Two things are easy to get wrong and are silent when you do:

- **Post-embedding scaling.** Gemma multiplies by `sqrt(hiddenSize)` computed in
  bfloat16. Omit it and every readout is wrong by a dtype-dependent factor.
- **Per-layer masking.** Gemma 3 alternates windowed and global attention on a
  `slidingWindowPattern` cycle. Approximating it with a uniform causal mask breaks
  split-path equivalence.

Currently conformed: Llama/Mistral, Qwen3, Gemma 3 text.

## Two implementation notes worth knowing

### The Jacobian is never materialized

`J_ℓ` is `hiddenSize × hiddenSize`. It is also never needed: a readout only wants
the product `J̄h`, and since every calibration prompt shares the same direction
`h`, that costs a fixed two forward passes *per prompt* regardless of model width.

### Forward-mode autodiff is not used, and cannot be

`J̄h` is exactly a Jacobian-vector product, which is what `MLX.jvp` computes. It
does not work through a transformer in mlx-swift 0.31.6:

- **Grouped-query attention** fails with `[addmm] Got 0 dimension input`. Every
  modern LLM here uses fewer KV heads than query heads.
- **The symbolic `.causal` mask** trips a raw C++ `assert(tangents.size() == 3)`
  in `Select::jvp` (`primitives.cpp:3077`), which calls `abort()` directly. The
  root cause is `where`, not attention: `Select::jvp` assumes all three of its
  inputs carry tangents, and a boolean condition never can, so any `where` under
  `jvp` aborts. No Swift error handler
  can intercept this, so even a "try it and fall back" strategy is unsafe: the
  attempt itself terminates the process.

So the directional derivative is computed by **central differences** — the same
two forward passes, second-order accurate, and working through every op.
`ResidualStreamReentryTests` validates it against reverse mode using the adjoint
identity `⟨c, Jv⟩ == ⟨Jᵀc, v⟩`.

Reverse mode (`vjp`) *is* fully supported and is what builds lens vectors, where
it is also the natural choice: one backward pass gives the gradient of a single
logit with respect to every hidden dimension at once.

`Tests/MLXInterpretTests/AutodiffCapabilityTests.swift` pins these boundaries down
and will start failing if MLX gains the missing rules — at which point forward mode
is worth revisiting.

## Interpreting results honestly

- **Report the calibration count.** `Readout.calibrationPromptCount` is not
  metadata. The paper averages over ~1000 pretraining-like prompts, and that
  averaging is what distinguishes a concept the model is *generally* poised to
  speak about from one that happens to be verbalizable in a single context. The
  built-in corpus defaults to 12 prompts for interactive use.
- **The J-space dictionary is a candidate set, not the vocabulary.** A lens vector
  costs one backward pass per calibration prompt, so a full-vocabulary dictionary
  is out of reach. A decomposition can only find concepts its candidate set
  contains.
- **Single tokens only.** A lens readout ranks vocabulary entries, so multi-token
  concepts have no direct representation. The paper notes the same limitation.
- **Quantization blurs everything.** Derivatives are computed in float32
  regardless of model dtype, but 4-bit weights still cost precision.

## Cost

A readout is `2 × calibrationPrompts` forward passes over the tail of the stack.
At the default 12 prompts that is seconds for one layer; a full-depth sweep is
minutes. Keep calibration prompts short — a traced attention pass grows with the
square of sequence length.

## Tests

Offline, no network and no checkpoints. Note `swift test` does not work in this
repo — see CONTRIBUTING.md.

```bash
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
  -skipPackagePluginValidation -only-testing:MLXInterpretTests
```

`AutodiffCapabilityTests` is worth knowing about before reading it: one of its
tests asserts that MLX *fails*, because the central-difference default exists only
because of that failure. If MLX gains the missing forward-mode rule the test starts
failing, which is the signal to revisit the default.
