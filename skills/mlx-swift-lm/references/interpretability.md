# Interpretability (MLXInterpret)

Read what a model is representing partway through a forward pass, and change it.
Implements the Jacobian lens and coordinate-swap intervention from
[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
(Transformer Circuits, July 2026).

Full detail: `Libraries/MLXInterpret/README.md`.

## Imports

The entire surface is behind one SPI group:

```swift
@_spi(Interpret) import MLXInterpret
@_spi(Interpret) import MLXLLM
@_spi(Interpret) import MLXLMCommon
```

## Reading the residual stream

Reading goes through the normal call path, so it sees the real cached decode:

```swift
let output = model(LMInput.Text(tokens: tokens), cache: cache,
                   state: .collectingHiddenStates())
let states = output.state?[hiddenStatesKey]   // layerCount + 1 entries
```

No arithmetic cost, but collection retains every layer's activation until eval —
see `collectHiddenStatesKey` before enabling it across a long prefill.

`ResidualStreamReentry` is the separate, narrower protocol for *re-entering* the
stack, which is needed only for differentiating the tail and for patching.

## Through a lens

```swift
await container.perform { context in
    guard let model = context.model as? any ResidualStreamReentry else { return }

    let lens = JacobianLens(model: model, tokenizer: context.tokenizer)
    // Depth on the paper's 0-100 scale. Pick it by looking: on Qwen3-4B the
    // readout only becomes informative past roughly 70%.
    let layer = model.layer(atDepthPercent: 78)

    let tokens = MLXArray(
        context.tokenizer.encode(text: "The capital of France is", addSpecialTokens: false)
            .map { Int32($0) })[.newAxis, .ellipsis]

    let hidden = Activations.vector(from: model.residualStream(of: tokens, at: layer))
        .asType(.float32)

    print(lens.readout(of: hidden, at: layer, topK: 10).description)
}
```

`LogitLens` has the same shape and is far cheaper. Use it as a baseline: at the
last layer boundary the two must agree exactly, since the Jacobian is the identity
there.

## Intervening

`PatchedModel` conforms to `LanguageModel`, so it works with `TokenIterator`,
`generate(...)`, and `ChatSession` unchanged.

```swift
let from = lens.lensVector(forToken: franceToken, at: layer)
let to   = lens.lensVector(forToken: chinaToken,  at: layer)

// Coordinate swap: exchange the coordinate, preserve the orthogonal remainder.
let swapped = PatchedModel(wrapping: model, at: layer, swapping: from, for: to)

// Ablation: project directions out.
let ablated = PatchedModel(wrapping: model, at: layer, ablating: [from, to])

// Anything else.
let custom = PatchedModel(wrapping: model, at: layer) { hidden in
    Interventions.steer(hidden, along: from, alpha: 2)
}
```

The patch runs on **every** forward pass, so during decoding it applies at every
token. That models "hold this concept in mind throughout," not "nudge the first
token."

## Sparse decomposition

```swift
let space = JSpace.build(lens: lens, at: layer, around: hidden, candidateCount: 32)
let decomposition = space.decompose(hidden, maxTerms: 25)

print(decomposition.description)
// [6.8% explained]  Paris×0.412 +  France×0.230 + …

// The split the paper's swap result rests on.
let (jPart, remainder) = space.project(hidden)
```

## Adding an architecture

Conform it in `MLXLLM` next to the model. `Libraries/MLXLLM/Models/Llama+ResidualStream.swift`
is the template and required no changes to `Llama.swift` — the conformance lives in
the same module, so it reads internal members directly.

The four methods must compose to reproduce the model's own forward pass:

```
unembed(finalNorm(runLayers(embed(tokens), range: 0..<layerCount, cache: nil))) == model(tokens)
```

Two silent failure modes: architecture-specific post-embedding scaling (Gemma
multiplies by `sqrt(hiddenSize)` in bfloat16) and per-layer mask selection (Gemma 3
alternates windowed and global attention). Add a split-path equivalence test.

Conformed today: Llama/Mistral, Qwen3, Gemma 3 text.

## Gotchas

| Issue | Detail |
|---|---|
| Forward-mode autodiff is unusable | `jvp` fails on grouped-query attention, and the symbolic `.causal` mask trips a C++ `assert` that aborts the process — uncatchable. The lens uses central differences instead. `vjp` is fine and builds the lens vectors. |
| Calibration count matters | `Readout.calibrationPromptCount` is part of the result, not metadata. The paper averages ~1000 prompts; the default here is 12. |
| Single-token concepts only | A readout ranks vocabulary entries. Multi-token concepts have no representation. |
| Dictionary is a candidate set | A full-vocabulary J-space is not affordable, so a decomposition only finds concepts in its candidate set. |
| Cost | A readout is `2 × calibrationPrompts` forward passes over the tail. Keep calibration prompts short. |

## Tests

```bash
# Offline, no weights. `swift test` does not work here — see CONTRIBUTING.md.
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
  -skipPackagePluginValidation -only-testing:MLXInterpretTests
```

`Tests/MLXInterpretTests/AutodiffCapabilityTests.swift` documents which MLX
autodiff modes work through a transformer, and is the canary for revisiting the
central-difference default.
