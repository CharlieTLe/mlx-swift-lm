# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

One SwiftPM package (`Package.swift` at the root) whose targets all live under `Libraries/`,
plus three things that are **not** part of it:

- `IntegrationTesting/IntegrationTesting.xcodeproj` — a separate Xcode project for tests that
  download real models and generate on Metal.
- `ShakespeareReader/`, `MuseGlimmerDemo/` — sibling SwiftPM packages with a *local path*
  dependency (`.package(name: "mlx-swift-lm", path: "..")`), so they build against the working
  copy rather than a published tag. Each has its own `Package.swift`.
- `skills/mlx-swift-lm/references/` — task-oriented reference docs (`kv-cache.md`,
  `model-porting.md`, `tool-calling.md`, `generation.md`, `concurrency.md`, and more). **Read
  the relevant one before working in that area**; they carry detail this file does not.

## Commands

### Build and test the libraries

`swift test` **does not work** here (an mlx-swift limitation). Use `xcodebuild`:

```bash
swift build                                  # building is fine via SwiftPM
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
  -skipPackagePluginValidation
```

Unit tests need no special hardware and download nothing. A single test — note the test
targets mix XCTest (`Target/Class/testMethod`) and swift-testing (`Target/Suite/test()`, with
the parens escaped):

```bash
xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
  -skipPackagePluginValidation -only-testing:MLXLMTests/EvalTests/testLlamaEval
```

Test targets: `MLXLMTests` (the bulk), `MLXFoundationModelsTests`,
`MLXGuidedGenerationTests`, `MLXHuggingFaceMacrosTests`, `CXGrammarTests`.

### Integration tests

Require macOS with Metal and download from Hugging Face on first run. They are **not** part of
PR checks, so nothing runs them on your branch — run them locally when you change model
loading, generation, or tokenizer behavior.

```bash
xcodebuild test -project IntegrationTesting/IntegrationTesting.xcodeproj \
  -scheme IntegrationTesting -destination 'platform=macOS' -skipPackagePluginValidation \
  -only-testing:IntegrationTestingTests/ToolCallIntegrationTests/qwen35FormatAutoDetection\(\)
```

### Format and docs

CI gates on both. `swift-format` is pinned to **603.0.0** — a newer one reformats files no PR
touched and turns every open PR red at once.

```bash
pre-commit run --all                         # what CI runs; formats the whole repo
swift-format format --in-place --configuration .swift-format --recursive .
scripts/verify-docs.sh                       # DocC must build warning-free for every library
```

`.swift-format` sets 4-space indent and `indentConditionalCompilationBlocks: false`, so `#if`
bodies sit at the enclosing indentation. (CONTRIBUTING.md's manual format command names `Tools`
and `Applications` directories that no longer exist; the command above is the accurate one.)

### ShakespeareReader

A SwiftUI app, run from its own directory. `--selftest` is the fast model-free, network-free
gate and is what to run before any manual pass:

```bash
cd ShakespeareReader
swift run -c release ShakespeareReader --selftest      # prints "selftest: all checks passed"
swift run -c release ShakespeareReader                 # the app
swift run -c release ShakespeareReader --metal-check    # one array on the GPU
swift run -c release ShakespeareReader --show-prompt    # assembled prompts + token counts
swift run -c release ShakespeareReader --benchmark      # latency table
```

Also `--diagnostics`, `--greedy`, `--model <hf-id>`, `--passage hamlet:3.1:62-96`. All flags
work identically in an installed copy and inside the `.app` bundle. The play corpus JSON under
`Sources/ShakespeareReader/Resources/Plays` is **checked in**; `tools/build_corpus.py` exists to
make the parse reproducible and is not a build step.

## Architecture

### Layering

`MLXLMCommon` is the core: `LanguageModel`, `KVCache`, `ModelContext`/`ModelContainer`,
`UserInput`/`LMInput`, `generate()`, `ChatSession`, tool calling, chat conventions.
`MLXLLM`, `MLXVLM` and `MLXEmbedders` sit on top and contribute model architectures.
Nothing in `MLXLMCommon` may depend upward on them — see the trampoline note below for how
that constraint is honored.

### Loading: factory, type registry, model registry

Three distinct registries, easy to confuse:

- **`LLMTypeRegistry.shared` / `VLMTypeRegistry.shared`** maps a config file's `model_type`
  string (`"llama"`, `"gemma4"`, …) to a closure that decodes the JSON into a
  `Configuration` and constructs the model. Adding an architecture means adding one line here.
- **`LLMRegistry.shared`** holds the predefined `ModelConfiguration` values
  (`LLMRegistry.gemma3_1B_qat_4bit`), which carry stop tokens, reasoning delimiters and
  default prompts.
- **`LLMModelFactory.shared`** ties them together and owns `_load`: resolve the directory,
  decode the base config, build the model with random weights, load and quantize weights,
  load the tokenizer, and (VLM) build the processor. The weight-preparation hook
  `model.sanitize(weights:metadata:)` is called from `MLXLMCommon/Load.swift`, not from the
  factory itself, which is where to look when a checkpoint's tensor names do not line up.

`ModelContainer` is an `actor` wrapping `ModelContext`; all inference goes through
`container.perform { context in ... }`. Treat that as the isolation boundary.

### The trampoline, and why linking matters

`ModelFactoryRegistry` (in `MLXLMCommon`) reaches factories in the layers above it via
`NSClassFromString("MLXVLM.TrampolineModelFactory")` rather than a compile-time dependency.
The consequence is real and silent: **merely linking `MLXLLM` / `MLXVLM` is what registers
their factories.** An app that omits `MLXVLM` and loads a VLM id gets a bare
`ModelConfiguration` back, losing its stop tokens and reasoning delimiters, and
`loadModelContainer` throws `.noModelFactoryAvailable` when no factory is registered at all.

### Package traits

`FoundationModelsIntegration` is default-on and gates `MLXFoundationModels` (the bridge to
Apple's `FoundationModels.LanguageModel`), whose surface needs the macOS/iOS/visionOS 27.0 SDK.
Disabling it compiles that target to an **empty library**, which is how consumers on older OS
versions still use `MLXLLM`/`MLXLMCommon`. The sibling app packages pass `traits: []` for
exactly that reason. An Xcode project cannot express traits, so an app built through
`.xcodeproj` compiles `MLXFoundationModels` whether it uses it or not.

### Vendored C++ (xgrammar)

`Libraries/MLXCXGrammar` compiles upstream xgrammar C++17 directly, plus a `shim.cc` exposing
an `extern "C"` API. Refresh with `scripts/sync-xgrammar-source.sh`; the pinned upstream sha
lives in `Libraries/MLXCXGrammar/xgrammar/VERSION` and is mirrored in `shim.cc`'s
`kXGrammarVersion`. Two deliberate build tricks: the `xgrammar` and `picojson` namespaces are
renamed via `-D` so this target cannot collide with another xgrammar in the same binary, and
warnings are suppressed with `-w` because the unmodified upstream is not warning-clean.
`MLXGuidedGeneration` is the Swift engine on top, standalone and with no FoundationModels
coupling.

## Platform constraints worth knowing before debugging

- **MLX has no Metal device on the iOS Simulator**, and the first touch of any `MLX.Memory`
  knob constructs `mlx::core::metal::Device`, which `abort()`s from C++ where no Swift `catch`
  can reach it. Code that might run there must probe first rather than try/catch — see
  `ShakespeareReader`'s `hasMLXDevice`.
- **`Font.system(size:)` does not follow Dynamic Type** while
  `Font.custom(_:size:relativeTo:)` does. Undocumented, measured, and the reason
  `ShakespeareReader/Sources/ShakespeareReader/Reader/ReaderFont.swift` carries two
  `systemSize` variants.
- The **`ShakespeareReader` iOS app target currently fails to build for the Simulator** for a
  reason unrelated to any change: Xcode compiles the `MLXHuggingFaceMacros` plugin target for
  `arm64-apple-ios-simulator` instead of for the host, so swift-syntax does not resolve.

## Conventions

- Comments in `ShakespeareReader` are unusually dense and explain *why*, often naming the
  alternative that was rejected. Match that register when editing there; match the
  surrounding density elsewhere.
- A deliberate prompt change in `ShakespeareReader` means bumping `Prompts.version` and
  regenerating the golden `PassageContext` string that `--selftest` compares against; the
  self test prints the replacement.
