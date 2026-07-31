// Copyright © 2026 Apple Inc.
//
// Chat conventions: models declare their own tool-call format and reasoning
// protocol (`ChatConventionsProviding`), and `ChatConventionsRegistry` supplies
// the conventions that are keyed on repo id rather than on the model itself.
// Tiny random-weight models so these run in CI without downloads.

import Foundation
import MLX
import MLXLMCommon
import Testing
import XCTest

@testable import MLXLLM

// MARK: - Model-declared conventions
//
// XCTest (like `NanbeigeTests`) because these construct MLX modules, which the
// Swift Testing harness cannot currently initialize in this package's test host.

final class ChatConventionsModelTests: XCTestCase {

    // MARK: Llama: model_type-gated secondary signals

    /// `LlamaModel` backs both the `llama` and `mistral` registry entries, so the
    /// Llama 3 signals must be gated on `model_type`. Mistral-Nemo shares Llama 3's
    /// large vocabulary (131072), and must not be handed `.llama3`.
    private func llama(
        modelType: String?, vocabularySize: Int,
        ropeScaling: [String: StringOrNumber]? = nil
    ) -> LlamaModel {
        LlamaModel(
            LlamaConfiguration(
                hiddenSize: 8, hiddenLayers: 1, intermediateSize: 16, attentionHeads: 2,
                rmsNormEps: 1e-5, vocabularySize: vocabularySize, kvHeads: 1,
                ropeScaling: ropeScaling, modelType: modelType))
    }

    func testLlama3RecognizedByVocabularySize() {
        XCTAssertEqual(llama(modelType: "llama", vocabularySize: 128_256).toolCallFormat, .llama3)
    }

    func testLlama3RecognizedByRopeScaling() {
        let model = llama(
            modelType: "llama", vocabularySize: 32_000,
            ropeScaling: ["rope_type": .string("llama3"), "factor": .float(8)])
        XCTAssertEqual(model.toolCallFormat, .llama3)
    }

    func testLlama2HasNoToolCallFormat() {
        XCTAssertNil(llama(modelType: "llama", vocabularySize: 32_000).toolCallFormat)
    }

    /// The regression this gate exists for: a Mistral checkpoint with a Llama
    /// 3-sized vocabulary must stay `nil`, not become `.llama3`.
    func testMistralWithLargeVocabularyIsNotLlama3() {
        XCTAssertNil(llama(modelType: "mistral", vocabularySize: 131_072).toolCallFormat)
        XCTAssertNil(llama(modelType: nil, vocabularySize: 131_072).toolCallFormat)
    }

    // MARK: Representative model declarations

    /// Representative coverage only: this asserts the `model_type` -> class ->
    /// declaration path works end to end through the real type registry. It does
    /// not enumerate all ~25 migrated classes.
    func testQwen3DeclaresReasoningAndNoToolFormat() async throws {
        let json = """
            {
                "model_type": "qwen3",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "rms_norm_eps": 1e-5,
                "vocab_size": 64
            }
            """
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(json.utf8), modelType: "qwen3")
        XCTAssertEqual(model.reasoningConfig, .thinkTagsWithEnableThinking)
        // Qwen3 (unlike Qwen3.5) declares no tool-call format.
        XCTAssertNil(model.toolCallFormat)
    }
}

// MARK: - Presets and the resolver registry
//
// Pure value logic, no MLX modules, so these stay in Swift Testing.

@Suite
struct ChatConventionsTests {

    // MARK: - Presets

    @Test func thinkTagsPreset() {
        let config = ReasoningConfig.thinkTagsWithEnableThinking
        #expect(config.startDelimiter == "<think>")
        #expect(config.endDelimiter == "</think>")
        #expect(config.promptStrategy == .templateFlag(key: "enable_thinking", defaultOn: true))
        #expect(config.isSpecialToken)
    }

    @Test func alwaysOnPreset() {
        #expect(ReasoningConfig.alwaysOnThinking.promptStrategy == .alwaysOn)
        #expect(ReasoningConfig.alwaysOnThinking.startDelimiter == "<think>")
    }

    // MARK: - Declaration inheritance

    /// `Qwen35MoEModel` and the VLM `Qwen35MoE` subclass their base model rather
    /// than declaring conventions themselves, so they rely on a subclass inheriting
    /// an extension-declared protocol witness. Pin that Swift semantic here: the
    /// subclass must resolve the base's declaration, not the `nil` protocol default.
    @Test func subclassInheritsExtensionDeclaredConventions() {
        let base: any ChatConventionsProviding = ConventionsBase()
        let derived: any ChatConventionsProviding = ConventionsDerived()
        #expect(base.toolCallFormat == .xmlFunction)
        #expect(derived.toolCallFormat == .xmlFunction)
        #expect(derived.reasoningConfig == .thinkTagsWithEnableThinking)
    }

    // MARK: - ChatConventionsRegistry

    /// R1-Distill reports a base `model_type` (`qwen2`/`llama`), so only the repo
    /// id identifies it. This is the case the resolver seam exists for.
    @Test func deepSeekResolverRecognizesDistillsById() {
        let resolver = DeepSeekR1ConventionsResolver()
        let ids = [
            "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
            "mlx-community/DeepSeek-R1-4bit",
            // Re-upload not prefixed "deepseek-".
            "someone/R1-Distill-Llama-8B-mlx",
        ]
        for id in ids {
            #expect(
                resolver.reasoningConfig(modelId: id, modelType: "qwen2")?.promptStrategy
                    == .alwaysOn, "\(id) should resolve always-on reasoning")
        }
    }

    @Test func deepSeekResolverIgnoresUnrelatedIds() {
        let resolver = DeepSeekR1ConventionsResolver()
        #expect(
            resolver.reasoningConfig(
                modelId: "mlx-community/Qwen2.5-7B-Instruct-4bit", modelType: "qwen2") == nil)
        // Declares no tool-call format at all.
        #expect(
            resolver.toolCallFormat(
                modelId: "mlx-community/DeepSeek-R1-4bit", modelType: "qwen2") == nil)
    }

    @Test func sharedRegistryResolvesDeepSeekOutOfTheBox() {
        let resolved = ChatConventionsRegistry.shared.reasoningConfig(
            modelId: "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit", modelType: "qwen2")
        #expect(resolved == .alwaysOnThinking)
    }

    @Test func emptyRegistryResolvesNothing() {
        let registry = ChatConventionsRegistry()
        #expect(registry.toolCallFormat(modelId: "any/model", modelType: "qwen2") == nil)
        #expect(registry.reasoningConfig(modelId: "any/model", modelType: "qwen2") == nil)
    }

    /// Most recently registered wins, so a downstream package can override a
    /// built-in rule.
    @Test func mostRecentlyRegisteredResolverWins() {
        struct Fixed: ChatConventionsResolving {
            let format: ToolCallFormat
            func toolCallFormat(modelId: String, modelType: String) -> ToolCallFormat? { format }
        }
        let registry = ChatConventionsRegistry(resolvers: [Fixed(format: .json)])
        #expect(registry.toolCallFormat(modelId: "a/b", modelType: "t") == .json)

        registry.register(Fixed(format: .glm4))
        #expect(registry.toolCallFormat(modelId: "a/b", modelType: "t") == .glm4)
    }

    /// A resolver that defers (returns nil) falls through to an earlier one
    /// rather than shadowing it.
    @Test func deferringResolverFallsThrough() {
        struct Deferring: ChatConventionsResolving {}
        let registry = ChatConventionsRegistry()
        registry.register(DeepSeekR1ConventionsResolver())
        registry.register(Deferring())
        #expect(
            registry.reasoningConfig(
                modelId: "mlx-community/DeepSeek-R1-4bit", modelType: "deepseek_v3")
                == .alwaysOnThinking)
    }

    /// Resolvers may key on `modelType` as well as id.
    @Test func resolverCanKeyOnModelType() {
        struct ByType: ChatConventionsResolving {
            func toolCallFormat(modelId: String, modelType: String) -> ToolCallFormat? {
                modelType == "special" ? .minimaxM2 : nil
            }
        }
        let registry = ChatConventionsRegistry(resolvers: [ByType()])
        #expect(registry.toolCallFormat(modelId: "a/b", modelType: "special") == .minimaxM2)
        #expect(registry.toolCallFormat(modelId: "a/b", modelType: "other") == nil)
    }
}

// MARK: - Fixtures for the inheritance pin
//
// Mirrors how the model families declare conventions: the base class conforms,
// the declarations live in an extension on it, and the subclass adds nothing.

private class ConventionsBase: ChatConventionsProviding {}

extension ConventionsBase {
    var toolCallFormat: ToolCallFormat? { .xmlFunction }
    var reasoningConfig: ReasoningConfig? { .thinkTagsWithEnableThinking }
}

private class ConventionsDerived: ConventionsBase {}
