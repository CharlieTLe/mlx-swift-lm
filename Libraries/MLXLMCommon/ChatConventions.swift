// Copyright © 2025 Apple Inc.

import Foundation

// MARK: - ChatConventionsProviding

/// A model's chat conventions: how it encodes tool calls and how it reasons.
///
/// Adopted by the model so this knowledge lives with the model definition rather
/// than in centralized `model_type` string tables (``ToolCallFormat/infer(from:configData:)``
/// and ``ReasoningConfig/infer(from:modelId:configData:)``). The model factories
/// probe this protocol before falling back to those inference chains.
///
/// Opt-in: the protocol extension defaults both properties to `nil`, so a
/// conforming type overrides only the property that applies to it. This mirrors
/// the ``ModelConfigurationValidating`` opt-in pattern.
public protocol ChatConventionsProviding {
    /// The tool-call format this model emits, or `nil` for the JSON default.
    var toolCallFormat: ToolCallFormat? { get }

    /// The model's reasoning protocol, or `nil` for non-reasoning models.
    var reasoningConfig: ReasoningConfig? { get }
}

extension ChatConventionsProviding {
    public var toolCallFormat: ToolCallFormat? { nil }
    public var reasoningConfig: ReasoningConfig? { nil }
}

// MARK: - LanguageModel probe

extension LanguageModel {
    /// The tool-call format this model declares via ``ChatConventionsProviding``,
    /// or `nil` if it does not conform / declares no format.
    ///
    /// Exposed as a method on the model so factories read it by borrowing `self`
    /// and receiving a `Sendable` result, rather than downcasting the model
    /// inline (which would taint the model's region and block returning it as a
    /// `sending` result).
    public var declaredToolCallFormat: ToolCallFormat? {
        (self as? ChatConventionsProviding)?.toolCallFormat
    }

    /// The reasoning config this model declares via ``ChatConventionsProviding``,
    /// or `nil` if it does not conform / declares none.
    public var declaredReasoningConfig: ReasoningConfig? {
        (self as? ChatConventionsProviding)?.reasoningConfig
    }
}
