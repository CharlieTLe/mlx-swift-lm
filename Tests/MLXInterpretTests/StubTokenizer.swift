// Copyright © 2026 Apple Inc.

import Foundation
@_spi(Interpret) import MLXInterpret
@_spi(Interpret) import MLXLMCommon

/// Deterministic in-memory tokenizer for offline tests.
///
/// Word-level, so `encode` / `decode` round-trip exactly and a readout's token
/// strings are predictable. Vocabulary ids are assigned in a fixed order, which
/// matters because tests assert on specific ids.
struct StubTokenizer: Tokenizer {

    /// Ordered vocabulary. Index is the token id.
    let vocabulary: [String]
    private let ids: [String: Int]

    /// Reserves ids 0–2 for special tokens so a real-looking id space is exercised
    /// rather than everything living at low indices.
    static let specialTokens = ["<pad>", "<bos>", "<eos>"]

    init(words: [String], vocabularySize: Int) {
        precondition(
            vocabularySize > Self.specialTokens.count,
            "vocabularySize \(vocabularySize) leaves no room for real tokens")

        // Truncate rather than trap: the calibration corpus contributes a couple of
        // hundred unique words, and tests legitimately want small vocabularies.
        // Words that do not fit encode as `<pad>`, which is fine for a stub.
        let room = vocabularySize - Self.specialTokens.count
        var vocabulary = Self.specialTokens + words.prefix(room)

        // Pad out to the model's vocabulary size so every logit index has a token.
        while vocabulary.count < vocabularySize {
            vocabulary.append("<extra_\(vocabulary.count)>")
        }
        self.vocabulary = vocabulary
        self.ids = Dictionary(
            vocabulary.enumerated().map { ($0.element, $0.offset) },
            uniquingKeysWith: { first, _ in first })
    }

    /// Builds a vocabulary from the words appearing in the calibration corpus plus
    /// any extra words a test needs, so `encode` never falls back to unknown.
    init(vocabularySize: Int, extraWords: [String] = []) {
        let corpusWords = CalibrationCorpus.defaultPrompts
            .flatMap { Self.split($0) }
        var seen = Set<String>()
        var ordered: [String] = []
        for word in corpusWords + extraWords where !seen.contains(word) {
            seen.insert(word)
            ordered.append(word)
        }
        self.init(words: ordered, vocabularySize: vocabularySize)
    }

    static func split(_ text: String) -> [String] {
        text.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
            .map(String.init)
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var result = addSpecialTokens ? [ids["<bos>"]!] : []
        for word in Self.split(text) {
            result.append(ids[word] ?? ids["<pad>"]!)
        }
        return result
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds
            .compactMap { id -> String? in
                guard let token = convertIdToToken(id) else { return nil }
                if skipSpecialTokens, Self.specialTokens.contains(token) { return nil }
                return token
            }
            .joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? { ids[token] }

    func convertIdToToken(_ id: Int) -> String? {
        guard vocabulary.indices.contains(id) else { return nil }
        return vocabulary[id]
    }

    var bosToken: String? { "<bos>" }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { "<pad>" }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
}
