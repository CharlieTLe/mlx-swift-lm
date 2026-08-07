// Copyright © 2026 Apple Inc.

import Foundation
@_spi(Interpret) import MLXLMCommon

/// Renders token ids as text a human can read in a lens readout.
///
/// `Tokenizer.decode(tokenIds:)` is the right call for reconstructing a *string*
/// from a *sequence*, but it is the wrong call for displaying individual tokens
/// from a ranked list: it drops the leading-space information that distinguishes
/// `" the"` from `"the"`, and it can rewrite whitespace when the pieces do not
/// form a valid sequence. So readouts go through `convertIdToToken` and normalize
/// the piece conventions here instead.
///
/// The conventions handled are the ones documented in
/// `Libraries/MLXGuidedGeneration/TokenizerVocabExtractor.swift`. That type is not
/// reused because it is shaped for xgrammar's byte-level consumption and this
/// target does not depend on `MLXGuidedGeneration`.
@_spi(Interpret)
public enum TokenDisplay {

    /// A displayable form of `id`.
    ///
    /// Leading whitespace is preserved as a real space so `" dog"` and `"dog"`
    /// stay visibly distinct.
    public static func string(for id: Int, tokenizer: any Tokenizer) -> String {
        guard let piece = tokenizer.convertIdToToken(id) else {
            return "<unk:\(id)>"
        }
        return normalize(piece)
    }

    /// Whitespace-trimmed, lowercased form, for matching a readout against a
    /// concept name.
    public static func normalized(for id: Int, tokenizer: any Tokenizer) -> String {
        string(for: id, tokenizer: tokenizer)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
    }

    /// Normalize a raw vocabulary piece into displayable text.
    public static func normalize(_ piece: String) -> String {
        // SentencePiece byte fallback: `<0x0A>` and friends carry a single raw
        // byte. Keep the escaped form — substituting the byte would inject
        // control characters into readout output.
        if piece.count == 6, piece.hasPrefix("<0x"), piece.hasSuffix(">") {
            return piece
        }

        // SentencePiece space marker (LOWER ONE EIGHTH BLOCK) is a plain
        // substitution and is not part of the byte-level scheme.
        let withSpaces = piece.replacingOccurrences(of: "\u{2581}", with: " ")

        // GPT-2-style byte-level BPE (Qwen, Llama, Mistral, …) stores each *byte*
        // of a token as a printable codepoint. Reversing that mapping to bytes and
        // decoding as UTF-8 is what makes multi-byte tokens legible: the piece
        // "ä¸ŃåĽ½" is the byte-level spelling of 中国, and rendering it verbatim
        // makes every non-ASCII readout look like mojibake.
        if let decoded = decodeByteLevel(withSpaces) {
            return escapeControlCharacters(decoded)
        }

        return escapeControlCharacters(withSpaces)
    }

    /// Reverse of GPT-2's `bytes_to_unicode`, then a UTF-8 decode.
    ///
    /// Returns `nil` when the piece contains a codepoint outside the mapping, or
    /// when the resulting bytes are not valid UTF-8. Both are expected: a
    /// SentencePiece vocabulary is not byte-level, and a byte-level token can hold
    /// a partial multi-byte sequence that only becomes valid alongside its
    /// neighbours. In either case the caller falls back to the raw piece.
    static func decodeByteLevel(_ piece: String) -> String? {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(piece.unicodeScalars.count)

        for scalar in piece.unicodeScalars {
            guard let byte = byteLevelToByte[scalar] else { return nil }
            bytes.append(byte)
        }

        return String(bytes: bytes, encoding: .utf8)
    }

    /// Make control characters visible rather than letting them corrupt output.
    static func escapeControlCharacters(_ text: String) -> String {
        var out = String()
        out.reserveCapacity(text.count)
        for character in text {
            switch character {
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if let scalar = character.unicodeScalars.first,
                    character.unicodeScalars.count == 1,
                    scalar.properties.generalCategory == .control
                {
                    out += String(format: "\\u{%02X}", scalar.value)
                } else {
                    out.append(character)
                }
            }
        }
        return out
    }

    /// GPT-2's byte-to-unicode table, inverted.
    ///
    /// Built exactly as the reference implementation does: printable Latin-1 ranges
    /// map to themselves, and the remaining 68 bytes are assigned to `U+0100`
    /// onward in ascending byte order.
    public static let byteLevelToByte: [Unicode.Scalar: UInt8] = {
        var identity: [UInt8] = []
        identity.append(contentsOf: UInt8(0x21) ... UInt8(0x7E))  // '!'...'~'
        identity.append(contentsOf: UInt8(0xA1) ... UInt8(0xAC))  // '¡'...'¬'
        identity.append(contentsOf: UInt8(0xAE) ... UInt8(0xFF))  // '®'...'ÿ'

        var mapping: [Unicode.Scalar: UInt8] = [:]
        for byte in identity {
            mapping[Unicode.Scalar(byte)] = byte
        }

        let identitySet = Set(identity)
        var next: UInt32 = 0x100
        for byte in UInt8.min ... UInt8.max where !identitySet.contains(byte) {
            if let scalar = Unicode.Scalar(next) {
                mapping[scalar] = byte
            }
            next += 1
        }
        return mapping
    }()
}
