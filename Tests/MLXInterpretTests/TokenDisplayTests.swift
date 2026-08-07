// Copyright © 2026 Apple Inc.

@_spi(Interpret) import MLXInterpret
@_spi(Interpret) import MLXLMCommon
import XCTest

/// Tests for token rendering.
///
/// This looks cosmetic and is not. A readout is a ranked list of vocabulary
/// entries, and byte-level BPE stores each *byte* of a token as a printable
/// codepoint — so without reversing that mapping, every non-ASCII concept renders
/// as mojibake. The real-model experiments surfaced exactly this: the lens vector
/// for "China" reads out as 中国, which appeared as `ä¸ŃåĽ½`.
final class TokenDisplayTests: XCTestCase {

    // MARK: - GPT-2 byte-level BPE

    func testDecodesTheByteLevelSpaceMarker() {
        // `Ġ` (U+0120) is the byte-level spelling of 0x20.
        XCTAssertEqual(TokenDisplay.normalize("\u{0120}the"), " the")
    }

    func testDecodesByteLevelNewlineAndTab() {
        XCTAssertEqual(TokenDisplay.normalize("\u{010A}"), "\\n")
        XCTAssertEqual(TokenDisplay.normalize("\u{0109}"), "\\t")
    }

    /// The case that motivated the decoder: a multi-byte UTF-8 token.
    func testDecodesMultiByteUTF8() {
        // 中国 is E4 B8 AD E5 9B BD in UTF-8. Byte-level BPE maps each byte to a
        // printable codepoint, spelling the token "ä¸ŃåĽ½".
        let bytes: [UInt8] = [0xE4, 0xB8, 0xAD, 0xE5, 0x9B, 0xBD]
        let piece = String(String.UnicodeScalarView(bytes.map { byteLevelScalar(for: $0) }))

        XCTAssertEqual(TokenDisplay.normalize(piece), "中国")
    }

    func testDecodesAccentedLatin() {
        // "café" — the é is C3 A9.
        let bytes = Array("café".utf8)
        let piece = String(String.UnicodeScalarView(bytes.map { byteLevelScalar(for: $0) }))
        XCTAssertEqual(TokenDisplay.normalize(piece), "café")
    }

    /// Every byte must round-trip through the mapping, or some tokens decode wrong
    /// while others decode correctly — the worst failure mode, because it looks
    /// like it works.
    func testEveryByteRoundTrips() {
        for byte in UInt8.min ... UInt8.max {
            let scalar = byteLevelScalar(for: byte)
            XCTAssertEqual(
                TokenDisplay.byteLevelToByte[scalar], byte,
                "byte 0x\(String(byte, radix: 16)) did not round-trip")
        }
    }

    func testMappingCoversExactly256Codepoints() {
        XCTAssertEqual(TokenDisplay.byteLevelToByte.count, 256)
    }

    /// ASCII is identity-mapped, so plain English tokens pass through untouched.
    func testPlainASCIIIsUnchanged() {
        XCTAssertEqual(TokenDisplay.normalize("Paris"), "Paris")
        XCTAssertEqual(TokenDisplay.normalize("spider"), "spider")
    }

    // MARK: - SentencePiece

    func testDecodesSentencePieceSpaceMarker() {
        XCTAssertEqual(TokenDisplay.normalize("\u{2581}dog"), " dog")
    }

    /// Byte-fallback pieces stay escaped: substituting the raw byte would inject a
    /// control character into readout output.
    func testPreservesByteFallbackPieces() {
        XCTAssertEqual(TokenDisplay.normalize("<0x0A>"), "<0x0A>")
        XCTAssertEqual(TokenDisplay.normalize("<0xFF>"), "<0xFF>")
    }

    /// A SentencePiece vocabulary is not byte-level, so pieces containing
    /// codepoints outside the mapping must fall back to the raw text rather than
    /// being dropped.
    func testFallsBackForNonByteLevelPieces() {
        // A literal CJK character, as a SentencePiece vocabulary would store it.
        XCTAssertEqual(TokenDisplay.normalize("中国"), "中国")
        XCTAssertEqual(TokenDisplay.normalize("日本語"), "日本語")
    }

    // MARK: - Tokenizer integration

    func testUnknownIdIsReportedRatherThanCrashing() {
        let tokenizer = StubTokenizer(vocabularySize: 64)
        XCTAssertEqual(TokenDisplay.string(for: 99999, tokenizer: tokenizer), "<unk:99999>")
    }

    func testNormalizedTrimsAndLowercases() {
        let tokenizer = StubTokenizer(vocabularySize: 256, extraWords: ["paris"])
        guard let id = tokenizer.convertTokenToId("paris") else {
            return XCTFail("test vocabulary is missing the expected word")
        }
        XCTAssertEqual(TokenDisplay.normalized(for: id, tokenizer: tokenizer), "paris")
    }

    // MARK: - Helpers

    /// Forward GPT-2 `bytes_to_unicode`, derived independently of the
    /// implementation's inverted table so the tests do not merely restate it.
    private func byteLevelScalar(for byte: UInt8) -> Unicode.Scalar {
        var identity: [UInt8] = []
        identity.append(contentsOf: UInt8(0x21) ... UInt8(0x7E))
        identity.append(contentsOf: UInt8(0xA1) ... UInt8(0xAC))
        identity.append(contentsOf: UInt8(0xAE) ... UInt8(0xFF))

        if identity.contains(byte) {
            return Unicode.Scalar(byte)
        }
        let identitySet = Set(identity)
        var offset: UInt32 = 0
        for candidate in UInt8.min ... UInt8.max where !identitySet.contains(candidate) {
            if candidate == byte { break }
            offset += 1
        }
        return Unicode.Scalar(0x100 + offset)!
    }
}
