import Foundation

/// One play, as produced by `tools/build_corpus.py`.
///
/// The JSON is checked in, so nothing here depends on the script having been run.
/// Decoding is strict — a field the parser stopped emitting is a decode failure the
/// self test catches, not a silently empty reader pane.
struct Play: Codable, Sendable, Identifiable {
    let schemaVersion: Int
    let id: String
    let title: String
    let author: String
    /// Always `sequential-within-scene` today. Recorded per play rather than
    /// assumed, because it is what the citation's "(this edition)" is claiming.
    let numbering: String
    let source: Source
    let personae: [Persona]
    let acts: [Act]

    struct Source: Codable, Sendable {
        let kind: String
        let ebookID: Int
        let url: String
        let retrieved: String
        let textSHA256: String
        let parserVersion: Int
        let note: String
    }
}

/// A character from Dramatis Personæ.
///
/// `aliases` carries the speech tokens that resolve to this entry, because the
/// personae list names the character and the dialogue labels the role: Hamlet's
/// Claudius and Gertrude speak throughout as `KING.` and `QUEEN.` The collective
/// and numbered speakers (`ALL.`, `BOTH MURDERERS.`, `FIRST WITCH.`) have no entry
/// at all and are deliberately left unmatched.
struct Persona: Codable, Sendable, Hashable {
    let name: String
    let display: String
    let blurb: String
    let aliases: [String]
}

struct Act: Codable, Sendable, Hashable {
    let number: Int
    let scenes: [Scene]
}

struct Scene: Codable, Sendable, Hashable {
    let number: Int
    let setting: String
    let lines: [Line]

    /// The direction the scene opens with, if it has one. Worth its own accessor
    /// because it is the single most informative line for a reader arriving cold:
    /// it usually names who walks on. A later `Enter` is not the opening, so this
    /// looks only at the first line.
    var openingDirection: String? {
        guard let first = lines.first, first.isDirection else { return nil }
        return first.plainText
    }

    var speechLineCount: Int { lines.count { $0.kind == .speech } }
}

/// One rendered line: either a verse/prose line of a speech, or a stage direction.
///
/// The speaker is repeated on every speech line so a single line is
/// self-describing and the context builder needs no backward walk. `speechStart`
/// tells the reader view where to print a heading.
struct Line: Codable, Sendable, Hashable {
    enum Kind: String, Codable, Sendable {
        case speech
        case direction
    }

    let kind: Kind
    let speaker: String?
    let number: Int?
    let text: String
    let speechStart: Bool?

    var isDirection: Bool { kind == .direction }
    var startsSpeech: Bool { speechStart ?? false }

    /// Direction text without the transcription's markup.
    ///
    /// The source marks italics with underscores and *usually* wraps a direction in
    /// brackets — `[_Exeunt._]` — but not always: Hamlet's `Enter Francisco and
    /// Barnardo, two sentinels.` has neither. Normalizing here means every consumer
    /// gets plain text and adds its own brackets exactly once, rather than one of
    /// them rendering `[[Exit.]]`.
    var plainText: String {
        guard isDirection else { return text }
        return text
            .replacingOccurrences(of: "_", with: "")
            .trimmingCharacters(in: CharacterSet(charactersIn: "[] "))
    }
}

/// Addresses one scene. Selections never cross a scene boundary because the
/// reader renders one scene at a time, so this is also the unit the synopsis
/// cache is keyed on.
struct SceneKey: Hashable, Sendable, Codable {
    var playID: String
    var act: Int
    var scene: Int

    var slug: String { "\(playID)-a\(act)s\(scene)" }
}

/// Addresses one selected passage.
///
/// `first` and `last` are indices into `Scene.lines`, not the citation's line
/// numbers. Indices are always defined — a selection can consist of nothing but
/// stage directions — and a re-parse that shifts them is caught by the passage
/// digest rather than by the key.
struct PassageKey: Hashable, Sendable, Codable {
    var scene: SceneKey
    var first: Int
    var last: Int

    var slug: String { "\(scene.slug)-\(first)_\(last)" }
}

// MARK: - Roman numerals

enum RomanNumeral {
    private static let values: [(Int, String)] = [
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]

    /// Acts are cited in upper case and scenes in lower, which is the convention
    /// every printed edition uses: `III.i`.
    static func string(_ number: Int, uppercase: Bool = true) -> String {
        var remainder = number
        var result = ""
        for (value, symbol) in values {
            while remainder >= value {
                result += symbol
                remainder -= value
            }
        }
        return uppercase ? result : result.lowercased()
    }
}

/// How a scene number is named and cited.
///
/// Exists because scene 0 is real: `tools/build_corpus.py` files a chorus block
/// (Romeo and Juliet's two Prologues) as scene 0 of the act it opens, and
/// `RomanNumeral.string(0)` is the empty string, so every call site would otherwise
/// render "Scene " and cite "I..1-14".
enum SceneLabel {
    /// The reader-facing name, for the navigator row and the scene heading.
    static func string(_ number: Int) -> String {
        number == 0 ? "Prologue" : "Scene \(RomanNumeral.string(number))"
    }

    /// The citation component. `Pro` is the Folger convention: `I.Pro.1-14`.
    static func citation(_ number: Int) -> String {
        number == 0 ? "Pro" : RomanNumeral.string(number, uppercase: false)
    }
}
