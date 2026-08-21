import CryptoKit
import Foundation

/// Everything the model is told about a passage, assembled deterministically from
/// the play's own structure.
///
/// No embeddings: the act / scene / speaker hierarchy is a better index than a
/// vector store here, and it is exact. The only inexact part is `onStage`, and it
/// is labelled as such all the way through to the prompt.
struct PassageContext: Sendable, Hashable {

    /// One line as the prompt will render it.
    struct Utterance: Sendable, Hashable {
        var speaker: String?
        var number: Int?
        var text: String
        var isDirection: Bool
    }

    /// A `(display, blurb)` pair. A struct rather than a tuple so the whole context
    /// can stay `Hashable`, which is what lets the view diff it cheaply.
    struct PersonaNote: Sendable, Hashable {
        var display: String
        var blurb: String
    }

    var playTitle: String
    var author: String
    var act: Int
    var scene: Int
    var setting: String?
    var openingDirection: String?
    /// Approximate, from the direction scan. See `OnStageTracker`.
    var onStage: [String]
    /// Only the speakers appearing in the window, and only those with a blurb
    /// worth its tokens.
    var personae: [PersonaNote]
    /// Up to 15 lines: one speech plus its cue.
    var preceding: [Utterance]
    /// Verbatim, with interleaved directions kept.
    var selected: [Utterance]
    /// Up to 4 lines: the reaction the passage lands on.
    var following: [Utterance]
    /// Per-scene, cached. `nil` never blocks a generation.
    var synopsis: String?
    var synopsisIsPartial: Bool
    var citation: String
    var key: PassageKey
    /// SHA-256 of the selected line text. This is what stops a re-parsed corpus
    /// with shifted line indices from serving an annotation of different lines —
    /// the one failure mode that would otherwise be invisible.
    var digest: String

    static let precedingLimit = 15
    static let followingLimit = 4

    static func build(
        play: Play,
        key: SceneKey,
        scene: Scene,
        selection: LineSelection,
        cast: Cast,
        synopsis: String? = nil,
        synopsisIsPartial: Bool = false
    ) -> PassageContext? {
        guard let clamped = selection.clamped(to: scene.lines) else { return nil }
        let range = clamped.range

        let precedingStart = precedingWindowStart(range.lowerBound, in: scene.lines)
        let preceding = utterances(scene.lines[precedingStart ..< range.lowerBound])
        let selected = utterances(scene.lines[range])

        // Half-open, so a selection ending at the last line of the scene yields an
        // empty slice rather than an invalid range.
        let followingStart = range.upperBound + 1
        let followingEnd = min(scene.lines.count, followingStart + followingLimit)
        let following =
            followingStart < followingEnd
            ? utterances(scene.lines[followingStart ..< followingEnd]) : []

        let windowSpeakers = OrderedSet(
            (preceding + selected + following).compactMap(\.speaker))
        let personae = windowSpeakers.values.compactMap { token -> PersonaNote? in
            guard let person = cast.persona(for: token), !person.blurb.isEmpty
            else { return nil }
            return PersonaNote(display: cast.label(token), blurb: person.blurb)
        }

        return PassageContext(
            playTitle: play.title,
            author: play.author,
            act: key.act,
            scene: key.scene,
            setting: scene.setting.isEmpty ? nil : scene.setting,
            openingDirection: scene.openingDirection,
            onStage: OnStageTracker.onStage(
                in: scene, upTo: range.upperBound, cast: cast),
            personae: personae,
            preceding: preceding,
            selected: selected,
            following: following,
            synopsis: synopsis,
            synopsisIsPartial: synopsisIsPartial,
            citation: Citation.string(play: play, key: key, scene: scene, range: range),
            key: PassageKey(
                scene: key, first: range.lowerBound, last: range.upperBound),
            digest: digest(of: selected))
    }

    /// Start of the preceding window, pulled back to a speech boundary when one
    /// falls inside it. Beginning mid-speech reads as a fragment and costs the same
    /// tokens as beginning at the cue.
    private static func precedingWindowStart(
        _ lowerBound: Int, in lines: [Line]
    ) -> Int {
        let floor = max(0, lowerBound - precedingLimit)
        let boundary = (floor ..< lowerBound).first { lines[$0].startsSpeech }
        return boundary ?? floor
    }

    private static func utterances(_ lines: ArraySlice<Line>) -> [Utterance] {
        lines.map {
            Utterance(
                speaker: $0.speaker,
                number: $0.number,
                text: $0.plainText,
                isDirection: $0.isDirection)
        }
    }

    private static func digest(of selected: [Utterance]) -> String {
        let joined = selected.map(\.text).joined(separator: "\n")
        return SHA256.hash(data: Data(joined.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
    }
}

/// Insertion-ordered unique values. The window's speakers have to keep the order
/// they appear in, because that is the order a reader meets them.
private struct OrderedSet<Element: Hashable> {
    private(set) var values: [Element] = []
    private var seen: Set<Element> = []

    init(_ elements: some Sequence<Element>) {
        for element in elements where !seen.contains(element) {
            seen.insert(element)
            values.append(element)
        }
    }
}

/// The reference printed with every annotation.
enum Citation {
    /// `Hamlet · III.i.56-61 (this edition)`.
    ///
    /// The "(this edition)" is not modesty. These line numbers are assigned
    /// sequentially within the scene over speech lines only; Folger and Arden count
    /// a verse line shared between two speakers once, so the numbers do not agree
    /// and the citation must not imply they do.
    static func string(play: Play, key: SceneKey, scene: Scene, range: ClosedRange<Int>)
        -> String
    {
        let title = play.title.split(separator: ",").first.map(String.init) ?? play.title
        let location =
            "\(RomanNumeral.string(key.act)).\(SceneLabel.citation(key.scene))"

        let numbers = scene.lines[range].compactMap(\.number)
        guard let first = numbers.first, let last = numbers.last else {
            // A selection of stage directions alone has no line numbers to cite.
            return "\(title) · \(location) (stage direction, this edition)"
        }
        let lines = first == last ? "\(first)" : "\(first)-\(last)"
        return "\(title) · \(location).\(lines) (this edition)"
    }
}
