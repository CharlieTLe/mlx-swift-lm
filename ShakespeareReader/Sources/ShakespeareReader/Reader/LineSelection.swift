import Foundation

/// A selection of lines, as an anchor and a moving head.
///
/// SwiftUI's `Text` does not expose a selected character range, and a play is
/// line-structured anyway, so selection is a range of indices into
/// `Scene.lines`. That is the better fit regardless: line indices are what the
/// context window, the cache key, and the citation are all built on.
///
/// Anchor and head are kept separate rather than normalized into a range because
/// shift-clicking and shift-arrowing have to extend from the *original* anchor,
/// including backwards through it.
struct LineSelection: Equatable, Sendable, Codable {
    var anchor: Int
    var head: Int

    init(anchor: Int, head: Int) {
        self.anchor = anchor
        self.head = head
    }

    init(at index: Int) {
        self.init(anchor: index, head: index)
    }

    var range: ClosedRange<Int> {
        min(anchor, head) ... max(anchor, head)
    }

    var count: Int { range.count }

    func contains(_ index: Int) -> Bool { range.contains(index) }

    /// The whole speech containing `index`: the contiguous run of one speaker.
    ///
    /// Stage directions interleaved in a speech (`[_Aside._]`, `[_Knocking._]`)
    /// do not break the run, because they are part of what the actor does inside
    /// it. A direction that is not inside any speech selects just itself.
    static func speech(at index: Int, in scene: Scene) -> LineSelection {
        let lines = scene.lines
        guard lines.indices.contains(index) else { return LineSelection(at: index) }

        guard let token = speaker(of: index, in: lines) else {
            return LineSelection(at: index)
        }

        var first = index
        while first > 0, speaker(of: first - 1, in: lines) == token { first -= 1 }
        var last = index
        while last < lines.count - 1, speaker(of: last + 1, in: lines) == token {
            last += 1
        }

        // Directions can trail a speech; a run should not end on one.
        while last > first, lines[last].isDirection { last -= 1 }
        while first < last, lines[first].isDirection { first += 1 }

        return LineSelection(anchor: first, head: last)
    }

    /// The speaker a line belongs to, treating an interleaved direction as part of
    /// the speech it sits inside.
    private static func speaker(of index: Int, in lines: [Line]) -> String? {
        if let speaker = lines[index].speaker { return speaker }
        guard lines[index].isDirection else { return nil }
        // A direction inherits the speech it interrupts only when the same
        // speaker resumes after it; otherwise it stands between two speeches.
        let before = (0 ..< index).last { lines[$0].speaker != nil }
        let after = (index + 1 ..< lines.count).first { lines[$0].speaker != nil }
        guard let before, let after,
            lines[before].speaker == lines[after].speaker,
            !lines[after].startsSpeech
        else { return nil }
        return lines[before].speaker
    }

    /// Moves the head, keeping the anchor. Used by shift-click and shift-arrow.
    mutating func extend(to index: Int) {
        head = index
    }

    /// Clamps into a scene, so a stale selection can never index out of bounds
    /// after the reader switches scenes.
    func clamped(to lines: [Line]) -> LineSelection? {
        guard !lines.isEmpty else { return nil }
        let upper = lines.count - 1
        return LineSelection(
            anchor: min(max(0, anchor), upper),
            head: min(max(0, head), upper))
    }
}
