import Foundation

/// The navigator's find field, as a function of the corpus and a query.
///
/// A pure function rather than something the view does inline, so `SelfTest` can
/// assert on the matching rules without a view — and so the view has a single render
/// path: one shape of result covers a query and no query alike.
struct NavigatorSearch {

    /// A play worth showing, with the acts and scenes worth showing inside it.
    struct Match: Identifiable {
        let play: Play
        let acts: [ActMatch]
        /// The query matched the play's own title, so the play came back whole. The
        /// view reads this as "let the reader choose an act": the acts stay shut.
        /// `false` is a settings match, whose acts are forced open around the hits.
        let matchedTitle: Bool

        var id: String { play.id }
    }

    struct ActMatch: Identifiable {
        let act: Act
        /// Every scene of the act for a title match, only the matching ones otherwise.
        let scenes: [Scene]

        var id: Int { act.number }
    }

    /// Plays to list, in corpus order. One rule, so the view never branches:
    ///
    /// - An empty query is every play, act and scene, `matchedTitle: true`. The
    ///   outline alone decides what is visible.
    /// - A title match returns the play whole with `matchedTitle: true`, and beats
    ///   that play's own setting matches: the reader asked for the play, so `"macb"`
    ///   is Macbeth and five act rows rather than 28 scenes.
    /// - A setting match returns `matchedTitle: false` with only the acts that hold
    ///   matching scenes, carrying only those scenes: `"churchyard"` is Hamlet, Act V,
    ///   Scene I.
    /// - Anything unmatched is absent, and `[]` is the view's cue for its "No plays
    ///   match" placeholder.
    static func matches(in corpus: Corpus, query: String) -> [Match] {
        let needle = fold(query.trimmingCharacters(in: .whitespacesAndNewlines))
        guard !needle.isEmpty else { return corpus.plays.map(whole) }

        return corpus.plays.compactMap { play in
            if fold(play.title).contains(needle) { return whole(play) }

            let acts = play.acts.compactMap { act -> ActMatch? in
                let scenes = act.scenes.filter { fold($0.setting).contains(needle) }
                return scenes.isEmpty ? nil : ActMatch(act: act, scenes: scenes)
            }
            guard !acts.isEmpty else { return nil }
            return Match(play: play, acts: acts, matchedTitle: false)
        }
    }

    private static func whole(_ play: Play) -> Match {
        Match(
            play: play,
            acts: play.acts.map { ActMatch(act: $0, scenes: $0.scenes) },
            matchedTitle: true)
    }

    /// Both sides of every comparison go through this: lowercased and
    /// diacritic-insensitive, with apostrophes dropped.
    ///
    /// Titles are mixed case in the JSON (`"King Henry IV, Part 1"`, `"A Midsummer
    /// Night's Dream"`), so folding is what makes `"henry iv"` and `"midsummer"` both
    /// work, and dropping `'` and `’` alike is what lets `"loves labours"` find
    /// *Love's Labour's Lost* without the reader guessing which quote mark the
    /// transcription used.
    private static func fold(_ text: String) -> String {
        text
            .folding(
                options: [.caseInsensitive, .diacriticInsensitive], locale: nil
            )
            .replacingOccurrences(of: "'", with: "")
            .replacingOccurrences(of: "\u{2019}", with: "")
    }
}
