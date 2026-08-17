import Foundation

/// The play's speaking cast, indexed for the two lookups the rest of the app needs:
/// resolving a name written in a stage direction to the token it speaks under, and
/// attaching a display name and blurb to a token.
///
/// Built once per play. Lives here because the direction scan is what needs the
/// name resolution; `PassageContext` reuses the same index for `WHO THEY ARE`.
struct Cast: Sendable {
    /// Every token that speaks in the play, longest first so `PLAYER KING` wins
    /// over `KING` when both could match a name in a direction.
    private let tokens: [String]
    private let byToken: [String: Persona]

    init(play: Play) {
        var speaking: Set<String> = []
        for act in play.acts {
            for scene in act.scenes {
                for line in scene.lines {
                    if let speaker = line.speaker { speaking.insert(speaker) }
                }
            }
        }
        tokens = speaking.sorted {
            $0.count != $1.count ? $0.count > $1.count : $0 < $1
        }

        var index: [String: Persona] = [:]
        for person in play.personae {
            index[person.name] = person
            for alias in person.aliases { index[alias] = person }
        }
        byToken = index
    }

    func persona(for token: String) -> Persona? { byToken[token] }

    /// A reader-facing name for a speech token. Falls back to title case, so the
    /// collective speakers with no personae entry still read as `All` rather
    /// than `ALL`.
    func display(_ token: String) -> String {
        // Only when the entry is filed under this exact token. An *alias* must not
        // rename the speaker: Hamlet's `FIRST CLOWN.` is filed under `Two Clowns`
        // for its blurb, and showing "Two Clowns" for both clowns would collapse
        // two people into one name.
        if let person = byToken[token], person.name == token { return person.display }
        return token.capitalizedWords
    }

    /// The name for the prompt's `WHO THEY ARE` block, which links the label the
    /// dialogue uses to the name Dramatis Personæ uses when they differ:
    /// `King (Claudius)`. Without that the model has to guess that the `KING:` in
    /// the verse and the `Claudius` in the notes are the same man.
    func label(_ token: String) -> String {
        guard let person = byToken[token], person.name != token,
            person.display.caseInsensitiveCompare(token) != .orderedSame
        else { return display(token) }
        return "\(display(token)) (\(person.display))"
    }

    /// The token a name in a stage direction speaks under, if any.
    ///
    /// Longest match wins, over contiguous word runs: `Enter King Duncan` resolves
    /// to `DUNCAN`, and Hamlet's `Enter the Player King` to `PLAYER KING` rather
    /// than `KING`. Names that are not in the speaking cast (`Attendants`,
    /// `Trumpets`) resolve to nothing, which is how crowd noise gets dropped.
    func resolve(_ name: String) -> String? {
        let words = name.uppercased()
            .split(whereSeparator: { !$0.isLetter && $0 != "’" && $0 != "'" })
            .map(String.init)
        guard !words.isEmpty else { return nil }

        for length in stride(from: words.count, through: 1, by: -1) {
            for start in 0...(words.count - length) {
                let candidate = words[start ..< start + length].joined(separator: " ")
                if byToken[candidate] != nil || tokens.contains(candidate) {
                    return candidate
                }
            }
        }
        return nil
    }
}

/// Who is plausibly on stage, from the scene's own stage directions.
///
/// This is a heuristic and the app says so: the prompt labels the section
/// `ON STAGE (approximate)`. Directions are written for actors, not parsers —
/// `Exeunt all but Hamlet`, `Enter two Clowns with spades and mattocks`, an `Exit`
/// with no name at all — so the scan aims to be usefully right rather than
/// complete.
enum OnStageTracker {

    /// Keywords in the order they must be tested: `Exeunt all but` before
    /// `Exeunt`, and `Exeunt` before `Enter` only because a single direction can
    /// contain both.
    private enum Movement {
        case enter, exit, exeunt, exeuntAllBut
    }

    private static let keywords: [(pattern: String, movement: Movement)] = [
        ("Exeunt all but", .exeuntAllBut),
        ("Exeunt", .exeunt),
        ("Exit", .exit),
        ("Re-enter", .enter),
        ("Enter", .enter),
    ]

    /// Speech labels that stand for a group already named rather than for a person.
    /// `ALL.` and `BOTH MURDERERS.` are not extra bodies to list on stage.
    private static let aggregateLabels: Set<String> = ["ALL", "BOTH"]

    private static func isAggregate(_ token: String) -> Bool {
        aggregateLabels.contains(token) || token.hasPrefix("BOTH ")
    }

    /// Display names of everyone plausibly on stage at `index`.
    ///
    /// A speech line adds its speaker, because someone speaking is necessarily
    /// present — that is what keeps the direction scan from being wrong in a way
    /// that matters. A later `Exit` or `Exeunt` then removes them again, which an
    /// after-the-fact union of every speaker seen could not do: at line 20 of
    /// Hamlet IV.iv it would still list Fortinbras, who walked off at line 9 under
    /// `Exeunt all but the Captain`.
    static func onStage(in scene: Scene, upTo index: Int, cast: Cast) -> [String] {
        var present: [String] = []
        var lastSpeaker: String?

        func add(_ token: String) {
            guard !isAggregate(token), !present.contains(token) else { return }
            present.append(token)
        }

        for line in scene.lines.prefix(min(index + 1, scene.lines.count)) {
            guard line.isDirection else {
                if let speaker = line.speaker {
                    lastSpeaker = speaker
                    add(speaker)
                }
                continue
            }

            for (movement, clause) in movements(in: line.text) {
                let named = names(in: clause).compactMap(cast.resolve)
                switch movement {
                case .enter:
                    named.forEach(add)
                case .exit:
                    if named.isEmpty {
                        // A bare `[_Exit._]` is the speaker who just finished.
                        if let lastSpeaker { present.removeAll { $0 == lastSpeaker } }
                    } else {
                        present.removeAll(where: named.contains)
                    }
                case .exeunt:
                    if named.isEmpty {
                        // A bare `[_Exeunt._]` clears the stage.
                        present.removeAll()
                    } else {
                        present.removeAll(where: named.contains)
                    }
                case .exeuntAllBut:
                    present = named
                }
            }
        }

        return present.map(cast.display)
    }

    /// Splits a direction into (movement, clause) pairs in reading order.
    ///
    /// A clause runs from the keyword to the next `.`, `;`, or `—`, which is what
    /// keeps `Alarum within. Enter King Duncan, Malcolm, Donalbain, Lennox, with
    /// Attendants, meeting a bleeding Captain.` from folding the stage business
    /// into the name list.
    private static func movements(in text: String) -> [(Movement, String)] {
        var found: [(offset: Int, movement: Movement, clause: String)] = []
        var claimed: [Range<Int>] = []

        for (pattern, movement) in keywords {
            var searchRange = text.startIndex ..< text.endIndex
            while let match = text.range(
                of: pattern, options: [.caseInsensitive], range: searchRange)
            {
                let start = text.distance(from: text.startIndex, to: match.lowerBound)
                let end = text.distance(from: text.startIndex, to: match.upperBound)
                // Skip a keyword already covered by a longer one: the `Exeunt`
                // inside `Exeunt all but`, and the `enter` inside `Re-enter`.
                if !claimed.contains(where: { $0.overlaps(start ..< end) }) {
                    claimed.append(start ..< end)
                    let tail = text[match.upperBound...]
                    let stop =
                        tail.firstIndex { $0 == "." || $0 == ";" || $0 == "—" }
                        ?? tail.endIndex
                    found.append((start, movement, String(tail[tail.startIndex ..< stop])))
                }
                searchRange = match.upperBound ..< text.endIndex
            }
        }
        return found.sorted { $0.offset < $1.offset }.map { ($0.movement, $0.clause) }
    }

    /// The candidate names in a clause. Underscores are the transcription's
    /// italic markers, not part of any name.
    private static func names(in clause: String) -> [String] {
        clause
            .replacingOccurrences(of: "_", with: "")
            .replacingOccurrences(of: "]", with: "")
            .replacingOccurrences(of: " and ", with: ",")
            .replacingOccurrences(of: " with ", with: ",")
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }
}

extension String {
    /// `LADY MACBETH` -> `Lady Macbeth`. `capitalized` would also lowercase the
    /// inside of a name that is already mixed case, which is fine here because
    /// every input is a speech token.
    var capitalizedWords: String {
        split(separator: " ")
            .map { $0.prefix(1).uppercased() + $0.dropFirst().lowercased() }
            .joined(separator: " ")
    }
}
