import Foundation
import MLXLLM
import MLXLMCommon

/// Model-free assertions, run by `--selftest`. No download, no network, no GPU.
///
/// They live in the executable because there is only one target: a test target
/// would need the corpus resources and the app's types duplicated or exported.
/// What they are for is the two things that break silently — a corpus that decodes
/// but is subtly wrong, and prompt drift.
enum SelfTest {

    /// Collects failures rather than trapping, so one run reports everything that
    /// is wrong instead of only the first thing. A local object rather than static
    /// state: mutable global state would need concurrency annotations it has no
    /// business needing.
    final class Log {
        private(set) var failures: [String] = []

        func fail(_ message: String) { failures.append(message) }

        func check(_ condition: Bool, _ message: @autoclosure () -> String) {
            if !condition { fail(message()) }
        }

        func equal<T: Equatable>(
            _ lhs: T, _ rhs: T, _ label: @autoclosure () -> String
        ) {
            if lhs != rhs { fail("\(label()): expected \(rhs), got \(lhs)") }
        }
    }

    static func run() -> Bool {
        let log = Log()

        corpus(log)
        selection(log)
        onStage(log)
        followUpParsing(log)
        goldenPromptRender(log)
        readerFonts(log)
        readingProgress(log)

        if log.failures.isEmpty {
            print("selftest: all checks passed")
            return true
        }
        for failure in log.failures {
            print("selftest FAIL: \(failure)")
        }
        print("selftest: \(log.failures.count) failure(s)")
        return false
    }

    // MARK: - Corpus

    /// Exact counts, measured on the real Gutenberg files. They are regression
    /// targets, not estimates.
    ///
    /// The speech-heading counts are the number of `speechStart` lines *after* the
    /// PG footer is stripped and after the two heading-shaped personae entries in
    /// Macbeth (`LADY MACBETH.`, `LADY MACDUFF.`) are excluded by the body scan. A
    /// raw count of plain all-caps headings gives 1,122 and 651 instead: 1,122
    /// because `DAMAGE.` from the license footer parses as one, and both numbers
    /// because that shape misses the headings the parser now also recognises — 11
    /// joint ones in Hamlet (`HORATIO and MARCELLUS.`) and 1 in Macbeth
    /// (`MACBETH, LENNOX.`), the 4 title-case collectives (`All.`, `Both.`,
    /// `Danes.`), and the one `BARNARDO` that lost its period.
    ///
    /// Romeo and Juliet's 840 include 2 `CHORUS.` headings inside the two scene-0
    /// chorus blocks, which are dropped entirely if the Prologue is not parsed, and
    /// 2 headings the transcription left on the same line as their first verse line
    /// (`ROMEO. Nurse, commend me…`, `THIRD WATCH. Here is a Friar…`), which are
    /// swallowed into the previous speaker's text if they are not recovered.
    ///
    /// `collectives` is only filled in for those three. It names speakers absent from
    /// Dramatis Personæ, and picking them is a reading of the play rather than
    /// something a count can produce, so the other 32 assert their shape and their
    /// three totals and leave it at that. Two of those totals are worth more than
    /// they look: As You Like It's 22 scenes and 25 personae, and The Winter's Tale's
    /// 15 and 28, are what fail if the personae block runs on into the body again —
    /// it used to swallow both plays whole, for 2,930 and 3,250 cast entries.
    private struct Expected {
        let acts: Int
        let scenes: Int
        let speechHeadings: Int
        var collectives: [String] = []
    }

    private static let expected: [String: Expected] = [
        "hamlet": Expected(
            acts: 5, scenes: 20, speechHeadings: 1137,
            collectives: ["ALL", "BOTH", "DANES", "FIRST CLOWN"]),
        "macbeth": Expected(
            acts: 5, scenes: 28, speechHeadings: 649,
            collectives: ["ALL", "BOTH MURDERERS", "FIRST WITCH", "APPARITION"]),
        // 26 scenes: 24 numbered plus the two Chorus blocks, filed as scene 0 of
        // Acts I and II. `THIRD WATCH` is the canary for the inline-heading rule:
        // it has no other heading anywhere in the play.
        "romeo-and-juliet": Expected(
            acts: 5, scenes: 26, speechHeadings: 840,
            collectives: ["FIRST CITIZEN", "FIRST SERVANT", "FIRST WATCH", "THIRD WATCH"]),
        "alls-well-that-ends-well": Expected(acts: 5, scenes: 23, speechHeadings: 936),
        "antony-and-cleopatra": Expected(acts: 5, scenes: 42, speechHeadings: 1176),
        "as-you-like-it": Expected(acts: 5, scenes: 22, speechHeadings: 812),
        "comedy-of-errors": Expected(acts: 5, scenes: 11, speechHeadings: 609),
        "coriolanus": Expected(acts: 5, scenes: 29, speechHeadings: 1104),
        "cymbeline": Expected(acts: 5, scenes: 29, speechHeadings: 857),
        "henry-iv-part-1": Expected(acts: 5, scenes: 19, speechHeadings: 775),
        "henry-iv-part-2": Expected(acts: 5, scenes: 19, speechHeadings: 900),
        // 27 scenes: 23 numbered plus the 4 Chorus blocks that open Acts II-V. Their
        // headings are `CHORUS.`, which is why this play needs no special case where
        // Troilus and Pericles are still unparsed.
        "henry-v": Expected(acts: 5, scenes: 27, speechHeadings: 739),
        "henry-vi-part-1": Expected(acts: 5, scenes: 27, speechHeadings: 644),
        "henry-vi-part-2": Expected(acts: 5, scenes: 24, speechHeadings: 768),
        "henry-vi-part-3": Expected(acts: 5, scenes: 28, speechHeadings: 782),
        // 17 scenes: 16 numbered plus the Prologue, scene 0 of Act I.
        "henry-viii": Expected(acts: 5, scenes: 17, speechHeadings: 709),
        "julius-caesar": Expected(acts: 5, scenes: 18, speechHeadings: 795),
        "king-john": Expected(acts: 5, scenes: 16, speechHeadings: 551),
        "king-lear": Expected(acts: 5, scenes: 26, speechHeadings: 1066),
        "loves-labours-lost": Expected(acts: 5, scenes: 9, speechHeadings: 1045),
        "measure-for-measure": Expected(acts: 5, scenes: 17, speechHeadings: 897),
        "merchant-of-venice": Expected(acts: 5, scenes: 20, speechHeadings: 634),
        "merry-wives-of-windsor": Expected(acts: 5, scenes: 23, speechHeadings: 1017),
        "midsummer-nights-dream": Expected(acts: 5, scenes: 9, speechHeadings: 489),
        "much-ado-about-nothing": Expected(acts: 5, scenes: 13, speechHeadings: 977),
        "othello": Expected(acts: 5, scenes: 15, speechHeadings: 1179),
        "richard-ii": Expected(acts: 5, scenes: 19, speechHeadings: 554),
        "richard-iii": Expected(acts: 5, scenes: 25, speechHeadings: 1078),
        "taming-of-the-shrew": Expected(acts: 5, scenes: 12, speechHeadings: 817),
        "tempest": Expected(acts: 5, scenes: 9, speechHeadings: 646),
        "timon-of-athens": Expected(acts: 5, scenes: 18, speechHeadings: 802),
        "titus-andronicus": Expected(acts: 5, scenes: 14, speechHeadings: 565),
        // The play the trailing-period act header was added for: its body sets `ACT
        // I.` where its Contents block sets `ACT I`, and it failed to parse at all
        // until the pattern tolerated the period.
        "twelfth-night": Expected(acts: 5, scenes: 18, speechHeadings: 922),
        "two-gentlemen-of-verona": Expected(acts: 5, scenes: 20, speechHeadings: 858),
        "winters-tale": Expected(acts: 5, scenes: 15, speechHeadings: 738),
    ]

    private static func corpus(_ log: Log) {
        guard let corpus = try? CorpusLoader.load() else {
            log.fail("the corpus did not load")
            return
        }
        log.check(corpus.plays.count >= 2, "expected at least two plays")

        // Every play `expected` describes is actually in the bundle. `CorpusLoader`
        // enumerates the directory, so a JSON file that failed to generate or never
        // got added leaves a corpus that loads, reads correctly, and is quietly short
        // a play — and the per-play loop below cannot see what is not there.
        let loaded = Set(corpus.plays.map(\.id))
        for id in expected.keys.sorted() where !loaded.contains(id) {
            log.fail("\(id).json is missing from the bundled corpus")
        }

        for play in corpus.plays {
            log.equal(play.schemaVersion, 1, "\(play.id) schemaVersion")
            log.equal(play.numbering, "sequential-within-scene", "\(play.id) numbering")
            log.check(!play.personae.isEmpty, "\(play.id) has no personae")

            var speakers: Set<String> = []
            var headings = 0
            var scenes = 0

            for act in play.acts {
                for scene in act.scenes {
                    scenes += 1
                    log.check(
                        scene.speechLineCount > 0,
                        "\(play.id) \(act.number).\(scene.number) has no speech lines")

                    // Contiguous per-scene numbering, from 1, over speech lines only.
                    var next = 1
                    for line in scene.lines {
                        switch line.kind {
                        case .speech:
                            log.equal(
                                line.number, next,
                                "\(play.id) \(act.number).\(scene.number) line numbering")
                            log.check(
                                line.speaker != nil,
                                "\(play.id) speech line with no speaker")
                            next += 1
                            if line.startsSpeech { headings += 1 }
                            if let speaker = line.speaker { speakers.insert(speaker) }
                        case .direction:
                            log.check(
                                line.number == nil,
                                "\(play.id) direction carries a line number")
                        }
                    }
                }
            }

            // Proves the PG license footer was stripped: leave it in and `DAMAGE.`
            // parses as a speaker. Checked for every play rather than only the ones
            // with a heading count, since the footer is identical in all 35 files and
            // this is the cheapest possible proof the body scan found its end.
            log.check(
                !speakers.contains("DAMAGE"),
                "\(play.id) has DAMAGE as a speaker — the PG footer was not stripped")

            guard let target = expected[play.id] else { continue }
            log.equal(play.acts.count, target.acts, "\(play.id) acts")
            log.equal(scenes, target.scenes, "\(play.id) scenes")
            log.equal(headings, target.speechHeadings, "\(play.id) speech headings")

            // Proves speakers are not gated on Dramatis Personæ, where none of
            // these appear.
            for token in target.collectives {
                log.check(
                    speakers.contains(token),
                    "\(play.id) lost the collective speaker \(token)")
            }
        }

        soliloquy(log, in: corpus)
        resumedSpeech(log, in: corpus)
        chorus(log, in: corpus)
        balconyDirection(log, in: corpus)
        inlineHeading(log, in: corpus)
    }

    /// "To be, or not to be" is exactly 35 consecutive `HAMLET` lines, with no
    /// internal blank in the source and so no break in the parse.
    private static func soliloquy(_ log: Log, in corpus: Corpus) {
        let key = SceneKey(playID: "hamlet", act: 3, scene: 1)
        guard let scene = corpus.scene(key) else {
            log.fail("Hamlet III.i not found")
            return
        }
        let speech = scene.lines.filter { $0.kind == .speech }
        guard let start = speech.firstIndex(where: {
            $0.text.hasPrefix("To be, or not to be")
        }) else {
            log.fail("'To be, or not to be' not found in Hamlet III.i")
            return
        }
        let run = speech[start...].prefix { $0.speaker == "HAMLET" }
        log.equal(run.count, 35, "the soliloquy's length")
    }

    /// A speech that Gutenberg interrupts with a blank-delimited unbracketed
    /// direction and then resumes with no repeated heading.
    ///
    /// The whole class of bug in one passage: `Re-enter Ghost.` used to close
    /// Horatio's speech, so everything under it — and 250 other verse lines in
    /// Hamlet alone, Claudius's prayer among them — was filed as one stage
    /// direction, unnumbered and unattributable. This is what fails loudly if the
    /// direction vocabulary regresses.
    private static func resumedSpeech(_ log: Log, in corpus: Corpus) {
        guard let scene = corpus.scene(SceneKey(playID: "hamlet", act: 1, scene: 1))
        else {
            log.fail("Hamlet I.i not found")
            return
        }
        guard
            let at = scene.lines.firstIndex(where: {
                $0.text.hasPrefix("But, soft, behold!")
            }), at > 0
        else {
            log.fail("'But, soft, behold!' not found in Hamlet I.i")
            return
        }
        let resumed = scene.lines[at]
        log.equal(resumed.kind, .speech, "'But, soft, behold!' kind")
        log.equal(resumed.speaker, "HORATIO", "'But, soft, behold!' speaker")
        log.check(resumed.number != nil, "'But, soft, behold!' carries no line number")
        // A resumed speech prints no second heading, matching print convention.
        log.equal(resumed.startsSpeech, false, "'But, soft, behold!' startsSpeech")

        let before = scene.lines[at - 1]
        log.equal(before.kind, .direction, "the line above 'But, soft, behold!'")
        log.equal(before.text, "Re-enter Ghost.", "the interrupting direction")
    }

    /// Romeo and Juliet's two Chorus blocks, filed as scene 0 of the act they open.
    ///
    /// The Prologue is the passage a parser loses most quietly: it sits *above*
    /// `ACT I` in the transcription, so anchoring the body scan on the first act
    /// header dropped all 14 lines without even counting them as unclassified. The
    /// Act II Chorus was the mirror image, inside an act but above the first
    /// `SCENE` header, so it fell to `unclassified` whole.
    private static func chorus(_ log: Log, in corpus: Corpus) {
        for act in [1, 2] {
            guard
                let scene = corpus.scene(
                    SceneKey(playID: "romeo-and-juliet", act: act, scene: 0))
            else {
                log.fail("Romeo and Juliet \(act).0 not found")
                continue
            }
            log.equal(scene.openingDirection, "Enter Chorus.", "the \(act).0 opening")
            let speech = scene.lines.filter { $0.kind == .speech }
            log.equal(speech.count, 14, "the length of the \(act).0 sonnet")
            log.check(
                speech.allSatisfy { $0.speaker == "CHORUS" },
                "Romeo and Juliet \(act).0 has a speaker other than CHORUS")
        }

        log.equal(
            corpus.scene(SceneKey(playID: "romeo-and-juliet", act: 1, scene: 0))?
                .lines.first(where: { $0.kind == .speech })?.text,
            "Two households, both alike in dignity,", "the Prologue's first line")

        // Scene 0 needs its own naming: `RomanNumeral.string(0)` is the empty
        // string, so the reader would show "Scene " and cite "I..1-14".
        log.equal(SceneLabel.string(0), "Prologue", "the scene-0 label")
        log.equal(SceneLabel.citation(0), "Pro", "the scene-0 citation component")
    }

    /// The balcony scene's own direction, which is unbracketed and matches no
    /// general direction opener.
    ///
    /// Left out of the vocabulary, ` Juliet appears above at a window.` is emitted
    /// as Romeo's speech line 2, which both puts a stage direction in his mouth and
    /// shifts every citation in II.ii by one.
    private static func balconyDirection(_ log: Log, in corpus: Corpus) {
        guard
            let scene = corpus.scene(
                SceneKey(playID: "romeo-and-juliet", act: 2, scene: 2))
        else {
            log.fail("Romeo and Juliet II.ii not found")
            return
        }
        guard
            let at = scene.lines.firstIndex(where: {
                $0.text.hasPrefix("But soft, what light")
            }), at > 0
        else {
            log.fail("'But soft, what light' not found in Romeo and Juliet II.ii")
            return
        }
        let before = scene.lines[at - 1]
        log.equal(before.kind, .direction, "the line above 'But soft, what light'")
        log.equal(
            before.text, "Juliet appears above at a window.",
            "the balcony scene's own direction")
        log.equal(scene.lines[at].number, 2, "'But soft, what light' line number")
    }

    /// A heading the transcription left on the same line as its first verse line.
    ///
    /// `THIRD WATCH.` has no heading of its own anywhere in the play, so without the
    /// inline rule this speech is attributed to whoever spoke last, with the heading
    /// left sitting inside that speaker's own text, and the Watch's third man never
    /// enters the cast at all.
    private static func inlineHeading(_ log: Log, in corpus: Corpus) {
        guard
            let scene = corpus.scene(
                SceneKey(playID: "romeo-and-juliet", act: 5, scene: 3))
        else {
            log.fail("Romeo and Juliet V.iii not found")
            return
        }
        guard
            let line = scene.lines.first(where: {
                $0.text.hasPrefix("Here is a Friar that trembles")
            })
        else {
            log.fail("'Here is a Friar that trembles' not found in R&J V.iii")
            return
        }
        log.equal(line.speaker, "THIRD WATCH", "the recovered inline speaker")
        log.equal(line.startsSpeech, true, "the recovered inline heading")
    }

    // MARK: - Selection

    private static func selection(_ log: Log) {
        // A scene shaped like the real thing: a direction, two speeches, an
        // interleaved direction, and a trailing direction.
        let lines = [
            direction("Enter Hamlet and Horatio."),  // 0
            speech("HAMLET", 1, "Line one.", start: true),  // 1
            speech("HAMLET", 2, "Line two."),  // 2
            direction("[_Aside._]"),  // 3
            speech("HAMLET", 3, "Line three."),  // 4
            speech("HORATIO", 4, "Line four.", start: true),  // 5
            direction("[_Exit._]"),  // 6
        ]
        let scene = Scene(number: 1, setting: "Nowhere.", lines: lines)

        // Range normalizes regardless of drag direction.
        log.equal(LineSelection(anchor: 4, head: 1).range, 1...4, "a backwards range")
        log.equal(LineSelection(anchor: 1, head: 4).range, 1...4, "a forwards range")
        log.equal(LineSelection(at: 2).count, 1, "a single-line selection")

        // Shift-click backwards through the anchor keeps the anchor.
        var backwards = LineSelection(at: 4)
        backwards.extend(to: 1)
        log.equal(backwards.anchor, 4, "the anchor after extending backwards")
        log.equal(backwards.range, 1...4, "the range after extending backwards")

        // Drag reversal: past the anchor and back again.
        var reversed = LineSelection(at: 2)
        reversed.extend(to: 5)
        reversed.extend(to: 1)
        log.equal(reversed.range, 1...2, "the range after a drag reverses")

        // Double-click takes the whole speech, across the interleaved direction.
        log.equal(
            LineSelection.speech(at: 2, in: scene).range, 1...4,
            "a double-clicked speech spanning a direction")
        log.equal(
            LineSelection.speech(at: 3, in: scene).range, 1...4,
            "a double-click on a direction inside a speech")
        // A direction between two different speakers stands alone.
        log.equal(
            LineSelection.speech(at: 6, in: scene).range, 6...6,
            "a double-clicked trailing direction")
        log.equal(
            LineSelection.speech(at: 0, in: scene).range, 0...0,
            "a double-clicked opening direction")
        log.equal(
            LineSelection.speech(at: 5, in: scene).range, 5...5,
            "a double-clicked one-line speech")

        // Clamping at the scene edges.
        log.equal(
            LineSelection(anchor: -4, head: 99).clamped(to: lines)?.range, 0...6,
            "clamping past both edges")
        log.check(
            LineSelection(at: 0).clamped(to: [])?.range == nil,
            "clamping into an empty scene should yield nil")
        log.equal(
            LineSelection.speech(at: 42, in: scene).range, 42...42,
            "a double-click on an out-of-range index")

        // Arrow moves. Nothing selected *lands* rather than steps, whichever way it was
        // pressed: the navigator clears the selection on every scene change, so stepping
        // from an implied head of 0 skipped line one of every scene.
        func moved(_ from: LineSelection?, _ step: Int, extending: Bool = false)
            -> LineSelection?
        {
            LineSelection.moved(from: from, by: step, extending: extending, in: lines)
        }

        log.equal(moved(nil, 1), LineSelection(at: 0), "the first press down")
        log.equal(moved(nil, -1), LineSelection(at: 0), "the first press up")
        log.equal(
            moved(nil, 1, extending: true), LineSelection(at: 0),
            "shift-down with no anchor to keep")
        log.equal(moved(LineSelection(at: 2), 1), LineSelection(at: 3), "stepping down")
        log.equal(moved(LineSelection(at: 2), -1), LineSelection(at: 1), "stepping up")

        // A collapsed selection moves as a whole rather than shrinking.
        log.equal(
            moved(LineSelection(anchor: 1, head: 4), 1), LineSelection(at: 5),
            "an unshifted step out of a range")

        // Off the edge with nothing to extend: the caller's cue to roll scenes.
        log.equal(moved(LineSelection(at: 6), 1), nil, "down at the last line")
        log.equal(moved(LineSelection(at: 0), -1), nil, "up at line 0")

        // Extending never rolls — a selection is scene-scoped, so it stops at the edge.
        log.equal(
            moved(LineSelection(at: 6), 1, extending: true), LineSelection(at: 6),
            "shift-down at the last line")
        log.equal(
            moved(LineSelection(at: 0), -1, extending: true), LineSelection(at: 0),
            "shift-up at line 0")
        log.equal(
            moved(LineSelection(anchor: 4, head: 3), -1, extending: true),
            LineSelection(anchor: 4, head: 2),
            "shift-up back through the anchor keeps it")

        log.equal(
            LineSelection.moved(from: nil, by: 1, extending: false, in: []), nil,
            "an arrow move in an empty scene")
        log.equal(
            LineSelection.moved(
                from: LineSelection(at: 0), by: 1, extending: false, in: []),
            nil, "an arrow move in an empty scene with a stale selection")
    }

    private static func speech(
        _ speaker: String, _ number: Int, _ text: String, start: Bool = false
    ) -> Line {
        Line(
            kind: .speech, speaker: speaker, number: number, text: text,
            speechStart: start)
    }

    private static func direction(_ text: String) -> Line {
        Line(kind: .direction, speaker: nil, number: nil, text: text, speechStart: nil)
    }

    // MARK: - On stage

    /// The direction scan, against four real scenes chosen because each one broke a
    /// naive version of it.
    private static func onStage(_ log: Log) {
        guard let corpus = try? CorpusLoader.load() else { return }

        /// Who the tracker reports at the line numbered `line` of a scene.
        func present(_ playID: String, _ act: Int, _ scene: Int, atLine line: Int)
            -> [String]
        {
            guard let play = corpus.play(playID),
                let scene = corpus.scene(
                    SceneKey(playID: playID, act: act, scene: scene)),
                let index = scene.lines.firstIndex(where: { $0.number == line })
            else {
                log.fail("could not locate \(playID) \(act).\(scene) line \(line)")
                return []
            }
            return OnStageTracker.onStage(
                in: scene, upTo: index, cast: Cast(play: play))
        }

        // `Exeunt all but the Captain` at line 9 takes Fortinbras off. He still
        // speaks inside the passage window, so a version that unioned in every
        // speaker seen kept reporting him as present.
        log.equal(
            present("hamlet", 4, 4, atLine: 20),
            ["Captain", "Hamlet", "Rosencrantz", "Guildenstern"],
            "Hamlet IV.iv after Fortinbras marches off")

        // Both grave-diggers are filed under the one `Two Clowns` personae entry.
        // Using its display name for both collapsed them into "Two Clowns, Two
        // Clowns".
        log.equal(
            present("hamlet", 5, 1, atLine: 12), ["First Clown", "Second Clown"],
            "Hamlet V.i, the two grave-diggers")

        // `ALL.` is a label for everyone already named, not a fourth witch.
        log.equal(
            present("macbeth", 1, 3, atLine: 48),
            ["First Witch", "Second Witch", "Third Witch", "Macbeth", "Banquo"],
            "Macbeth I.iii, where ALL. speaks")

        // `Enter Ghost and Hamlet.` with no exit in between.
        log.equal(
            present("hamlet", 1, 5, atLine: 96), ["Ghost", "Hamlet"],
            "Hamlet I.v")

        // The personae/dialogue name link, which the prompt's WHO THEY ARE block
        // depends on: Claudius speaks throughout as `KING.`
        guard let hamlet = corpus.play("hamlet") else { return }
        let cast = Cast(play: hamlet)
        log.equal(cast.label("KING"), "King (Claudius)", "the KING label")
        log.equal(cast.display("KING"), "King", "the KING display name")
        log.equal(cast.label("HAMLET"), "Hamlet", "an unaliased label")
        log.equal(
            cast.resolve("the Player King"), "PLAYER KING", "longest-match resolution")
        log.equal(cast.resolve("Trumpets"), nil, "a non-cast name in a direction")

        // `Enter King Duncan` has to reach DUNCAN rather than stopping at a word
        // that happens to be a speech token in another play — in Macbeth's own cast
        // `KING` is not one.
        if let macbeth = corpus.play("macbeth") {
            log.equal(
                Cast(play: macbeth).resolve("King Duncan"), "DUNCAN",
                "a two-word name in a direction")
        }

        // The same link in Romeo and Juliet: the Prince speaks as `PRINCE.` and
        // Dramatis Personæ files him as `ESCALUS, Prince of Verona.` Friar Lawrence
        // needs no alias, because `Cast.resolve` falls back to the speaking tokens,
        // and this is what proves it.
        if let romeo = corpus.play("romeo-and-juliet") {
            let cast = Cast(play: romeo)
            log.equal(cast.label("PRINCE"), "Prince (Escalus)", "the PRINCE label")
            log.equal(cast.display("PRINCE"), "Prince", "the PRINCE display name")
            log.equal(
                cast.resolve("Friar Lawrence"), "FRIAR LAWRENCE",
                "an unaliased two-word name in a direction")
        }

        // A chorus block is addressable through `SceneKey` like any other scene, and
        // the tracker reads its opening direction: scene 0 is not a special case
        // anywhere but in `SceneLabel`.
        log.equal(
            present("romeo-and-juliet", 1, 0, atLine: 1), ["Chorus"],
            "the Prologue, where Chorus enters and speaks")
    }

    // MARK: - Follow-up parsing

    private static func followUpParsing(_ log: Log) {
        let plain = """
            1. What does "quietus" mean here?
            2. Why does Hamlet say this now?
            3. Who is listening?
            4. How should this be staged?
            """
        log.equal(Prompts.FollowUps.parse(plain).count, 4, "a clean numbered list")

        log.equal(
            Prompts.FollowUps.parse("1) Why the first one?\n2) Why the second one?")
                .count, 2, "the `1)` style")

        log.equal(
            Prompts.FollowUps.parse("1. **What does the cold do**\n2. Another").first,
            "What does the cold do?", "bold markers stripped, question mark added")

        // A statement is not a question. Muse-Glimmer answered the follow-up request
        // with sentences from its own commentary, and appending a bare "?" produced
        // rows like "It establishes a wary, military tone at the castle gate.?"
        log.equal(
            Prompts.FollowUps.parse(
                "1. It establishes a wary, military tone at the castle gate.\n"
                    + "2. Why does Francisco demand a name"),
            ["Why does Francisco demand a name?"],
            "a declarative item rejected, an interrogative one completed")
        log.equal(
            Prompts.FollowUps.parse("1. How does the cold set the mood.").first,
            "How does the cold set the mood?",
            "a trailing stop replaced rather than doubled")

        let withPreamble = """
            Here are four questions a reader might ask:
            1. First question?
            2. Second question?
            3. Third question?
            4. Fourth question?
            """
        log.equal(
            Prompts.FollowUps.parse(withPreamble).count, 4,
            "a preamble line ignored")

        log.equal(
            Prompts.FollowUps.parse("1. One?\n2. Two?\n3. Three?").count, 3,
            "a three-item list")
        log.equal(Prompts.FollowUps.parse("1. Only one?").count, 1, "a one-item list")

        log.equal(
            Prompts.FollowUps.parse("1. Same question?\n2. same question?\n3. Other?")
                .count, 2, "duplicates collapsed")

        log.equal(
            Prompts.FollowUps.parse(
                "1. First?\n2. Second?\n3. Third?\n4. Fourth?\n5. Fifth?"
            ).count, 4, "capped at four")

        log.equal(
            Prompts.FollowUps.parse(
                "1. Already asked?\n2. Fresh one?",
                asked: [Prompts.FollowUps.normalized("Already asked?")]
            ), ["Fresh one?"], "dedupe against questions already asked")

        // Too short, and too long to be a question rather than a paragraph.
        log.equal(Prompts.FollowUps.parse("1. ok").count, 0, "a two-character question")
        log.equal(
            Prompts.FollowUps.parse("1. \(String(repeating: "long ", count: 40))").count,
            0, "a paragraph masquerading as a question")
    }

    // MARK: - Golden prompt render

    /// One assembled prompt compared against a checked-in string.
    ///
    /// This is what catches prompt drift: any change to the labelled blocks, the
    /// speaker grouping, or the window sizes shows up here as a diff, and the fix
    /// is to regenerate this string *deliberately*, alongside a `Prompts.version`
    /// bump.
    private static func goldenPromptRender(_ log: Log) {
        guard let corpus = try? CorpusLoader.load(),
            let play = corpus.play("hamlet"),
            case let key = SceneKey(playID: "hamlet", act: 1, scene: 5),
            let scene = corpus.scene(key)
        else {
            log.fail("could not load Hamlet I.v for the golden render")
            return
        }

        // The Ghost's "Remember me", which exercises every block: a setting, an
        // opening direction, an on-stage scan across an Exeunt, personae blurbs,
        // and a speech that runs across a direction.
        guard
            let start = scene.lines.firstIndex(where: {
                $0.text.hasPrefix("Adieu, adieu, adieu. Remember me.")
            })
        else {
            log.fail("could not find the Ghost's exit line in Hamlet I.v")
            return
        }

        let context = PassageContext.build(
            play: play, key: key, scene: scene,
            selection: LineSelection(at: start), cast: Cast(play: play))
        guard let context else {
            log.fail("the golden context did not build")
            return
        }

        let rendered = Prompts.annotationRequest(context)
        if rendered != Self.goldenPrompt {
            log.fail(
                """
                the golden prompt render drifted. If the change was intended, bump \
                Prompts.version and replace SelfTest.goldenPrompt with:
                ----- begin -----
                \(rendered)
                ----- end -----
                """)
        }

        log.equal(context.citation, "Hamlet · I.v.96 (this edition)", "the citation")
        log.equal(context.digest.count, 64, "the digest length")
    }

    /// Regenerated deliberately, alongside a `Prompts.version` bump.
    private static let goldenPrompt = """
        PLAY: Hamlet, Prince of Denmark by William Shakespeare

        LOCATION: Act I, Scene V

        SETTING: A more remote part of the Castle.

        SCENE OPENS: Enter Ghost and Hamlet.

        ON STAGE (approximate): Ghost, Hamlet

        WHO THEY ARE:
        - Ghost: of the late king, Hamlet’s father
        - Hamlet: Prince of Denmark

        BEFORE (lines 81-95):
        GHOST:
          Cut off even in the blossoms of my sin,
          Unhous’led, disappointed, unanel’d;
          No reckoning made, but sent to my account
          With all my imperfections on my head.
          O horrible! O horrible! most horrible!
          If thou hast nature in thee, bear it not;
          Let not the royal bed of Denmark be
          A couch for luxury and damned incest.
          But howsoever thou pursu’st this act,
          Taint not thy mind, nor let thy soul contrive
          Against thy mother aught; leave her to heaven,
          And to those thorns that in her bosom lodge,
          To prick and sting her. Fare thee well at once!
          The glow-worm shows the matin to be near,
          And ’gins to pale his uneffectual fire.

        SELECTED PASSAGE (line 96):
        GHOST:
          Adieu, adieu, adieu. Remember me.

        AFTER (lines 97-99):
        [Exit.]
        HAMLET:
          O all you host of heaven! O earth! What else?
          And shall I couple hell? O, fie! Hold, my heart;
          And you, my sinews, grow not instant old,

        Annotate the selected passage.
        """

    // MARK: - Reader fonts

    /// The typeface picker's inputs. Model-free and network-free.
    ///
    /// Nothing here may touch `ReaderFontLibrary`, which is `@MainActor` while
    /// `run()` is synchronous and non-isolated — which is exactly why
    /// `installedFamilyNames()` is a `static` on `ReaderFont` that the library merely
    /// calls.
    ///
    /// Deliberately does not construct a `Font`: it would compile and assert nothing,
    /// because a SwiftUI `Font` cannot be measured. The inputs are what is worth
    /// asserting.
    /// Named once so the bound and the failure message cannot drift apart. Wide
    /// enough for any real optical correction and narrow enough that a fat-fingered
    /// `11.5` cannot ship.
    private static let plausibleOpticalScales: ClosedRange<CGFloat> = 0.9 ... 1.3

    private static func readerFonts(_ log: Log) {
        for font in ReaderFont.allCases {
            // Literally the `@AppStorage("readerFont")` contract: a raw value that
            // stops round-tripping silently resets every reader to the system face.
            log.check(
                ReaderFont(rawValue: font.rawValue) == font,
                "ReaderFont.\(font) does not round-trip through its raw value")
            log.check(
                plausibleOpticalScales.contains(font.opticalScale),
                "ReaderFont.\(font) opticalScale \(font.opticalScale) is outside "
                    + "\(plausibleOpticalScales), which is not an optical correction")
        }

        log.equal(
            Set(ReaderFont.allCases.map(\.displayName)).count,
            ReaderFont.allCases.count, "the number of distinct display names")
        log.check(
            ReaderFont.system.familyName == nil,
            "the system face should not name a family")
        for font in ReaderFont.allCases where font != .system {
            log.check(font.familyName != nil, "ReaderFont.\(font) names no family")
        }

        // The only nontrivial logic in the feature, and pure CoreText: Baskerville
        // has a real italic cut, Big Caslon is a single face and has none — which is
        // what `ReaderTypeface.direction` shears by hand.
        log.check(
            ReaderFont.baskerville.hasItalicFace,
            "Baskerville reported no italic face, so stage directions lost their cut")
        log.check(
            !ReaderFont.caslon.hasItalicFace,
            "Big Caslon reported an italic face, so the synthetic oblique is dead code")

        // Machine-dependent, and kept anyway: a typo like "BigCaslon" is otherwise
        // completely silent — the reader would just get the system face forever.
        // Deliberately not Garamond, which is an on-demand Apple asset and legitimately
        // absent until someone picks it.
        let installed = ReaderFont.installedFamilyNames()
        for font in [ReaderFont.caslon, .baskerville] {
            guard let family = font.familyName else { continue }
            log.check(
                installed.contains(family),
                "\"\(family)\" is not among the installed font families, so "
                    + "ReaderFont.\(font) would silently render as the system face")
        }
    }

    // MARK: - Reading progress

    /// The two pure seams of the restore: the record's codec, and resolving a record
    /// against the corpus.
    ///
    /// Nothing here touches `ProgressStore`, so `--selftest` cannot read or overwrite
    /// the real reader's position.
    private static func readingProgress(_ log: Log) {
        guard let corpus = try? CorpusLoader.load(),
            let firstScene = corpus.firstScene,
            let macbeth = corpus.play("macbeth")
        else {
            log.fail("could not load the corpus for the reading-progress checks")
            return
        }

        let key = SceneKey(playID: "macbeth", act: 1, scene: 3)
        guard let scene = corpus.scene(key) else {
            log.fail("Macbeth I.iii not found")
            return
        }
        let stamp = macbeth.source.textSHA256

        /// A record's exact round trip through JSON, which is the storage format.
        func roundTrip(_ record: ReadingProgress, _ label: String) {
            guard let data = try? JSONEncoder().encode(record),
                let back = try? JSONDecoder().decode(ReadingProgress.self, from: data)
            else {
                log.fail("\(label) did not round-trip through JSON")
                return
            }
            log.equal(back, record, label)
        }

        let selection = LineSelection(anchor: 12, head: 20)
        let record = ReadingProgress(
            schemaVersion: ProgressStore.schemaVersion, key: key,
            selection: selection, corpusStamp: stamp)
        roundTrip(record, "a record carrying a selection")
        roundTrip(
            ReadingProgress(
                schemaVersion: ProgressStore.schemaVersion, key: key, selection: nil,
                corpusStamp: stamp),
            "a record with no selection")

        // `LineSelection`'s own conformance, which the record's synthesis rests on.
        if let data = try? JSONEncoder().encode(selection),
            let back = try? JSONDecoder().decode(LineSelection.self, from: data)
        {
            log.equal(back, selection, "a LineSelection round trip")
        } else {
            log.fail("LineSelection did not round-trip through JSON")
        }

        // A cold start: the first scene, nothing selected.
        let cold = corpus.opening(from: nil)
        log.equal(cold.key, firstScene, "the opening with no stored record")
        log.check(cold.selection == nil, "a cold start should select nothing")

        // The ordinary case.
        let restored = corpus.opening(from: record)
        log.equal(restored.key, key, "the restored scene")
        log.equal(restored.selection, selection, "the restored selection")

        // A play that is no longer in the corpus, and an act it never had: both would
        // leave `ContentView` on its loading spinner forever if handed through.
        //
        // The id is deliberately synthetic. This check used to name Coriolanus, which
        // was a play the corpus did not have until it was one — at which point the
        // record resolved, the fallback never fired, and the assertion failed. No real
        // play's slug can be safely used here.
        var removedPlay = record
        removedPlay.key = SceneKey(playID: "a-play-no-longer-here", act: 1, scene: 1)
        log.equal(
            corpus.opening(from: removedPlay).key, firstScene,
            "the opening for a play the corpus no longer has")
        var missingAct = record
        missingAct.key = SceneKey(playID: "macbeth", act: 99, scene: 1)
        log.equal(
            corpus.opening(from: missingAct).key, firstScene,
            "the opening for an act the corpus does not have")

        // A rebuilt corpus keeps the scene and drops the highlight, rather than
        // putting it over whatever lines those indices now name.
        var drifted = record
        drifted.corpusStamp = "not the digest of anything"
        let afterDrift = corpus.opening(from: drifted)
        log.equal(afterDrift.key, key, "the scene after the corpus text changed")
        log.check(
            afterDrift.selection == nil,
            "a selection should be dropped when the corpus stamp does not match")

        // A record written against a longer scene.
        var past = record
        past.selection = LineSelection(anchor: scene.lines.count + 10, head: 99_999)
        log.equal(
            corpus.opening(from: past).selection,
            LineSelection(at: scene.lines.count - 1),
            "a selection past the end of the scene")
    }
}

/// `--show-prompt`: renders the assembled prompt and its exact token count for a
/// sample of passages, then exits.
///
/// The token count is the real one, from
/// `tokenizer.applyChatTemplate(messages:tools:additionalContext:)`, which is why
/// this loads the model. Early modern verse runs about 1.4 Qwen3 tokens per word —
/// elisions (`o’er`, `on’t`) and curly apostrophes split more than modern prose —
/// so a words-per-token estimate is not good enough to budget with.
@MainActor
enum PromptDump {

    static func run(options: AppOptions) async -> Bool {
        guard let corpus = try? CorpusLoader.load() else {
            print("could not load the corpus")
            return false
        }

        let service = AnnotationService(
            modelID: options.modelID ?? LLMRegistry.qwen3_4b_4bit.name,
            greedy: options.greedy)
        await service.load()
        guard service.isReady else {
            print("the model did not load; cannot report exact token counts")
            return false
        }

        let specs = options.passages.isEmpty ? SamplePassages.specs : options.passages
        var counted: [(String, Int)] = []

        for spec in specs {
            guard let parsed = PassageSpec(spec),
                let play = corpus.play(parsed.key.playID),
                let scene = corpus.scene(parsed.key),
                let selection = parsed.selection(in: scene),
                let context = PassageContext.build(
                    play: play, key: parsed.key, scene: scene, selection: selection,
                    cast: Cast(play: play),
                    synopsis: service.synopsis(for: parsed.key)?.text)
            else {
                print("could not resolve \(spec)")
                return false
            }

            let request = Prompts.annotationRequest(context)
            let tokens = await service.promptTokenCount(for: request)
            counted.append((context.citation, tokens))

            print(String(repeating: "=", count: 78))
            print("\(spec) \u{2192} \(context.citation) — \(tokens) prompt tokens")
            print(String(repeating: "-", count: 78))
            print(request)
            print()
        }

        print(String(repeating: "=", count: 78))
        print("prompt token counts (system instructions and chat template included)")
        for (citation, tokens) in counted {
            print(String(format: "  %5d  %@", tokens, citation))
        }
        let counts = counted.map(\.1)
        if let low = counts.min(), let high = counts.max(), !counts.isEmpty {
            print("  min \(low) · max \(high) · mean \(counts.reduce(0, +) / counts.count)")
        }
        return true
    }
}
