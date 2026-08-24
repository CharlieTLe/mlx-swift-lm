import Foundation
import MLXLLM
import MLXLMCommon
import SwiftUI

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
        readerTextSizes(log)
        readingProgress(log)
        navigator(log)
        wordTokenizer(log)

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
        guard
            let start = speech.firstIndex(where: {
                $0.text.hasPrefix("To be, or not to be")
            })
        else {
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
        log.equal(LineSelection(anchor: 4, head: 1).range, 1 ... 4, "a backwards range")
        log.equal(LineSelection(anchor: 1, head: 4).range, 1 ... 4, "a forwards range")
        log.equal(LineSelection(at: 2).count, 1, "a single-line selection")

        // Shift-click backwards through the anchor keeps the anchor.
        var backwards = LineSelection(at: 4)
        backwards.extend(to: 1)
        log.equal(backwards.anchor, 4, "the anchor after extending backwards")
        log.equal(backwards.range, 1 ... 4, "the range after extending backwards")

        // Drag reversal: past the anchor and back again.
        var reversed = LineSelection(at: 2)
        reversed.extend(to: 5)
        reversed.extend(to: 1)
        log.equal(reversed.range, 1 ... 2, "the range after a drag reverses")

        // Double-click takes the whole speech, across the interleaved direction.
        log.equal(
            LineSelection.speech(at: 2, in: scene).range, 1 ... 4,
            "a double-clicked speech spanning a direction")
        log.equal(
            LineSelection.speech(at: 3, in: scene).range, 1 ... 4,
            "a double-click on a direction inside a speech")
        // A direction between two different speakers stands alone.
        log.equal(
            LineSelection.speech(at: 6, in: scene).range, 6 ... 6,
            "a double-clicked trailing direction")
        log.equal(
            LineSelection.speech(at: 0, in: scene).range, 0 ... 0,
            "a double-clicked opening direction")
        log.equal(
            LineSelection.speech(at: 5, in: scene).range, 5 ... 5,
            "a double-clicked one-line speech")

        // Clamping at the scene edges.
        log.equal(
            LineSelection(anchor: -4, head: 99).clamped(to: lines)?.range, 0 ... 6,
            "clamping past both edges")
        log.check(
            LineSelection(at: 0).clamped(to: [])?.range == nil,
            "clamping into an empty scene should yield nil")
        log.equal(
            LineSelection.speech(at: 42, in: scene).range, 42 ... 42,
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

        // MARK: Passage identity

        // What `ContentView.commit` tests before it decides a tap is a re-tap of the
        // passage already in the pane rather than a new request.
        func built(_ selection: LineSelection, synopsis: String? = nil) -> PassageContext? {
            PassageContext.build(
                play: play, key: key, scene: scene, selection: selection,
                cast: Cast(play: play), synopsis: synopsis,
                synopsisIsPartial: synopsis != nil)
        }

        guard let again = built(LineSelection(at: start)),
            let neighbour = built(LineSelection(at: start + 1)),
            let speech = built(LineSelection.speech(at: start, in: scene)),
            let summarized = built(LineSelection(at: start), synopsis: "The Ghost departs.")
        else {
            log.fail("a passage-identity context did not build")
            return
        }

        log.check(context.isSamePassage(as: again), "the same line rebuilt")
        log.check(!context.isSamePassage(as: neighbour), "the next line along")
        log.check(
            !context.isSamePassage(as: speech),
            "the whole speech, starting on the same line")
        // The case the comment on `isSamePassage` is defending: the synopsis arrives in
        // the background, and a `==` here would call the passage new when it did.
        log.check(context.isSamePassage(as: summarized), "the same line, now with a synopsis")
        log.check(context != summarized, "a synopsis is still a difference under ==")
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

    // MARK: - Reader text size

    /// The same bound as `plausibleOpticalScales`, and for the same reason: named once
    /// so the range and the failure message cannot drift apart. Wider, because this one
    /// is a reader's preference rather than a correction, and still narrow enough that a
    /// mistyped `15` cannot ship.
    private static let plausibleTextSizeMultipliers: ClosedRange<CGFloat> = 0.7 ... 1.7

    /// The size ladder's inputs, and the arithmetic `ReaderTypeface` does with them.
    ///
    /// Same philosophy as `readerFonts`: no `Font` is constructed, because a SwiftUI
    /// `Font` cannot be measured. Everything asserted here is a number or a raw value.
    /// `installed: []` also short-circuits `hasItalicFace`, so nothing here reaches
    /// CoreText or the `@MainActor` `ReaderFontLibrary`.
    private static func readerTextSizes(_ log: Log) {
        /// The system face at a size step, which is every `ReaderTypeface` this section
        /// needs. `installed: []` keeps it away from CoreText, and `.large` is the only
        /// category macOS has.
        func typeface(
            _ size: ReaderTextSize, at category: DynamicTypeSize = .large
        ) -> ReaderTypeface {
            ReaderTypeface(
                .system, textSize: size, dynamicTypeSize: category, installed: [])
        }

        for size in ReaderTextSize.allCases {
            // The `@AppStorage("readerTextSize")` contract, exactly as `readerFonts`
            // asserts it for the face: a raw value that stops round-tripping silently
            // resets every reader to Default.
            log.check(
                ReaderTextSize(rawValue: size.rawValue) == size,
                "ReaderTextSize.\(size) does not round-trip through its raw value")
            log.check(
                plausibleTextSizeMultipliers.contains(size.multiplier),
                "ReaderTextSize.\(size) multiplier \(size.multiplier) is outside "
                    + "\(plausibleTextSizeMultipliers), which is not a reading size")
            // `isDefault` is what every role branches on to keep the shipped rendering
            // byte-for-byte, so it has to mean "changes nothing" and not merely
            // "is the case spelled default".
            log.equal(
                size.isDefault, size.multiplier == 1,
                "ReaderTextSize.\(size).isDefault against a multiplier of exactly 1")
        }

        log.equal(ReaderTextSize.default.multiplier, 1, "the default multiplier")
        log.equal(
            ReaderTextSize.allCases.filter(\.isDefault).count, 1,
            "the number of neutral size steps")
        log.equal(
            Set(ReaderTextSize.allCases.map(\.displayName)).count,
            ReaderTextSize.allCases.count, "the number of distinct size display names")

        // Declaration order is menu order, so the multipliers have to rise along it or
        // the menu lists "Larger" above something larger still.
        for (smaller, bigger) in zip(
            ReaderTextSize.allCases, ReaderTextSize.allCases.dropFirst())
        {
            log.check(
                smaller.multiplier < bigger.multiplier,
                "ReaderTextSize.\(smaller) (\(smaller.multiplier)) does not sort below "
                    + "\(bigger) (\(bigger.multiplier)) in `allCases` order")
        }

        // The byte-for-byte promise, in the units that can actually be read back off a
        // `ReaderTypeface`. Every one of these is a multiplication by exactly 1.0.
        let shipped = ReaderTypeface.system
        log.equal(shipped.textSize, .default, "the shipped typeface's size step")
        log.equal(shipped.speechGap, 6, "the shipped speech gap")
        log.equal(shipped.directionIndent, 28, "the shipped stage-direction indent")
        log.equal(shipped.gutterWidth, 30, "the shipped gutter width")
        log.equal(shipped.actSceneTracking, 0, "the shipped act-heading tracking")
        log.equal(shipped.speakerTracking, 0.6, "the shipped speaker tracking")

        // The same promise for the fonts, and the one place in this file that does
        // construct a `Font`. `readerFonts` above declines to, on the grounds that a
        // SwiftUI `Font` cannot be *measured* — but it can be compared, and comparing is
        // what is wanted here: these assert that Default returns the very same values
        // the app returned before there was a size setting at all, rather than a
        // computed `Font.system(size:)` that merely resolves to the same points. The two
        // are not interchangeable, since only the text style follows Dynamic Type.
        log.equal(shipped.verse, .body, "the shipped verse font")
        log.equal(shipped.actSceneHeading, .headline, "the shipped act-heading font")
        log.equal(shipped.sceneSetting, .subheadline, "the shipped scene-setting font")
        log.equal(
            shipped.speakerHeading, .caption.weight(.semibold),
            "the shipped speaker-heading font")
        log.equal(
            shipped.direction, .callout.italic(), "the shipped stage-direction font")
        log.equal(
            shipped.gutterFont, .caption2.monospacedDigit(), "the shipped gutter font")

        // And the other direction, which is what would catch the whole feature quietly
        // becoming a no-op for the system face: a non-default step must *not* return the
        // bare style.
        log.check(
            typeface(.large).verse != .body,
            "the system face at the Large step still returns `Font.body`, so choosing a "
                + "size does nothing")

        // End-to-end through `size(_:)`'s rounding: a ladder whose steps round to the
        // same point size is still a ladder, one that goes *down* somewhere is not.
        for (smaller, bigger) in zip(
            ReaderTextSize.allCases, ReaderTextSize.allCases.dropFirst())
        {
            let low = typeface(smaller)
            let high = typeface(bigger)
            log.check(
                low.speechGap <= high.speechGap,
                "the speech gap falls from \(smaller) (\(low.speechGap)) to "
                    + "\(bigger) (\(high.speechGap))")
        }

        // What makes `SceneReaderView`'s `.onChange(of: typeface)` re-anchor the scroll
        // position when only the size changed — and what would catch someone splitting
        // the size back out into its own environment key, which would not fire it.
        log.check(
            typeface(.default) != typeface(.largest),
            "two size steps of the same face compare equal, so a size change would not "
                + "re-anchor the reader's scroll position")
        log.check(
            typeface(.large) == typeface(.large),
            "the same face and size compare unequal, so every render would re-anchor")

        // The same contract for the Dynamic Type category, which is stored on the
        // typeface for the same reason and carries more weight: at a non-default step
        // the system face is a *fixed*-size `Font.system(size:)`, so if a category
        // change did not change this value the reader's Larger Text slider would stop
        // doing anything at all.
        log.check(
            typeface(.large, at: .large) != typeface(.large, at: .accessibility5),
            "two Dynamic Type categories compare equal, so the play text would not "
                + "follow the reader's Larger Text setting")

        // The two multipliers meet in `ReaderTypeface.scale`, and it is their *product*
        // that sets the type: a per-family correction that is fine on its own can still
        // land somewhere absurd at the top of the ladder.
        let plausibleProducts: ClosedRange<CGFloat> = 0.7 ... 2.0
        for font in ReaderFont.allCases {
            for size in ReaderTextSize.allCases {
                let product = font.opticalScale * size.multiplier
                log.check(
                    plausibleProducts.contains(product),
                    "\(font) at \(size) scales the type by \(product), outside "
                        + "\(plausibleProducts)")
            }
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

    // MARK: - Navigator

    /// The accordion and the find field, which are pure functions of a `SceneKey` and
    /// of the corpus respectively — which is the whole reason they are not written
    /// inside `NavigatorView`. This is the first coverage the navigator has had.
    private static func navigator(_ log: Log) {
        guard let corpus = try? CorpusLoader.load() else {
            log.fail("could not load the corpus for the navigator checks")
            return
        }

        // MARK: The outline

        // What a launch gets: the restored position, and nothing else open.
        let outline = NavigatorOutline.following(
            SceneKey(playID: "macbeth", act: 2, scene: 1))
        log.check(outline.isOpen(play: "macbeth"), "the read play should be open")
        log.check(
            outline.isOpen(act: 2, in: "macbeth"), "the read act should be open")
        log.check(!outline.isOpen(play: "hamlet"), "another play should be shut")
        log.check(
            !outline.isOpen(act: 1, in: "macbeth"),
            "another act of the open play should be shut")
        // The invariant: an open act belongs to the open play.
        log.check(
            !outline.isOpen(act: 2, in: "hamlet"),
            "the open act number should not read as open in another play")

        // Tapping the open play closes it, and takes its act with it.
        var closed = outline
        closed.toggle(play: "macbeth")
        log.check(!closed.isOpen(play: "macbeth"), "the tapped-open play should close")
        log.check(
            !closed.isOpen(act: 2, in: "macbeth"),
            "closing a play should drop its open act")

        // Tapping another play moves the accordion, with that play's acts shut: five
        // act rows to choose between, not 28 scene rows.
        var moved = outline
        moved.toggle(play: "hamlet")
        log.check(moved.isOpen(play: "hamlet"), "the tapped play should open")
        log.check(!moved.isOpen(play: "macbeth"), "the play left behind should close")
        log.check(
            !moved.isOpen(act: 1, in: "hamlet"),
            "a newly opened play should start with its acts shut")

        // An act in another play opens both.
        var reached = outline
        reached.toggle(act: 3, in: "hamlet")
        log.check(reached.isOpen(play: "hamlet"), "an act should open its play with it")
        log.check(reached.isOpen(act: 3, in: "hamlet"), "the tapped act should open")
        log.check(!reached.isOpen(play: "macbeth"), "the play left behind should close")

        // Tapping the open act closes it and leaves the play open.
        reached.toggle(act: 3, in: "hamlet")
        log.check(!reached.isOpen(act: 3, in: "hamlet"), "the tapped-open act should close")
        log.check(
            reached.isOpen(play: "hamlet"), "closing an act should leave its play open")

        // MARK: The find field

        /// The play ids a query lists, in corpus order.
        func ids(_ query: String) -> [String] {
            NavigatorSearch.matches(in: corpus, query: query).map(\.play.id)
        }

        // No query is the whole corpus, whole: the outline alone decides what shows.
        let all = NavigatorSearch.matches(in: corpus, query: "  ")
        log.equal(all.count, 35, "the number of plays with no query")
        for match in all {
            log.check(
                match.matchedTitle, "\(match.play.id) should match by title with no query")
            log.equal(match.acts.count, 5, "\(match.play.id) acts with no query")
            log.equal(
                match.acts.map(\.scenes.count), match.play.acts.map(\.scenes.count),
                "\(match.play.id) scene counts with no query")
        }

        // A title match: the play alone, whole, with its acts left for the reader.
        log.equal(ids("macb"), ["macbeth"], "the plays matching \"macb\"")
        log.check(
            NavigatorSearch.matches(in: corpus, query: "macb").first?.matchedTitle == true,
            "\"macb\" should match Macbeth by title")
        // Folding: mixed-case titles, and the apostrophe the reader will not type.
        log.equal(ids("henry iv"), ["henry-iv-part-1", "henry-iv-part-2"], "\"henry iv\"")
        log.equal(ids("loves labours"), ["loves-labours-lost"], "\"loves labours\"")
        log.equal(ids("midsummer"), ["midsummer-nights-dream"], "\"midsummer\"")

        // A setting match: only the acts that hold hits, carrying only those scenes.
        let churchyard = NavigatorSearch.matches(in: corpus, query: "CHURCHYARD")
        log.equal(
            churchyard.map(\.play.id), ["hamlet", "romeo-and-juliet"],
            "the plays matching \"churchyard\"")
        for match in churchyard {
            log.check(
                !match.matchedTitle,
                "\(match.play.id) matched a setting, not a title")
            for actMatch in match.acts {
                log.check(
                    !actMatch.scenes.isEmpty,
                    "\(match.play.id) act \(actMatch.act.number) came back with no scenes")
                log.check(
                    actMatch.scenes.allSatisfy {
                        $0.setting.lowercased().contains("churchyard")
                    },
                    "\(match.play.id) act \(actMatch.act.number) carried a scene that "
                        + "does not match")
            }
        }
        // The grave-diggers, which is the passage this is for: Hamlet, Act V, Scene I.
        log.equal(
            churchyard.first?.acts.map(\.act.number), [5],
            "the acts of Hamlet matching \"churchyard\"")
        log.equal(
            churchyard.first?.acts.first?.scenes.map(\.number), [1],
            "the scenes of Hamlet V matching \"churchyard\"")

        // Nothing at all, which the view renders as its placeholder.
        log.equal(ids("zzz"), [], "the plays matching \"zzz\"")
    }

    // MARK: - Word lookup

    /// Where one word of the verse begins and ends.
    ///
    /// The only pure seam of word lookup, and so the only one this file can reach:
    /// `WordHitTest` needs a resolved face and a wrap width and answers with a point on a
    /// line nobody drew, and whether it agrees with what SwiftUI rendered is a thing that
    /// has to be looked at rather than asserted. That is what the hover mark is for. This
    /// asserts the half that is arithmetic.
    private static func wordTokenizer(_ log: Log) {
        /// Every word of `text`, as strings.
        func words(_ text: String) -> [String] {
            WordTokenizer.words(in: text).map { String(text[$0]) }
        }

        /// The invariant that makes a hover mark trustworthy: the ranges tile the string.
        /// Ordered, non-overlapping, non-empty, and everything they leave out is
        /// punctuation or space — a gap with a letter in it is a word the reader can point
        /// at and be told nothing about.
        func tiles(_ text: String, _ label: String) {
            let ranges = WordTokenizer.words(in: text)
            var cursor = text.startIndex
            for range in ranges {
                log.check(
                    !range.isEmpty, "\(label): an empty word range in \"\(text)\"")
                log.check(
                    range.lowerBound >= cursor,
                    "\(label): word ranges overlap or run backwards in \"\(text)\"")
                for character in text[cursor ..< range.lowerBound] {
                    log.check(
                        !character.isLetter && !character.isNumber,
                        "\(label): \"\(character)\" in \"\(text)\" is in no word")
                }
                cursor = range.upperBound
            }
            for character in text[cursor...] {
                log.check(
                    !character.isLetter && !character.isNumber,
                    "\(label): trailing \"\(character)\" in \"\(text)\" is in no word")
            }

            // And what the pointer will actually ask: every character of a word resolves
            // back to that same word.
            for range in ranges {
                for index in text[range].indices {
                    log.equal(
                        WordTokenizer.word(at: index, in: text), range,
                        "\(label): the word at \"\(text[index])\" in \"\(text)\"")
                }
            }
        }

        // The ten words of the most-read line in English, which is also the shape that
        // matters most: the commas and the closing colon are not part of any word.
        log.equal(
            words("To be, or not to be, that is the question:"),
            ["To", "be", "or", "not", "to", "be", "that", "is", "the", "question"],
            "the words of \"To be, or not to be\"")

        // Elisions stay whole, in both apostrophes: the corpus is typeset with the curly
        // one and a reader's own typing is not. `’tis` is the one `.byWords` gets wrong on
        // its own, handing back `tis` and leaving the apostrophe in no word at all.
        log.equal(
            words("The undiscover’d country"), ["The", "undiscover’d", "country"],
            "an elided participle")
        log.equal(words("o’er the"), ["o’er", "the"], "\"o’er\"")
        log.equal(words("o'er the"), ["o'er", "the"], "\"o’er\" with a straight apostrophe")
        log.equal(words("on’t"), ["on’t"], "\"on’t\"")
        log.equal(words("’tis so"), ["’tis", "so"], "a leading elision")
        log.equal(words("Who’s there?"), ["Who’s", "there"], "\"Who’s there?\"")
        log.equal(words("Hamlet’s father"), ["Hamlet’s", "father"], "a possessive")

        // Compounds stay whole, which `.byWords` also gets wrong on its own: it splits at
        // every hyphen.
        log.equal(words("well-a-day"), ["well-a-day"], "\"well-a-day\"")
        log.equal(words("to-morrow and"), ["to-morrow", "and"], "\"to-morrow\"")
        // And the reason a joiner merges across one character only: the transcription
        // writes a dash as two hyphens, and a run of them would fuse two words into a
        // portmanteau the dictionary has never heard of.
        log.equal(words("death,--and"), ["death", "and"], "a double-hyphen dash")

        // What the dictionary and the model are handed: the word, without the sentence
        // leaning on it. The apostrophe and the hyphen *inside* a word are the word.
        func term(_ text: String) -> String {
            guard let range = WordTokenizer.words(in: text).first else { return "" }
            return WordTokenizer.term(for: range, in: text)
        }
        log.equal(term("there?"), "there", "the term of \"there?\"")
        log.equal(term("death,"), "death", "the term of \"death,\"")
        log.equal(term("o’er"), "o’er", "the term of \"o’er\"")
        log.equal(term("to-morrow"), "to-morrow", "the term of \"to-morrow\"")

        // Nothing is a word on a space, and nothing is offered there.
        let line = "To be, or not to be"
        guard let space = line.firstIndex(of: " ") else {
            log.fail("no space in a string with two of them")
            return
        }
        log.check(
            WordTokenizer.word(at: space, in: line) == nil,
            "a space resolved to a word")
        log.check(
            WordTokenizer.word(at: line.startIndex, in: line) != nil,
            "the first letter of a line resolved to no word")

        tiles("", "an empty line")
        tiles("[_Exeunt._]", "a direction's markup")
        for text in ["To be, or not to be, that is the question:", "’tis well-a-day, o’er"] {
            tiles(text, "a hand-written line")
        }

        // And against the real thing, because the invariant is about punctuation the
        // corpus has and hand-written strings do not. One play rather than all 35: the
        // tokenizer knows nothing about which play it is reading, and Hamlet is 4,000
        // lines of the punctuation in question.
        guard let corpus = try? CorpusLoader.load(), let play = corpus.play("hamlet") else {
            log.fail("could not load Hamlet for the word-tokenizer checks")
            return
        }
        for act in play.acts {
            for scene in act.scenes {
                for line in scene.lines {
                    tiles(line.plainText, "\(play.id) \(act.number).\(scene.number)")
                }
            }
        }
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
