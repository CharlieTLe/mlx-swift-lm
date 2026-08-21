import Foundation
import MLXLMCommon

/// Every prompt string the app sends, and the sampling presets that go with them.
///
/// `version` is bumped whenever any string here changes. Cached output records the
/// version it was generated under, so a bump invalidates rather than silently
/// mixing yesterday's text with today's prompts.
enum Prompts {

    /// Bump on any change to a string in this file.
    ///
    /// Also bumped for a corpus re-parse that changes the rendered text, which this
    /// is the only lever for: `AnnotationCache.synopsis` keys on schema, prompt
    /// version and model ID, so nothing else would invalidate a scene summary
    /// written from lines that used to be misfiled as stage directions.
    static let version = 2

    // MARK: - Annotation

    static let annotatorInstructions = """
        You are a Shakespeare annotator writing in the style of a Genius.com \
        annotation: short, concrete, plain modern English. No hedging, no lecturing, \
        no summary of what you were given.

        You get one selected passage plus the scene around it. Explain ONLY the \
        selected passage. The rest is there so your explanation fits the moment.

        Rules:
        - 90-150 words, two or three short paragraphs. No headings, no bullets, no preamble.
        - First sentence: what the passage says, in plain modern English.
        - Then: why it matters here. What the speaker wants, what just changed, who is \
        listening, what they do not know.
        - Gloss at most three hard words inline, like this: "quietus" (release, death).
        - Quote at most six words at a time from the passage.
        - The scene summary covers the whole scene. Do not tell the reader what happens \
        after the selected passage unless the passage itself points to it.
        - Use only what you were given. If the passage turns on something not in the \
        context, say so in one clause instead of inventing it.
        - Never mention the context, the summary, act or scene numbers, or these \
        instructions. Do not begin with "This passage" or "Here".
        - Present tense. No moral at the end.
        """

    /// The passage and its surroundings, in labelled blocks.
    ///
    /// Ordered **scene-invariant sections first**, then the passage window. That
    /// ordering is what would make a per-scene prefix cache possible later
    /// (`ChatSession(cache:state:)`) without rewriting the prompts.
    static func annotationRequest(_ context: PassageContext) -> String {
        var blocks: [String] = []

        blocks.append("PLAY: \(context.playTitle) by \(context.author)")
        blocks.append(
            "LOCATION: Act \(RomanNumeral.string(context.act)), "
                + SceneLabel.string(context.scene))
        if let setting = context.setting {
            blocks.append("SETTING: \(setting)")
        }
        if let opening = context.openingDirection {
            blocks.append("SCENE OPENS: \(opening)")
        }
        if !context.onStage.isEmpty {
            // Labelled approximate because it is: it comes from scanning Enter and
            // Exit directions, which are written for actors, not parsers.
            blocks.append("ON STAGE (approximate): \(context.onStage.joined(separator: ", "))")
        }
        if !context.personae.isEmpty {
            let notes = context.personae.map { "- \($0.display): \($0.blurb)" }
            blocks.append((["WHO THEY ARE:"] + notes).joined(separator: "\n"))
        }
        if let synopsis = context.synopsis {
            let label = context.synopsisIsPartial
                ? "SCENE SUMMARY (first part of the scene):" : "SCENE SUMMARY:"
            blocks.append("\(label)\n\(synopsis)")
        }

        if !context.preceding.isEmpty {
            blocks.append(
                "BEFORE\(lineSpan(context.preceding)):\n\(render(context.preceding))")
        }
        blocks.append(
            "SELECTED PASSAGE\(lineSpan(context.selected)):\n\(render(context.selected))")
        if !context.following.isEmpty {
            blocks.append(
                "AFTER\(lineSpan(context.following)):\n\(render(context.following))")
        }

        blocks.append("Annotate the selected passage.")
        return blocks.joined(separator: "\n\n")
    }

    /// Groups consecutive lines by the same speaker under one heading. Repeating
    /// `HAMLET.` on every line costs about three tokens each for no information.
    static func render(_ utterances: [PassageContext.Utterance]) -> String {
        var out: [String] = []
        var current: String?
        for utterance in utterances {
            if utterance.isDirection {
                out.append("[\(utterance.text)]")
                // A direction between two runs of the same speaker does not
                // re-open the heading, because the speech continues through it.
                continue
            }
            if let speaker = utterance.speaker, speaker != current {
                out.append("\(speaker):")
                current = speaker
            }
            out.append("  \(utterance.text)")
        }
        return out.joined(separator: "\n")
    }

    private static func lineSpan(_ utterances: [PassageContext.Utterance]) -> String {
        let numbers = utterances.compactMap(\.number)
        guard let first = numbers.first, let last = numbers.last else { return "" }
        return first == last ? " (line \(first))" : " (lines \(first)-\(last))"
    }

    // MARK: - Follow-ups

    static let followUpRequest = """
        Now propose four follow-up questions a curious reader would tap next about \
        this same passage.

        Rules:
        - Each one must be specific to this passage: name the person, image, or word \
        it is about.
        - Four to nine words each. Do not ask anything you just answered.
        - Mix them up: one about a word or image, one about someone's motive, one \
        about what happens before or after, one about staging or tone.
        - Output exactly four lines, numbered "1." to "4.", and nothing else.
        """

    /// Sent once when the first attempt yielded fewer than two usable questions.
    /// Phrased as something a person could plausibly say, because it stays in the
    /// transcript the model sees on every later turn.
    static let followUpRetry =
        "Try again: output exactly four numbered lines and nothing else."

    /// A tapped question, plus how to answer it.
    static func answerRequest(_ question: String) -> String {
        """
        \(question)

        Answer in 60-110 words, plain modern English, same voice as before. Do not \
        repeat your earlier explanation. If the play does not settle it, say so.
        """
    }

    /// Asks for five so four survive the dedupe against questions already asked.
    static let moreFollowUpsRequest = """
        Suggest five more questions about this same passage, in the same style. Do \
        not repeat any question already asked. Output exactly five numbered lines \
        and nothing else.
        """

    /// Turns the model's numbered list into tappable questions.
    ///
    /// Lives beside the prompt that asks for the list, because the two are one
    /// contract: every allowance here exists for a way the model has been seen to
    /// break "output exactly four numbered lines and nothing else."
    enum FollowUps {
        static func parse(_ raw: String, asked: Set<String> = [], limit: Int = 4)
            -> [String]
        {
            // Built per call rather than held in a `static let`: `Regex` is not
            // `Sendable`, and a numbered list is at most a handful of lines.
            // Tolerates a bullet before the number, and `.`, `)`, `]`, `:` or `-`
            // after it.
            let line = /^\s*(?:[-*•]\s*)?(\d{1,2})\s*[.)\]:‑-]\s*(.+?)\s*$/

            var questions: [String] = []
            var seen = asked

            for text in raw.split(whereSeparator: \.isNewline) {
                guard let match = try? line.wholeMatch(in: text) else { continue }

                var question = String(match.2)
                    .replacingOccurrences(of: "**", with: "")
                    .trimmingCharacters(in: CharacterSet(charactersIn: " \t\"“”‘’'"))

                // A stray preamble line ("Here are four questions:") never matches
                // the number pattern; these bounds catch the other end, a numbered
                // line that is actually a paragraph.
                guard question.count >= 3, question.count <= 120 else { continue }

                if !question.hasSuffix("?") {
                    // Not every numbered line is a question. Muse-Glimmer answered
                    // the follow-up request with declarative sentences lifted from
                    // its own commentary, and appending a bare "?" produced the row
                    // "It establishes a wary, military tone at the castle gate.?"
                    // Dropping the trailing stop is the easy half; the real fix is
                    // to require the shape of a question rather than to costume a
                    // statement as one.
                    guard looksInterrogative(question) else { continue }
                    while let last = question.last, ".!,;:".contains(last) {
                        question.removeLast()
                    }
                    question += "?"
                }

                let key = normalized(question)
                guard !seen.contains(key) else { continue }
                seen.insert(key)
                questions.append(question)
                if questions.count == limit { break }
            }
            return questions
        }

        /// Dedupe key: case and punctuation are not a difference worth showing the
        /// reader two rows for.
        static func normalized(_ question: String) -> String {
            question.lowercased().filter { $0.isLetter || $0.isNumber || $0 == " " }
                .trimmingCharacters(in: .whitespaces)
        }

        /// Whether an item with no question mark still reads as a question.
        ///
        /// Deliberately a whitelist of openings rather than anything cleverer: the
        /// cost of rejecting a real question is one fewer row, and the cost of
        /// accepting a statement is a row that lies about being a question.
        private static let interrogatives: Set<String> = [
            "what", "why", "how", "who", "whom", "whose", "when", "where", "which",
            "is", "are", "was", "were", "does", "do", "did", "can", "could", "should",
            "would", "will", "has", "have", "had", "in", "at",
        ]

        static func looksInterrogative(_ question: String) -> Bool {
            guard
                let first = question.lowercased()
                    .split(whereSeparator: { !$0.isLetter }).first
            else { return false }
            return interrogatives.contains(String(first))
        }
    }

    // MARK: - Synopsis

    /// Its own instructions in its own throwaway session: the annotation session's
    /// KV cache has to begin with a prefix that every follow-up reuses, and
    /// splicing a summarization turn into it would both lengthen that prefix and
    /// ask one session to hold two voices.
    static let synopsisInstructions = """
        You summarize one scene of a play for a reader who is about to read it. One \
        paragraph, 60-90 words, plain modern English. Only what is in the text you \
        are given, in the order it happens: who is present, what they want, what \
        changes. No quotations, no interpretation, no list, no preamble.
        """

    static func synopsisRequest(
        _ scene: Scene, act: Int, number: Int, lineLimit: Int
    ) -> String {
        let lines = scene.lines.prefix(lineLimit)
        let coverage =
            lines.count < scene.lines.count
            ? "SCENE TEXT (lines 1-\(lines.count) of \(scene.lines.count))"
            : "SCENE TEXT"

        let utterances = lines.map {
            PassageContext.Utterance(
                speaker: $0.speaker, number: $0.number,
                text: $0.plainText,
                isDirection: $0.isDirection)
        }

        return """
            Act \(RomanNumeral.string(act)), \(SceneLabel.string(number))\
            \(scene.setting.isEmpty ? "" : " — \(scene.setting)")

            \(coverage):
            \(render(utterances))

            Summarize this scene.
            """
    }

    /// Long scenes are capped rather than chunked. Map-reduce over the handful of
    /// very long scenes (Hamlet II.ii is ~600 lines) is explicitly v2; what matters
    /// now is that the cap is stated in the prompt and surfaced in the UI.
    static let synopsisLineLimit = 300

    /// Drops a trailing fragment from a summary that ran into its token limit.
    ///
    /// The summary is asked for 60-90 words and sometimes writes 110, which used to
    /// arrive cut mid-clause — and then went into the annotation prompt that way. A
    /// summary that stops one sentence early reads as deliberate; one that stops
    /// mid-phrase reads as a bug, to the reader and to the model.
    static func tidySynopsis(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let last = trimmed.last, !".!?".contains(last) else { return trimmed }
        guard let end = trimmed.lastIndex(where: { ".!?".contains($0) }) else {
            return trimmed
        }
        return String(trimmed[...end])
    }

    // MARK: - Sampling

    /// The four presets, carried as a value rather than as mutable globals so
    /// `--greedy` is a different `SamplingPresets` and not a process-wide mutation
    /// that Swift 6 would rightly complain about.
    struct SamplingPresets: Sendable {
        var commentary: GenerateParameters
        var followUp: GenerateParameters
        var answer: GenerateParameters
        var synopsis: GenerateParameters

        /// Qwen3's own recommendation for non-thinking mode. `maxTokens` is the
        /// only field that differs between turns, which matters: mutating
        /// `kvCache`, `maxKVSize`, or `kvBits` on a live session throws
        /// `kvCacheConfigurationChanged`.
        static let recommended = SamplingPresets(
            commentary: GenerateParameters(
                maxTokens: 320, temperature: 0.7, topP: 0.8, topK: 20),
            followUp: GenerateParameters(
                maxTokens: 140, temperature: 0.7, topP: 0.8, topK: 20),
            answer: GenerateParameters(
                maxTokens: 260, temperature: 0.7, topP: 0.8, topK: 20),
            synopsis: GenerateParameters(
                maxTokens: 220, temperature: 0.3, topP: 0.8, topK: 20))

        /// `--greedy`, for prompt A/B work: two runs of the same prompt are
        /// byte-identical, so a wording change is the only variable. The seed is
        /// inert at `temperature: 0` (argmax has no RNG) and set only so the
        /// intent is legible.
        static let greedy: SamplingPresets = {
            var presets = recommended
            for keyPath in [
                \SamplingPresets.commentary, \SamplingPresets.followUp,
                \SamplingPresets.answer, \SamplingPresets.synopsis,
            ] {
                presets[keyPath: keyPath].temperature = 0
                presets[keyPath: keyPath].seed = 0
            }
            return presets
        }()
    }
}
