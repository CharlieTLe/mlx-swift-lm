import Foundation
import MLXLLM
import MLXLMCommon

/// `play:act.scene:firstLine-lastLine`, in the citation's line numbers, as the
/// `--passage` flag takes them.
///
/// Flags speak the citation's language; selections speak the array's. This is the
/// one place that translates.
struct PassageSpec {
    let key: SceneKey
    let firstLine: Int
    let lastLine: Int

    init?(_ spec: String) {
        let parts = spec.split(separator: ":")
        guard parts.count == 3 else { return nil }
        let location = parts[1].split(separator: ".")
        let lines = parts[2].split(separator: "-")
        guard location.count == 2, let act = Int(location[0]),
            let scene = Int(location[1]), let first = Int(lines[0])
        else { return nil }
        key = SceneKey(playID: String(parts[0]), act: act, scene: scene)
        firstLine = first
        lastLine = lines.count > 1 ? Int(lines[1]) ?? first : first
    }

    /// Resolves against a scene, or returns nil if those line numbers do not exist.
    func selection(in scene: Scene) -> LineSelection? {
        guard let start = scene.lines.firstIndex(where: { $0.number == firstLine }),
            let end = scene.lines.lastIndex(where: { $0.number == lastLine }),
            start <= end
        else { return nil }
        return LineSelection(anchor: start, head: end)
    }
}

/// Passages the headless modes walk, chosen to span the range that matters: a cold
/// scene opening, the longest scene in either play, a 35-line selection, prose, and
/// heavy wordplay.
enum SamplePassages {
    static let specs = [
        "hamlet:1.1:1-6",  // the opening challenge, cold start
        "hamlet:1.5:88-91",  // the Ghost's exit
        "hamlet:2.2:250-262",  // deep into the longest scene in either play
        "hamlet:3.1:62-96",  // the soliloquy, all 35 lines
        "hamlet:4.4:20-24",  // a short scene
        "hamlet:5.1:1-12",  // prose, the grave-diggers
        "macbeth:1.3:38-48",  // the witches' prophecy
        "macbeth:2.3:1-20",  // the Porter, prose and wordplay
        "macbeth:5.5:17-28",  // "Tomorrow, and tomorrow"
        "macbeth:1.7:1-28",  // a whole soliloquy
    ]
}

/// `--benchmark`: runs the real annotation path and reports what it cost.
///
/// This exists because the app's latency budget was an assumption until it was
/// measured, and a prompt edit can quietly spend it. It drives `AnnotationService`
/// exactly as the UI does — same context builder, same prompts, same session reuse
/// — so the numbers are the ones a reader gets, not a synthetic approximation.
@MainActor
enum Benchmark {

    private struct Row {
        let citation: String
        var promptTokens = 0
        var timeToFirstToken: TimeInterval = 0
        var promptTokensPerSecond: Double = 0
        var decodeTokensPerSecond: Double = 0
        var followUpPromptTokens = 0
        var followUpCount = 0
        var peakBytes = 0
        var commentaryWords = 0
    }

    static func run(options: AppOptions) async -> Bool {
        guard let corpus = try? CorpusLoader.load() else {
            print("could not load the corpus")
            return false
        }

        let service = AnnotationService(
            modelID: options.modelID ?? LLMRegistry.qwen3_4b_4bit.name,
            greedy: options.greedy)
        print("loading \(service.modelID)…")
        await service.load()
        guard service.isReady else {
            print("the model did not load")
            return false
        }

        var rows: [Row] = []
        for spec in options.passages.isEmpty ? SamplePassages.specs : options.passages {
            guard let parsed = PassageSpec(spec),
                let play = corpus.play(parsed.key.playID),
                let scene = corpus.scene(parsed.key),
                let selection = parsed.selection(in: scene),
                let context = PassageContext.build(
                    play: play, key: parsed.key, scene: scene, selection: selection,
                    cast: Cast(play: play))
            else {
                print("could not resolve \(spec)")
                return false
            }

            // `ignoringCache: true` so a second run measures the model rather than
            // the disk.
            var row = Row(citation: context.citation)
            let started = Date()
            var commentary = ""

            for await event in await service.annotate(context, ignoringCache: true) {
                switch event {
                case .promptTokens(let count):
                    row.promptTokens = count
                case .commentary(let chunk):
                    if commentary.isEmpty {
                        row.timeToFirstToken = Date().timeIntervalSince(started)
                    }
                    commentary += chunk
                case .stats(let stats):
                    row.promptTokensPerSecond = stats.promptTokensPerSecond
                    row.decodeTokensPerSecond = stats.tokensPerSecond
                    row.peakBytes = stats.peakBytes
                    print(
                        "  [turn 1] generated \(stats.generationTokenCount) tokens, "
                            + "stop: \(stats.stopReason), "
                            + "rejected tool calls: \(stats.rejectedToolCallCount)")
                case .followUpStats(let stats):
                    row.followUpPromptTokens = stats.promptTokenCount
                case .followUps(let questions):
                    row.followUpCount = questions.count
                default:
                    break
                }
            }

            row.commentaryWords = commentary.split(whereSeparator: \.isWhitespace).count
            rows.append(row)

            print(String(repeating: "=", count: 78))
            print(row.citation)
            print(String(repeating: "-", count: 78))
            print(commentary.trimmingCharacters(in: .whitespacesAndNewlines))
            print(
                String(
                    format: """

                        %d prompt tok · %.0f tok/s prefill · %.2fs to first token · \
                        %.1f tok/s decode · %d words · %d follow-ups · \
                        turn-2 prompt %d tok · %.2f GB peak
                        """,
                    row.promptTokens, row.promptTokensPerSecond, row.timeToFirstToken,
                    row.decodeTokensPerSecond, row.commentaryWords, row.followUpCount,
                    row.followUpPromptTokens, Double(row.peakBytes) / 1_073_741_824))
            print()
        }

        summarize(rows)
        if let spec = (options.passages.isEmpty ? SamplePassages.specs : options.passages)
            .first
        {
            await checkCacheAndSynopsis(spec, corpus: corpus, service: service)
        }
        return !rows.isEmpty
    }

    /// The two paths the timing table cannot show: a cache hit, and a scene summary.
    ///
    /// Both are easy to break in a way that still looks fine — a cache that never
    /// hits just costs a regeneration, and a synopsis that never arrives just makes
    /// the annotations slightly worse.
    private static func checkCacheAndSynopsis(
        _ spec: String, corpus: Corpus, service: AnnotationService
    ) async {
        guard let parsed = PassageSpec(spec),
            let play = corpus.play(parsed.key.playID),
            let scene = corpus.scene(parsed.key),
            let selection = parsed.selection(in: scene)
        else { return }

        print(String(repeating: "=", count: 78))

        // The previous pass wrote this passage to disk, so this must hit.
        guard
            let context = PassageContext.build(
                play: play, key: parsed.key, scene: scene, selection: selection,
                cast: Cast(play: play))
        else { return }

        let started = Date()
        var servedFromCache = false
        var questions = 0
        for await event in await service.annotate(context) {
            if case .cached = event { servedFromCache = true }
            if case .followUps(let list) = event { questions = list.count }
        }
        print(
            String(
                format: "cache: %@ in %.0f ms, %d follow-ups rehydrated",
                servedFromCache ? "hit" : "MISS", Date().timeIntervalSince(started) * 1000,
                questions))

        // The prewarm normally runs while the reader is reading. Awaiting it here is
        // the only way to see it from a batch run.
        service.prewarmSynopsis(key: parsed.key, scene: scene)
        await service.waitForSynopsis()
        if let synopsis = service.synopsis(for: parsed.key) {
            let words = synopsis.text.split(whereSeparator: \.isWhitespace).count
            print(
                "synopsis: \(words) words"
                    + (synopsis.isPartial ? " (partial — long scene)" : ""))
            print("  \(synopsis.text)")
        } else {
            print("synopsis: MISSING")
        }
    }

    private static func summarize(_ rows: [Row]) {
        print(String(repeating: "=", count: 78))
        print("| passage | prompt tok | prefill tok/s | TTFT | decode tok/s | words |")
        print("|---|---|---|---|---|---|")
        for row in rows {
            print(
                String(
                    format: "| %@ | %d | %.0f | %.2f s | %.1f | %d |",
                    row.citation.replacingOccurrences(of: " (this edition)", with: ""),
                    row.promptTokens, row.promptTokensPerSecond, row.timeToFirstToken,
                    row.decodeTokensPerSecond, row.commentaryWords))
        }

        func mean(_ values: [Double]) -> Double {
            values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
        }

        print()
        print(
            String(
                format: """
                    mean: %.0f prompt tok · %.0f tok/s prefill · %.2fs to first token · \
                    %.1f tok/s decode · %.0f words
                    """,
                mean(rows.map { Double($0.promptTokens) }),
                mean(rows.map(\.promptTokensPerSecond)),
                mean(rows.map(\.timeToFirstToken)),
                mean(rows.map(\.decodeTokensPerSecond)),
                mean(rows.map { Double($0.commentaryWords) })))
        print(
            String(
                format: """
                    turn 2 prefills %.0f tokens on average against a %.0f-token turn 1 \
                    — the KV cache is being reused across turns
                    """,
                mean(rows.map { Double($0.followUpPromptTokens) }),
                mean(rows.map { Double($0.promptTokens) })))
        let peak = rows.map(\.peakBytes).max() ?? 0
        print(String(format: "peak memory %.2f GB", Double(peak) / 1_073_741_824))
    }
}
