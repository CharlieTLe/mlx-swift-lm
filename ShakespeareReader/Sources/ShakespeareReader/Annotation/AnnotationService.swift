import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

/// Where a generation is, between a selection and a finished annotation. Every wait
/// here needs a name, in the spirit of MuseGlimmer's `Phase`.
enum Phase: Equatable, Sendable {
    case idle
    case cached
    case summarizingScene
    case prefilling
    /// Prefill is done but nothing is streaming yet. Qwen3 under
    /// `enable_thinking: false` passes through this in milliseconds, but a model
    /// that reasons privately sits here for seconds with nothing to show: the
    /// framework withholds reasoning from the public stream, so an unnamed window
    /// looks like a hang.
    case reasoning
    case streaming
    case listingFollowUps
    case answering
}

struct GenerationStats: Sendable, Equatable {
    var promptTokenCount: Int
    var generationTokenCount: Int
    var promptTime: TimeInterval
    var tokensPerSecond: Double
    var peakBytes: Int
    /// Why generation stopped. `.length` on an empty annotation means the budget
    /// went somewhere invisible — reasoning, or a rejected tool call.
    var stopReason: GenerateStopReason
    var rejectedToolCallCount: Int

    /// Prefill throughput, the number the plan's shedding list is decided on.
    var promptTokensPerSecond: Double {
        promptTime > 0 ? Double(promptTokenCount) / promptTime : 0
    }
}

enum AnnotationEvent: Sendable {
    /// Served entirely from disk: no model work at all.
    case cached(CachedPassage)
    case phase(Phase)
    case promptTokens(Int)
    case prefill(processed: Int, total: Int)
    case commentary(String)
    case answer(String)
    case followUps([String])
    case stats(GenerationStats)
    /// The follow-up turn's own numbers. Separate from `.stats` because the UI
    /// should keep showing the reader's own request, and because this is the one
    /// measurement that shows KV cache reuse: turn 2 prefills 250-350 tokens
    /// against a 526-1,023-token turn 1, and its cost tracks the length of the
    /// commentary rather than the length of the context.
    case followUpStats(GenerationStats)
    case failed(String)
}

/// Loads Qwen3-4B once and streams annotations from it.
///
/// Follows `MuseGlimmerService` closely: `@MainActor @Observable`, a `LoadState`,
/// `#hubDownloader()` / `#huggingFaceTokenizerLoader()` to load, and an
/// `AsyncStream` whose `continuation.onTermination` cancels the work `Task`.
///
/// `ChatSession` is a non-`Sendable` `final class`, which shapes the whole API: the
/// session lives inside this main-actor class, work runs in a `Task {}` created
/// from a main-actor method so it inherits that isolation, and the `@Sendable`
/// `onTermination` closure may only `cancel()` — it must never touch the session.
@MainActor
@Observable
final class AnnotationService {

    enum LoadState {
        case idle
        case loading(Progress?)
        case ready
        case failed(String)
    }

    private(set) var loadState: LoadState = .idle

    /// True while a scene summary is being generated in the background. The
    /// annotation never waits for it, so this is reported in the status strip only
    /// when nothing else is running — otherwise a background job would look like the
    /// reader's own request stalling.
    private(set) var isSummarizing = false

    let modelID: String
    private let presets: Prompts.SamplingPresets
    private let cache: AnnotationCache
    private var container: ModelContainer?

    /// One session per passage, shared by the commentary, the follow-up list, and
    /// every tapped question. Turn 1 streams the commentary, turn 2 produces the
    /// numbered list, turn 3+ answers taps — all off the same KV cache, so the
    /// 600-900 token context block is prefilled once and a later turn costs about
    /// 300 tokens (the re-rendered assistant text) rather than the whole prompt.
    private var passage: PassageSession?

    private var activeTask: Task<Void, Never>?
    private var synopsisTask: Task<Void, Never>?
    private var synopses: [SceneKey: CachedSynopsis] = [:]

    /// Multiplier on every `maxTokens`, for a model that reasons before answering.
    ///
    /// Reasoning tokens are generated and then withheld from the stream, but they
    /// still count against `maxTokens`. Muse-Glimmer reasons at length under
    /// `to=self<|message|>`, so the 320-token commentary budget would be spent
    /// before the visible answer began. Set from the loaded configuration rather
    /// than from the model id, so any model that declares reasoning delimiters gets
    /// the headroom.
    private var tokenBudgetMultiplier = 1

    private func budgeted(_ parameters: GenerateParameters) -> GenerateParameters {
        guard tokenBudgetMultiplier > 1, let maxTokens = parameters.maxTokens else {
            return parameters
        }
        var scaled = parameters
        scaled.maxTokens = maxTokens * tokenBudgetMultiplier
        return scaled
    }

    init(modelID: String = LLMRegistry.qwen3_4b_4bit.name, greedy: Bool = false) {
        self.modelID = modelID
        presets = greedy ? .greedy : .recommended
        cache = AnnotationCache(modelID: modelID)
    }

    private final class PassageSession {
        let key: PassageKey
        let digest: String
        let session: ChatSession
        var commentary = ""
        var followUpsRaw = ""
        var followUps: [String] = []
        var asked: Set<String> = []
        var synopsisUsed: Bool
        var promptTokenCount = 0
        var tokensPerSecond: Double = 0
        /// A cache hit rehydrates lazily, on the first follow-up tap, so a cache hit
        /// itself costs nothing.
        var needsHistoryPrefill: Bool

        init(
            key: PassageKey, digest: String, session: ChatSession,
            synopsisUsed: Bool, needsHistoryPrefill: Bool = false
        ) {
            self.key = key
            self.digest = digest
            self.session = session
            self.synopsisUsed = synopsisUsed
            self.needsHistoryPrefill = needsHistoryPrefill
        }
    }

    // MARK: - Loading

    /// The registered configuration for `modelID`, from whichever registry has it.
    ///
    /// This matters more than it looks. `AbstractModelRegistry.configuration(id:)`
    /// returns a *bare* configuration for an id it does not know, which silently
    /// drops the model's stop tokens, tool-call format and reasoning delimiters —
    /// Qwen3 needs `extraEOSTokens: ["<|im_end|>"]` or it runs past the end of every
    /// turn, and Muse-Glimmer needs its `reasoningConfig` or its private reasoning
    /// arrives as visible text.
    static func configuration(for modelID: String) -> ModelConfiguration {
        if LLMRegistry.shared.contains(id: modelID) {
            return LLMRegistry.shared.configuration(id: modelID)
        }
        if VLMRegistry.shared.contains(id: modelID) {
            return VLMRegistry.shared.configuration(id: modelID)
        }
        return LLMRegistry.shared.configuration(id: modelID)
    }

    /// Sizes the MLX buffer-reuse pool to the weights that were actually loaded.
    ///
    /// 256 MB is the figure `MLXFoundationModels` picks for a ~4B model and is
    /// right for the default. It thrashes badly at 19 GB, where a single forward
    /// pass churns far larger activations — `MuseGlimmerDemo` needed 2 GB for the
    /// 30B VLM. Deciding after the load, from resident size, means `--model` picks
    /// the right pool without a table of model sizes to keep current.
    ///
    /// The 48 GB ceiling is a Mac number and stays on the Mac. On iOS the ceiling is
    /// applied unconditionally instead, and low: 6 GB is well over the 4B model's
    /// ~3 GB peak but under what iOS will hand a single app even with the
    /// increased-memory-limit entitlement, so MLX applies backpressure, waiting for
    /// buffers to free rather than allocating, instead of the app being jetsam-killed
    /// with no error to report. The `isLarge` branch cannot fire on a phone at 4B, so
    /// this is not a second guess at the same question.
    private static func tuneMemory() {
        let resident = Memory.snapshot().activeMemory
        let isLarge = resident > 8 * 1024 * 1024 * 1024
        Memory.cacheLimit = isLarge ? 2 * 1024 * 1024 * 1024 : 256 * 1024 * 1024
        #if os(macOS)
        if isLarge {
            Memory.memoryLimit = 48 * 1024 * 1024 * 1024
        }
        #else
        Memory.memoryLimit = 6 * 1024 * 1024 * 1024
        #endif
    }

    /// Where the weights come from.
    ///
    /// macOS takes the default, which resolves to `~/.cache/huggingface/hub` and is
    /// shared with every other MLX tool and with Python's `huggingface_hub`.
    ///
    /// iOS may not. `swift-huggingface`'s location provider picks
    /// `Library/Caches/huggingface/hub` for a sandboxed app, and iOS reclaims `Caches`
    /// under disk pressure whenever it likes, which for a 2.2 GB snapshot means a
    /// silent full re-download on some later launch, with the reader watching a
    /// progress bar they already sat through once. `Application Support` is not
    /// purgeable, and it is already where `AnnotationCache` writes.
    private static var downloader: any Downloader {
        #if os(macOS)
        #hubDownloader()
        #else
        #hubDownloader(HubClient(cache: HubCache(cacheDirectory: hubCacheDirectory())))
        #endif
    }

    #if !os(macOS)
    /// `Application Support/huggingface/hub`, a sibling of the annotation cache's
    /// own `ShakespeareReader` directory rather than a child of it, so the layout
    /// under `huggingface/` stays the Python-compatible one `HubCache` documents.
    ///
    /// Excluded from backup: these are bytes Hugging Face will hand back on
    /// demand, and 2.2 GB of them has no business in anyone's iCloud backup or in
    /// the restore time of their next phone.
    private static func hubCacheDirectory() -> URL {
        var root = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("huggingface", isDirectory: true)
        let hub = root.appendingPathComponent("hub", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: hub, withIntermediateDirectories: true)

        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? root.setResourceValues(values)
        return hub
    }
    #endif

    func load() async {
        if case .ready = loadState { return }
        if case .loading = loadState { return }

        // Before the first `Memory` touch below, which is what would abort the process
        // on a Simulator rather than fail. See `hasMLXDevice`.
        guard hasMLXDevice else {
            loadState = .failed(
                "The iOS Simulator has no Metal device for MLX, so nothing can be "
                    + "annotated here. The reader, the corpus and the layout all work; "
                    + "run on a device to generate.")
            return
        }

        Memory.cacheLimit = 256 * 1024 * 1024
        loadState = .loading(nil)
        do {
            let container = try await loadModelContainer(
                from: Self.downloader,
                using: #huggingFaceTokenizerLoader(),
                configuration: Self.configuration(for: modelID)
            ) { progress in
                Task { @MainActor in
                    self.loadState = .loading(progress)
                }
            }
            self.container = container
            let configuration = await container.configuration
            tokenBudgetMultiplier = configuration.reasoningConfig != nil ? 3 : 1
            Self.tuneMemory()
            loadState = .ready
        } catch {
            loadState = .failed(String(describing: error))
        }
    }

    var isReady: Bool {
        if case .ready = loadState { return true }
        return false
    }

    // MARK: - Annotation

    /// Streams the commentary for `context`, then the follow-up questions.
    ///
    /// `async` because the previous generation has to be cancelled and waited out
    /// *before* the new task exists. Doing it inside the new task would mean either
    /// awaiting itself or capturing the old `ChatSession` in a `@Sendable` closure,
    /// and `ChatSession` is not `Sendable`.
    ///
    /// - Parameter ignoringCache: ⌘R, which regenerates rather than re-reading.
    func annotate(_ context: PassageContext, ignoringCache: Bool = false) async
        -> AsyncStream<AnnotationEvent>
    {
        await displaceActiveWork().finish()

        let (stream, continuation) = AsyncStream<AnnotationEvent>.makeStream()

        let work = Task { @MainActor in
            guard let container else {
                continuation.yield(.failed(AnnotationError.notLoaded.localizedDescription))
                continuation.finish()
                return
            }

            if !ignoringCache,
                let entry = await cache.passage(for: context.key, digest: context.digest)
            {
                continuation.yield(.phase(.cached))
                continuation.yield(.cached(entry))
                // The text is restored now; the session is built but not prefilled
                // until the first follow-up tap, so a cache hit itself costs nothing.
                let restored = PassageSession(
                    key: context.key, digest: context.digest,
                    session: rehydratedSession(container, context: context, entry: entry),
                    synopsisUsed: entry.synopsisUsed, needsHistoryPrefill: true)
                restored.commentary = entry.commentary
                restored.followUpsRaw = entry.followUpsRaw
                restored.followUps = entry.followUps
                restored.asked = Set(entry.followUps.map(Prompts.FollowUps.normalized))
                passage = restored
                continuation.yield(.followUps(entry.followUps))
                continuation.finish()
                schedulePrewarmRetry()
                return
            }

            // A selection cancels the synopsis prewarm immediately and proceeds with
            // whatever the caller already had. Waiting for it would put a
            // multi-second stall on the first selection in every scene, because the
            // prewarm holds the single `ModelContainer` while it runs.
            cancelPrewarm()

            var parameters = budgeted(presets.commentary)
            parameters.prefill = PrefillParameters(progress: { processed, total in
                continuation.yield(.prefill(processed: processed, total: total))
            })

            let session = ChatSession(
                container,
                instructions: Prompts.annotatorInstructions,
                generateParameters: parameters,
                additionalContext: Self.nonThinking)
            let current = PassageSession(
                key: context.key, digest: context.digest, session: session,
                synopsisUsed: context.synopsis != nil)
            passage = current

            let request = Prompts.annotationRequest(context)
            current.promptTokenCount = await Self.tokenCount(
                container: container,
                instructions: Prompts.annotatorInstructions,
                user: request)
            continuation.yield(.promptTokens(current.promptTokenCount))
            continuation.yield(.phase(.prefilling))

            do {
                var stopReason: GenerateStopReason?
                for try await item in session.streamDetails(to: request) {
                    if let chunk = item.chunk {
                        continuation.yield(.phase(.streaming))
                        current.commentary += chunk
                        continuation.yield(.commentary(chunk))
                    }
                    if let info = item.info {
                        stopReason = info.stopReason
                        current.tokensPerSecond = info.tokensPerSecond
                        continuation.yield(.stats(Self.stats(info)))
                    }
                }

                // Partial output is never cached: only a stream that reached `.info`
                // with a stop reason other than `.cancelled` produced a whole
                // annotation.
                guard stopReason != nil, stopReason != .cancelled, !Task.isCancelled
                else {
                    continuation.finish()
                    return
                }

                continuation.yield(.phase(.listingFollowUps))
                let followUps = try await requestFollowUps(
                    current, stats: { continuation.yield(.followUpStats($0)) })
                continuation.yield(.followUps(followUps))

                await cache.store(
                    CachedPassage(
                        schemaVersion: AnnotationCache.schemaVersion,
                        promptVersion: Prompts.version,
                        modelID: modelID,
                        passageDigest: current.digest,
                        commentary: current.commentary,
                        followUpsRaw: current.followUpsRaw,
                        followUps: followUps,
                        synopsisUsed: current.synopsisUsed,
                        generatedAt: Date(),
                        promptTokenCount: current.promptTokenCount,
                        tokensPerSecond: current.tokensPerSecond),
                    for: current.key)
            } catch is CancellationError {
                // Stopped by the reader; whatever streamed already stays on screen.
            } catch {
                continuation.yield(.failed(String(describing: error)))
            }

            continuation.yield(.phase(.idle))
            continuation.finish()
            schedulePrewarmRetry()
        }

        // Cancelling the consumer has to cancel the generation. This closure is
        // `@Sendable` and runs off the main actor, so it may only cancel — it must
        // not touch the session.
        continuation.onTermination = { _ in work.cancel() }
        activeTask = work
        return stream
    }

    /// Answers a tapped question on the same session, then refreshes the ask rows.
    func answer(_ question: String) -> AsyncStream<AnnotationEvent> {
        let (stream, continuation) = AsyncStream<AnnotationEvent>.makeStream()

        let work = Task { @MainActor in
            guard let current = passage else {
                continuation.finish()
                return
            }

            current.asked.insert(Prompts.FollowUps.normalized(question))
            // Only `maxTokens` changes between turns. Mutating `kvCache`,
            // `maxKVSize`, or `kvBits` on a live session throws
            // `kvCacheConfigurationChanged`, and `instructions` is never touched
            // mid-session.
            current.session.generateParameters = budgeted(presets.answer)
            if current.needsHistoryPrefill {
                // A rehydrated session prefills the whole recorded transcript —
                // context, commentary and question list, so roughly 1.2-1.5k tokens
                // or about a second. Paid here, on the first tap, rather than on
                // every cache hit.
                continuation.yield(.phase(.prefilling))
                current.needsHistoryPrefill = false
            }
            continuation.yield(.phase(.answering))

            do {
                for try await item in current.session.streamDetails(
                    to: Prompts.answerRequest(question))
                {
                    if let chunk = item.chunk {
                        continuation.yield(.answer(chunk))
                    }
                    if let info = item.info {
                        continuation.yield(.stats(Self.stats(info)))
                    }
                }

                guard !Task.isCancelled else {
                    continuation.finish()
                    return
                }

                // Sustain the loop: ask for five, keep the first four that are not
                // already in `asked`.
                continuation.yield(.phase(.listingFollowUps))
                current.session.generateParameters = budgeted(presets.followUp)
                let raw = try await collect(
                    current.session.streamDetails(to: Prompts.moreFollowUpsRequest))
                let fresh = Prompts.FollowUps.parse(raw, asked: current.asked)
                if !fresh.isEmpty {
                    current.followUps = fresh
                    continuation.yield(.followUps(fresh))
                }
            } catch is CancellationError {
            } catch {
                continuation.yield(.failed(String(describing: error)))
            }

            continuation.yield(.phase(.idle))
            continuation.finish()
        }

        continuation.onTermination = { _ in work.cancel() }
        activeTask = work
        return stream
    }

    /// Turn 2: the numbered list.
    private func requestFollowUps(
        _ current: PassageSession,
        stats report: (GenerationStats) -> Void = { _ in }
    ) async throws -> [String] {
        current.session.generateParameters = budgeted(presets.followUp)

        var raw = try await collect(
            current.session.streamDetails(to: Prompts.followUpRequest), stats: report)
        var parsed = Prompts.FollowUps.parse(raw, asked: current.asked)

        // Fewer than two survivors gets one retry, then the row is hidden — one
        // lonely chip looks broken.
        if parsed.count < 2 {
            raw = try await collect(
                current.session.streamDetails(to: Prompts.followUpRetry), stats: report)
            parsed = Prompts.FollowUps.parse(raw, asked: current.asked)
        }

        current.followUpsRaw = raw
        current.followUps = parsed.count < 2 ? [] : parsed
        current.asked.formUnion(current.followUps.map(Prompts.FollowUps.normalized))
        return current.followUps
    }

    /// Drains a stream to a string, reporting the turn's own statistics.
    private func collect(
        _ stream: AsyncThrowingStream<Generation, Error>,
        stats report: (GenerationStats) -> Void = { _ in }
    ) async throws -> String {
        var text = ""
        for try await item in stream {
            if let chunk = item.chunk { text += chunk }
            if let info = item.info { report(Self.stats(info)) }
        }
        return text
    }

    /// Rebuilds a session from a cache entry's recorded text.
    ///
    /// `followUpsRaw` rather than the parsed list, so the rehydrated history is
    /// byte-identical to what the model actually said.
    ///
    /// The user turn has to be reconstructed rather than stored, and the one part of
    /// the context that can have changed since is the scene summary — it may have
    /// finished generating after this passage was annotated. `synopsisUsed` is
    /// recorded for exactly this: replaying a prompt containing a summary the model
    /// never saw would make its own earlier answer look like a non sequitur.
    private func rehydratedSession(
        _ container: ModelContainer, context: PassageContext, entry: CachedPassage
    ) -> ChatSession {
        var replayed = context
        if !entry.synopsisUsed {
            replayed.synopsis = nil
            replayed.synopsisIsPartial = false
        }

        return ChatSession(
            container,
            instructions: Prompts.annotatorInstructions,
            history: [
                .user(Prompts.annotationRequest(replayed)),
                .assistant(entry.commentary),
                .user(Prompts.followUpRequest),
                .assistant(entry.followUpsRaw),
            ],
            generateParameters: budgeted(presets.answer),
            additionalContext: Self.nonThinking)
    }

    // MARK: - Cancellation

    /// In-flight work, detached from the service so it can be waited out from the
    /// task that replaces it.
    ///
    /// `@unchecked Sendable`, with a specific reason rather than to quiet the
    /// checker. `ChatSession.synchronize()` is a `@concurrent` `async` method on a
    /// non-`Sendable` class, so awaiting it from the main actor is "sending" the
    /// session off its actor. That is safe *here* because by construction this box
    /// holds the only remaining reference: `displaceActiveWork()` clears `passage`
    /// before the box exists, and `finish()` awaits the task that was using the
    /// session before touching it. The box is dropped immediately after. The
    /// alternative — skipping `synchronize()` — reintroduces the stall this path
    /// exists to remove.
    private struct DisplacedWork: @unchecked Sendable {
        let task: Task<Void, Never>?
        let session: ChatSession?

        /// Cancel, then wait, then wait again for exclusive cache access. All three
        /// steps matter. `cancel()` alone returns while the forward pass is still
        /// resident. `await task.value` alone returns while `ChatSession`'s own
        /// unstructured generation task may still be winding down and holding the
        /// cache lock — which is exactly what `synchronize()` waits for.
        func finish() async {
            task?.cancel()
            await task?.value
            await session?.synchronize()
        }
    }

    /// Takes ownership of whatever is running and clears the service's references.
    ///
    /// The session goes with it: a cancelled generation invalidates
    /// `ChatSession`'s token ledger, so a cancelled session must never be reused.
    private func displaceActiveWork() -> DisplacedWork {
        let work = DisplacedWork(task: activeTask, session: passage?.session)
        activeTask?.cancel()
        activeTask = nil
        passage = nil
        return work
    }

    /// Cancels in-flight work and discards the session. Bound to Esc.
    func stopActiveWork() async {
        await displaceActiveWork().finish()
    }

    // MARK: - Scene synopsis

    /// Starts summarizing a scene when it opens.
    ///
    /// The reader spends tens of seconds reading before selecting anything, so this
    /// is usually ready in time. It holds the single `ModelContainer` while it runs,
    /// which is why a selection cancels it rather than queueing behind it.
    func prewarmSynopsis(key: SceneKey, scene: Scene) {
        guard synopses[key] == nil, isReady, let container else { return }
        cancelPrewarm()

        synopsisTask = Task { @MainActor in
            if let entry = await cache.synopsis(for: key) {
                synopses[key] = entry
                return
            }
            guard !Task.isCancelled else { return }

            let session = ChatSession(
                container,
                instructions: Prompts.synopsisInstructions,
                generateParameters: budgeted(presets.synopsis),
                additionalContext: Self.nonThinking)
            let isPartial = scene.lines.count > Prompts.synopsisLineLimit
            let request = Prompts.synopsisRequest(
                scene, act: key.act, number: key.scene,
                lineLimit: Prompts.synopsisLineLimit)

            isSummarizing = true
            defer { isSummarizing = false }
            do {
                let text = Prompts.tidySynopsis(try await session.respond(to: request))
                guard !Task.isCancelled, !text.isEmpty else { return }
                let entry = CachedSynopsis(
                    schemaVersion: AnnotationCache.schemaVersion,
                    promptVersion: Prompts.version,
                    modelID: modelID,
                    text: text,
                    isPartial: isPartial,
                    generatedAt: Date())
                synopses[key] = entry
                await cache.store(entry, for: key)
            } catch {
                // A synopsis never blocks an annotation, so a failure is silent by
                // design; the annotation proceeds without it.
            }
            // The session is discarded here: it is a throwaway, and the synopsis
            // enters the annotation prompt as text rather than as KV state.
        }
    }

    func cancelPrewarm() {
        synopsisTask?.cancel()
        synopsisTask = nil
    }

    /// Waits for an in-flight prewarm to finish. For `--benchmark` only: the reader's
    /// path never waits on a synopsis, which is the whole point of prewarming it.
    func waitForSynopsis() async {
        await synopsisTask?.value
    }

    /// Retries the prewarm after a couple of seconds of idle, so a scene the reader
    /// keeps selecting in eventually gets its summary.
    private var pendingPrewarm: (key: SceneKey, scene: Scene)?

    func rememberSceneForPrewarm(key: SceneKey, scene: Scene) {
        pendingPrewarm = (key, scene)
    }

    private func schedulePrewarmRetry() {
        guard let pendingPrewarm, synopses[pendingPrewarm.key] == nil else { return }
        cancelPrewarm()
        synopsisTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }
            prewarmSynopsis(key: pendingPrewarm.key, scene: pendingPrewarm.scene)
        }
    }

    func synopsis(for key: SceneKey) -> CachedSynopsis? { synopses[key] }

    // MARK: - Helpers

    /// Template variables that ask a model not to think at length before answering.
    ///
    /// Both keys are sent to every model because a chat template simply ignores a
    /// variable it does not reference, and the two families spell this differently:
    ///
    /// - `enable_thinking` is Qwen3's. Without it Qwen3 reasons before the first
    ///   visible token — the silent window MuseGlimmer's README documents. The
    ///   pattern is `IntegrationTestHelpers.structuredToolContinuation`.
    /// - `reasoning_strength` is Muse-Glimmer's, from its own
    ///   `chat_template.jinja`: `render_reasoning()` writes "Reasoning strength:
    ///   <value>." into the system block and **defaults to `high`**. At `high` it
    ///   spent all 960 available tokens reasoning about four lines of Hamlet and
    ///   produced no visible answer at all, because reasoning tokens count against
    ///   `maxTokens` while being withheld from the stream.
    static let nonThinking: [String: any Sendable] = [
        "enable_thinking": false,
        "reasoning_strength": "low",
    ]

    /// The exact prompt length, without generating anything. Used by
    /// `--show-prompt`.
    func promptTokenCount(for request: String) async -> Int {
        guard let container else { return 0 }
        return await Self.tokenCount(
            container: container, instructions: Prompts.annotatorInstructions,
            user: request)
    }

    /// The exact prompt length, without generating anything.
    private static func tokenCount(
        container: ModelContainer, instructions: String, user: String
    ) async -> Int {
        let tokenizer = await container.tokenizer
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": instructions],
            ["role": "user", "content": user],
        ]
        return
            (try? tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: nonThinking))?.count ?? 0
    }

    private static func stats(_ info: GenerateCompletionInfo) -> GenerationStats {
        GenerationStats(
            promptTokenCount: info.promptTokenCount,
            generationTokenCount: info.generationTokenCount,
            promptTime: info.promptTime,
            tokensPerSecond: info.tokensPerSecond,
            peakBytes: Memory.snapshot().peakMemory,
            stopReason: info.stopReason,
            rejectedToolCallCount: info.rejectedToolCallCount)
    }
}

enum AnnotationError: LocalizedError {
    case notLoaded

    var errorDescription: String? {
        switch self {
        case .notLoaded: "The model is not loaded yet."
        }
    }
}
