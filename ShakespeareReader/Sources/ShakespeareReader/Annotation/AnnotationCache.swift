import Foundation

/// One cached annotation.
struct CachedPassage: Codable, Sendable {
    let schemaVersion: Int
    let promptVersion: Int
    let modelID: String
    /// SHA-256 of the selected line text.
    let passageDigest: String
    let commentary: String
    /// The model's literal numbered list...
    let followUpsRaw: String
    /// ...and the parsed form. Both, so a rehydrated history is byte-identical to
    /// what the model actually said.
    let followUps: [String]
    let synopsisUsed: Bool
    let generatedAt: Date
    let promptTokenCount: Int
    let tokensPerSecond: Double
}

struct CachedSynopsis: Codable, Sendable {
    let schemaVersion: Int
    let promptVersion: Int
    let modelID: String
    let text: String
    let isPartial: Bool
    let generatedAt: Date
}

/// JSON on disk, behind an actor so the file IO stays off the main actor.
///
/// `saveCache(to:)` snapshots were considered and rejected: hundreds of MB per
/// passage. A cache hit here renders instantly with zero model work, and the
/// session is rehydrated from the recorded text only when a follow-up is tapped.
actor AnnotationCache {
    static let schemaVersion = 1

    private let root: URL
    private let modelID: String

    /// An unbundled SwiftPM executable has no bundle identifier, so the directory
    /// is named explicitly rather than derived from one.
    init(modelID: String) {
        self.modelID = modelID
        root = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ShakespeareReader", isDirectory: true)
    }

    var directory: URL { root }

    // MARK: - Passages

    /// A hit requires `schemaVersion`, `promptVersion`, `modelID` **and**
    /// `passageDigest` to match. The digest is what stops a re-parsed corpus with
    /// shifted line indices from serving an annotation of different lines.
    func passage(for key: PassageKey, digest: String) -> CachedPassage? {
        guard let entry: CachedPassage = read(url: url(for: key)) else { return nil }
        guard entry.schemaVersion == Self.schemaVersion,
            entry.promptVersion == Prompts.version,
            entry.modelID == modelID,
            entry.passageDigest == digest
        else { return nil }
        return entry
    }

    func store(_ entry: CachedPassage, for key: PassageKey) {
        write(entry, to: url(for: key))
    }

    // MARK: - Synopses

    func synopsis(for key: SceneKey) -> CachedSynopsis? {
        guard let entry: CachedSynopsis = read(url: url(for: key)) else { return nil }
        guard entry.schemaVersion == Self.schemaVersion,
            entry.promptVersion == Prompts.version,
            entry.modelID == modelID
        else { return nil }
        return entry
    }

    func store(_ entry: CachedSynopsis, for key: SceneKey) {
        write(entry, to: url(for: key))
    }

    // MARK: - Paths and IO

    private func url(for key: PassageKey) -> URL {
        root.appendingPathComponent("passages", isDirectory: true)
            .appendingPathComponent("\(key.slug).json")
    }

    private func url(for key: SceneKey) -> URL {
        root.appendingPathComponent("synopses", isDirectory: true)
            .appendingPathComponent("\(key.slug).json")
    }

    private func read<T: Decodable>(url: URL) -> T? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try? decoder.decode(T.self, from: data)
    }

    /// Cache writes are best-effort. A full disk should not take down a reader.
    private func write<T: Encodable>(_ value: T, to url: URL) {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try encoder.encode(value).write(to: url, options: .atomic)
        } catch {
            print("cache write failed for \(url.lastPathComponent): \(error)")
        }
    }
}
