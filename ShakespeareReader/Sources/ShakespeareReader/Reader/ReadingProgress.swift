import Foundation

/// Where the reader was when they last quit.
///
/// One position for the whole app rather than one per play: "where I left off" is a
/// single place, and clicking a play in the navigator still opens its first scene.
struct ReadingProgress: Codable, Sendable, Equatable {
    let schemaVersion: Int
    var key: SceneKey
    var selection: LineSelection?
    /// `Play.source.textSHA256` for `key.playID`.
    ///
    /// The mirror of `AnnotationCache`'s `passageDigest` gate: a stored selection is a
    /// pair of indices into `Scene.lines`, so a corpus rebuilt with a different parse
    /// must not restore a highlight over different lines. On mismatch the scene is
    /// kept and the selection dropped.
    var corpusStamp: String
}

/// `ReadingProgress`, in `UserDefaults`.
///
/// `UserDefaults` rather than JSON beside the annotation cache: the record is one
/// small value, and it belongs with the window frame and `readerFont` rather than
/// with generated model output a reader might reasonably want to delete. The app is
/// an unbundled SwiftPM executable with no bundle id, but CFPreferences falls back to
/// the process name, so this lands in `~/Library/Preferences/ShakespeareReader.plist`
/// alongside those.
///
/// **Not** `@AppStorage`. Nothing renders from the record: it is read once at launch
/// and written thereafter, so the observation `@AppStorage` provides buys nothing. It
/// would also force a `RawRepresentable where RawValue == String` bridge onto
/// `ReadingProgress`, and the stdlib's own `Codable` conformance for string-backed raw
/// values would then shadow the synthesized one and recurse. `Data` goes in directly.
enum ProgressStore {
    static let schemaVersion = 1

    private static let progressKey = "readingProgress"

    // MARK: - Position

    /// `nil` for anything unreadable, whether absent, corrupt, or written by a schema
    /// this build does not know, matching `AnnotationCache.read(url:)`. A position is a
    /// convenience, so a bad one is worth nothing more than starting at the beginning.
    static func progress() -> ReadingProgress? {
        guard let data = UserDefaults.standard.data(forKey: progressKey),
            let record = try? JSONDecoder().decode(ReadingProgress.self, from: data),
            record.schemaVersion == schemaVersion
        else { return nil }
        return record
    }

    static func save(_ record: ReadingProgress) {
        guard let data = try? JSONEncoder().encode(record) else { return }
        UserDefaults.standard.set(data, forKey: progressKey)
    }
}
