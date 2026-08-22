import Foundation

/// Everything the reader knows how to read.
struct Corpus: Sendable {
    let plays: [Play]

    func play(_ id: String) -> Play? {
        plays.first { $0.id == id }
    }

    func act(_ key: SceneKey) -> Act? {
        play(key.playID)?.acts.first { $0.number == key.act }
    }

    func scene(_ key: SceneKey) -> Scene? {
        act(key)?.scenes.first { $0.number == key.scene }
    }

    /// Every scene in reading order, which is what the navigator lists and what
    /// `--show-prompt` walks.
    var sceneKeys: [SceneKey] {
        plays.flatMap { play in
            play.acts.flatMap { act in
                act.scenes.map {
                    SceneKey(playID: play.id, act: act.number, scene: $0.number)
                }
            }
        }
    }

    var firstScene: SceneKey? { sceneKeys.first }

    /// What to open on launch, given the last recorded position.
    ///
    /// A record is only a hint, and every way it can fail to describe *this* corpus
    /// falls back rather than being handed through: `ContentView.body` renders the
    /// loading spinner forever for a `sceneKey` the corpus cannot resolve, so a play
    /// that was removed or an act that was renumbered has to come back as
    /// `firstScene`. A corpus whose text changed under a resolvable key keeps the
    /// scene but drops the selection, since the indices it holds may now point at
    /// different lines.
    func opening(from record: ReadingProgress?) -> (key: SceneKey?, selection: LineSelection?) {
        guard let record, let play = play(record.key.playID),
            let scene = scene(record.key)
        else { return (firstScene, nil) }

        guard play.source.textSHA256 == record.corpusStamp else { return (record.key, nil) }

        return (record.key, record.selection?.clamped(to: scene.lines))
    }
}

enum CorpusLoader {
    /// Where `Plays/` is.
    ///
    /// `Bundle.module` is synthesized for a SwiftPM target and simply does not exist
    /// in the iOS app target, which compiles these same sources directly. Xcode
    /// defines `SWIFT_PACKAGE` only for package targets, so it is the flag that tells
    /// the two builds apart. In the app the directory is a folder reference copied to
    /// the `.app` root, so `Bundle.main` finds it the same way.
    #if SWIFT_PACKAGE
    private static let resources = Bundle.module
    #else
    private static let resources = Bundle.main
    #endif

    enum Failure: LocalizedError {
        case resourcesMissing
        case noPlays(URL)
        case decode(String, Error)

        var errorDescription: String? {
            switch self {
            case .resourcesMissing:
                // `Bundle.module` for an executable target resolves to a
                // `.bundle` directory beside the binary, so this fires whenever
                // the executable has been moved out on its own — which is the one
                // way an installed copy can be broken, hence both audiences here.
                // The iOS app has a fourth: `Plays/` missing from the `.app` means
                // the folder reference dropped out of Copy Bundle Resources.
                "Could not find the Plays resource directory. "
                    + "ShakespeareReader_ShakespeareReader.bundle must sit beside the "
                    + "executable. Run `brew reinstall shakespeare-reader` if this was "
                    + "installed with Homebrew, or `swift run -c release ShakespeareReader` "
                    + "from a source checkout. In the iOS app, check that Plays is still "
                    + "a folder reference in Copy Bundle Resources."
            case .noPlays(let url):
                "No play JSON found in \(url.path). From a source checkout, generate it "
                    + "with `python3 tools/build_corpus.py --all --out "
                    + "Sources/ShakespeareReader/Resources/Plays`."
            case .decode(let name, let error):
                "\(name) could not be decoded: \(error)"
            }
        }
    }

    /// Loads every `*.json` in the bundled `Plays` directory.
    ///
    /// Enumerating rather than naming files is why `Package.swift` uses `.copy`
    /// instead of `.process`: adding a play is dropping a file in, with no build
    /// file to edit.
    static func load() throws -> Corpus {
        guard let directory = resources.url(forResource: "Plays", withExtension: nil)
        else {
            throw Failure.resourcesMissing
        }

        let files = try FileManager.default
            .contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
            // Stable order, so the navigator does not reshuffle between launches.
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        guard !files.isEmpty else { throw Failure.noPlays(directory) }

        let decoder = JSONDecoder()
        let plays = try files.map { url in
            do {
                return try decoder.decode(Play.self, from: Data(contentsOf: url))
            } catch {
                throw Failure.decode(url.lastPathComponent, error)
            }
        }
        return Corpus(plays: plays)
    }
}
