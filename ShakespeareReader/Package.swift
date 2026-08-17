// swift-tools-version: 6.2

import PackageDescription

// A standalone SwiftUI macOS app that annotates Shakespeare on-device.
//
// This is a sibling package with a *local path* dependency on the checkout, so it
// builds against this working copy rather than a published tag. Mirrors
// MuseGlimmerDemo/Package.swift, with MLXLLM in place of MLXVLM.
//
// Run with:  cd ShakespeareReader && swift run -c release ShakespeareReader
let package = Package(
    name: "ShakespeareReader",
    platforms: [.macOS(.v14)],
    // Declared explicitly so a packager can ask for it by name
    // (`swift build --product ShakespeareReader`) rather than rely on the
    // implicit product SwiftPM synthesizes for an executable target.
    //
    // The name stays CamelCase deliberately. It is the binary name, which is the
    // process name, which is both the `UserDefaults` domain
    // (`~/Library/Preferences/ShakespeareReader.plist`, holding `readerFont`,
    // `readingProgress` and `showsCommentary`) and the resource bundle name
    // (`ShakespeareReader_ShakespeareReader.bundle`). Renaming it would orphan
    // every reader's saved position; the kebab-case `shakespeare-reader` command
    // is a wrapper Homebrew installs, not this.
    products: [
        .executable(name: "ShakespeareReader", targets: ["ShakespeareReader"])
    ],
    dependencies: [
        // `traits: []` turns off the default `FoundationModelsIntegration` trait,
        // which compiles MLXFoundationModels to an empty library. This app never
        // touches Apple's FoundationModels adapter, and MLXHuggingFace pulls that
        // target in only when the trait is on — so leaving it enabled would make
        // this build depend on a target the app does not use.
        .package(name: "mlx-swift-lm", path: "..", traits: []),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "ShakespeareReader",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                // For `--model mlx-community/Muse-Glimmer-30B-4bit` and friends.
                // Linking it is what registers the VLM factory and its registry;
                // without it a VLM id loads as a bare `ModelConfiguration` and
                // silently loses its stop tokens and reasoning delimiters.
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/ShakespeareReader",
            // `.copy`, not `.process`: this keeps `Plays/` a real directory in the
            // bundle so CorpusLoader can enumerate it. Adding a play is then
            // dropping a JSON file in, with no edit here.
            //
            // `Bundle.module` in an *executable* target resolves to a
            // `ShakespeareReader_ShakespeareReader.bundle` sitting beside the binary
            // in `.build/release/`. That is correct under `swift run`; moving the
            // binary alone leaves the resources behind, and CorpusLoader reports
            // that rather than silently showing an empty library.
            resources: [.copy("Resources/Plays")]
        )
    ]
)
