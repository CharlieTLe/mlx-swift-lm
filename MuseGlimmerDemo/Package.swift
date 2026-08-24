// swift-tools-version: 6.1

import PackageDescription

// A standalone SwiftUI macOS app for running Muse-Glimmer fully on-device.
//
// This is a sibling package with a *local path* dependency on the checkout, so it
// picks up `MuseGlimmer.swift` before it is merged upstream. The apps in
// mlx-swift-examples reach mlx-swift-lm through a remote package reference and
// therefore cannot see unmerged model code without editing their pbxproj.
//
// Run with:  cd MuseGlimmerDemo && swift run -c release MuseGlimmerDemo
let package = Package(
    name: "MuseGlimmerDemo",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(name: "mlx-swift-lm", path: ".."),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "MuseGlimmerDemo",
            dependencies: [
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MuseGlimmerDemo"
        ),
        // Numerical parity harness against the Python MLX reference. Not part of
        // the app; see Sources/MuseGlimmerParity/main.swift.
        .executableTarget(
            name: "MuseGlimmerParity",
            dependencies: [
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MuseGlimmerParity"
        ),
    ]
)
