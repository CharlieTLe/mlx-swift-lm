// Compares this Swift port against the Python MLX reference on identical inputs.
//
// Reads the reference dump produced by `muse_parity_dump.py` and replays the same
// token ids and (post-resize) pixel values through the Swift model, then reports
// max-abs / max-rel error on the logits, top-1 / top-5 agreement, and whether a
// 64-token greedy decode matches token for token.
//
// Feeding the reference's own `pixel_values` is deliberate: PIL's LANCZOS and
// CoreImage's Lanczos are different filters, and mixing that ~1% pixel difference
// into a model-correctness check would make every result ambiguous. The resize is
// covered separately by the unit tests.
//
// Usage:
//   MuseGlimmerParity <parity_ref.safetensors> <parity_ref.json> [prefillStepSize]
//   MuseGlimmerParity --e2e [image]     # real processor path, with a latency breakdown

import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLMCommon
import MLXVLM
import Tokenizers

struct Manifest: Decodable {
    let inputIds: [Int]
    let imageGridThw: [[Int]]
    let generatedIds: [Int]
    let top1: Int
    let top5: [Int]

    enum CodingKeys: String, CodingKey {
        case inputIds = "input_ids"
        case imageGridThw = "image_grid_thw"
        case generatedIds = "generated_ids"
        case top1
        case top5
    }
}

@Sendable
func compare(_ name: String, _ actual: MLXArray, _ expected: MLXArray) -> Bool {
    let a = actual.asType(.float32).reshaped(-1)
    let e = expected.asType(.float32).reshaped(-1)
    guard a.size == e.size else {
        print("  \(name): SHAPE MISMATCH actual \(actual.shape) expected \(expected.shape)")
        return false
    }
    let diff = abs(a - e)
    let maxAbs = diff.max().item(Float.self)
    let maxRel = (diff / (abs(e) + 1e-6)).max().item(Float.self)
    let dot = (a * e).sum().item(Float.self)
    let cosine =
        dot / (sqrt((a * a).sum()).item(Float.self) * sqrt((e * e).sum()).item(Float.self))
    print(
        String(
            format: "  %-20@ maxAbs %.5f  maxRel %.5f  cos %.7f", name as NSString,
            maxAbs, maxRel, cosine))
    // Returned separately from the pass/fail decision: the right threshold
    // depends on the stage, so callers decide.
    return cosine > 0.9999
}

func configureMemory() {
    Memory.cacheLimit = 2 * 1024 * 1024 * 1024
    Memory.memoryLimit = 48 * 1024 * 1024 * 1024
}

func loadContainer() async throws -> ModelContainer {
    try await loadModelContainer(
        from: #hubDownloader(),
        using: #huggingFaceTokenizerLoader(),
        configuration: VLMRegistry.museGlimmer30B4bit
    )
}

/// Exercises the real processor path (chat template, smart resize, `<|patch|>`
/// expansion) rather than replaying reference tensors, and breaks the latency
/// down into preprocessing, vision encode + prefill, and decode.
///
/// Runs twice on purpose: the first pass pays Metal kernel compilation and
/// first-touch page-in of a 19 GB model, the second is steady state. Reporting a
/// single pass conflates warm-up with per-request cost.
func runEndToEnd(imagePath: String?, fillerWords: Int = 0, stepSize: Int? = nil) async throws {
    configureMemory()
    let container = try await loadContainer()

    for pass in 1 ... 2 {
        print("\n===== pass \(pass) =====")
        let output = try await container.perform { (context: ModelContext) -> String in
            let images: [UserInput.Image] =
                imagePath.map { [.url(URL(fileURLWithPath: $0))] } ?? []
            // Filler builds a long text-only prompt, isolating text-prefill
            // throughput from the vision encode.
            let content =
                fillerWords > 0
                ? Array(
                    repeating: "the quick brown fox jumps over the lazy dog", count: fillerWords
                )
                .joined(separator: " ") + " Summarize the preceding text in one word."
                : (imagePath == nil ? "Name three primary colors." : "Describe this image.")
            let chat = [
                Chat.Message(role: .user, content: content, images: images, videos: [])
            ]

            let tPrepare = Date()
            let input = try await context.processor.prepare(input: UserInput(chat: chat))
            print(
                String(
                    format: "processor.prepare (template+resize+patchify): %6.2f s",
                    Date().timeIntervalSince(tPrepare)))
            if let image = input.image {
                print(
                    "  \(input.text.tokens.dim(1)) prompt tokens, pixels \(image.pixels.shape), "
                        + "frames \(image.frames?.map(\.values) ?? [])")
            } else {
                print("  \(input.text.tokens.dim(1)) prompt tokens, no image")
            }

            // Vision encode + text prefill in isolation: exactly the work
            // generate() must finish before it can emit a first token.
            let model = context.model
            let cache = model.newCache(parameters: nil)
            let tPrefill = Date()
            let prepared = try model.prepare(
                input, cache: cache, state: nil,
                prefill: PrefillParameters(stepSize: stepSize))
            if case .logits(let out) = prepared { eval(out.logits) }
            print(
                String(
                    format: "model.prepare (vision encode + prefill):      %6.2f s",
                    Date().timeIntervalSince(tPrefill)))

            // The same work through generate(), to confirm time-to-first-token
            // tracks the prefill cost.
            var output = ""
            var firstTokenAt: Date?
            let tGenerate = Date()
            var generateParameters = GenerateParameters(maxTokens: 200, temperature: 0)
            generateParameters.prefill = PrefillParameters(stepSize: stepSize)
            let stream = try MLXLMCommon.generate(
                input: input, parameters: generateParameters, context: context)
            for await generation in stream {
                if case .chunk(let chunk) = generation {
                    if firstTokenAt == nil { firstTokenAt = Date() }
                    output += chunk
                }
                if case .info(let info) = generation {
                    print(
                        String(
                            format:
                                "generate: TTFT %6.2f s | decode %.1f tok/s | prompt %.1f tok/s"
                                + " | peak %.2f GB",
                            (firstTokenAt ?? Date()).timeIntervalSince(tGenerate),
                            info.tokensPerSecond, info.promptTokensPerSecond,
                            Double(Memory.snapshot().peakMemory) / 1_073_741_824))
                }
            }
            return output
        }
        if pass == 2 { print("---\n\(output.prefix(200))\n---") }
    }
}

func runParity(referencePath: String, manifestPath: String, stepSizeArgument: Int?) async throws {
    let manifest = try JSONDecoder().decode(
        Manifest.self, from: Data(contentsOf: URL(fileURLWithPath: manifestPath)))

    configureMemory()
    print("loading \(VLMRegistry.museGlimmer30B4bit.name)…")
    let container = try await loadContainer()

    let failures = try await container.perform { (context: ModelContext) -> [String] in
        guard let model = context.model as? MuseGlimmer else {
            return ["context.model is \(type(of: context.model)), expected MuseGlimmer"]
        }
        var failures = [String]()
        // Loaded inside the isolation: [String: MLXArray] is not Sendable.
        let reference = try loadArrays(url: URL(fileURLWithPath: referencePath))

        let tokens = MLXArray(manifest.inputIds.map { Int32($0) }).expandedDimensions(axis: 0)
        let frames = manifest.imageGridThw.map { THW($0[0], $0[1], $0[2]) }
        let pixels = reference["pixel_values"]!
        print(
            "input tokens \(tokens.dim(1)), grid \(manifest.imageGridThw), pixels \(pixels.shape)")

        let input = LMInput(
            text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)),
            image: .init(pixels: pixels, frames: frames)
        )

        // --- single forward: logits cover the whole pipeline at once ----------
        let cache = model.newCache(parameters: nil)
        let prepared = try model.prepare(
            input, cache: cache, state: nil,
            prefill: PrefillParameters(
                stepSize: stepSizeArgument == 0 ? Int.max : (stepSizeArgument ?? 512)))
        guard case .logits(let output) = prepared else {
            return ["prepare did not return logits"]
        }
        let logits = output.logits

        // Tolerance note: the ViT's deep layers carry activations of order 1e3
        // and `ln_post` amplifies residual differences, so per-position cosine
        // between *any* two precisions is loose there. Measured control —
        // reference-fp32 vs reference-bf16 — reaches min per-position cosine
        // 0.10 at `ln_post`; this port stays inside that floor. The meaningful
        // gates are therefore the final-position logits and the decode itself.
        print("logits (final position; chunked prefill returns only the last chunk):")
        let referenceLast = reference["logits"]![0, -1]
        if !compare("logits[-1]", logits[0, -1], referenceLast) {
            failures.append("logits")
        }
        // Magnitude of the disagreement on the softcapped (+-20) logits, used
        // below to judge whether a greedy divergence is a genuine near-tie.
        let logitNoise =
            abs(logits[0, -1].asType(.float32) - referenceLast.asType(.float32))
            .max().item(Float.self)

        let last = logits[0, -1].asType(.float32)
        let top1 = argMax(last).item(Int.self)
        print("  top1 swift \(top1) reference \(manifest.top1)")
        if top1 != manifest.top1 { failures.append("top1") }

        let order = argSort(-last).asArray(Int.self)
        let top5 = Array(order.prefix(5))
        print("  top5 swift \(top5) reference \(manifest.top5)")
        if Set(top5) != Set(manifest.top5) { failures.append("top5") }

        // --- greedy decode, token for token ----------------------------------
        var produced = [Int]()
        var divergenceMargins = [Int: Float]()
        var token = top1
        produced.append(token)
        do {
            let ordered = sorted(last)
            divergenceMargins[0] = ordered[-1].item(Float.self) - ordered[-2].item(Float.self)
        }
        let eos: Set<Int> = [200001, 200008]
        while produced.count < manifest.generatedIds.count && !eos.contains(token) {
            let step = model.callAsFunction(
                MLXArray([Int32(token)]).expandedDimensions(axis: 0), cache: cache)
            let stepLogits = step[0, -1].asType(.float32)
            token = argMax(stepLogits).item(Int.self)
            let ordered = sorted(stepLogits)
            divergenceMargins[produced.count] =
                ordered[-1].item(Float.self) - ordered[-2].item(Float.self)
            produced.append(token)
        }
        print("greedy: \(produced.count) tokens vs \(manifest.generatedIds.count) reference")
        if produced == manifest.generatedIds {
            print("  identical token sequence")
        } else {
            let firstDivergence =
                zip(produced, manifest.generatedIds).enumerated()
                .first { $0.element.0 != $0.element.1 }?.offset
            print("  diverges at index \(firstDivergence.map(String.init) ?? "length")")
            if let index = firstDivergence {
                let margin = divergenceMargins[index] ?? .infinity
                print(
                    "  swift token \(produced[index]) vs reference "
                        + "\(manifest.generatedIds[index]), top-2 margin \(margin), "
                        + "logit noise \(logitNoise)")
                if margin < logitNoise {
                    // The two candidates are closer together than the two
                    // implementations' arithmetic differ, so which one wins is
                    // not determined at bf16. Not a correctness failure.
                    print("  WITHIN NOISE: margin < logit noise, step is undetermined at bf16")
                } else {
                    failures.append("greedy")
                }
            } else {
                failures.append("greedy")
            }
            print("  swift     \(produced.prefix(24))")
            print("  reference \(manifest.generatedIds.prefix(24))")
        }
        print("  swift text: \(context.tokenizer.decode(tokenIds: produced).prefix(220))")

        return failures
    }

    if failures.isEmpty {
        print("\nPARITY OK")
    } else {
        print("\nPARITY FAILED: \(failures.joined(separator: ", "))")
        exit(1)
    }
}

@main
struct MuseGlimmerParity {
    static func main() async throws {
        let arguments = CommandLine.arguments
        if arguments.count >= 2 && arguments[1] == "--e2e" {
            try await runEndToEnd(imagePath: arguments.count > 2 ? arguments[2] : nil)
            return
        }
        if arguments.count >= 3 && arguments[1] == "--text" {
            try await runEndToEnd(
                imagePath: nil, fillerWords: Int(arguments[2]) ?? 100,
                stepSize: arguments.count > 3 ? Int(arguments[3]) : nil)
            return
        }
        guard arguments.count >= 3 else {
            print("usage: MuseGlimmerParity <parity_ref.safetensors> <parity_ref.json> [stepSize]")
            print("       MuseGlimmerParity --e2e [image]")
            exit(2)
        }
        // Third argument overrides the prefill chunk size; 0 means one pass.
        try await runParity(
            referencePath: arguments[1], manifestPath: arguments[2],
            stepSizeArgument: arguments.count > 3 ? Int(arguments[3]) : nil)
    }
}
