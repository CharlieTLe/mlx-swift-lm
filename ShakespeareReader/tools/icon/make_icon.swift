//
// Regenerate the app icon from the Chandos portrait.
//
//     swift tools/icon/make_icon.swift
//
// No shebang, unlike tools/build_corpus.py: swift-format treats one as a comment and
// folds the following line into it, so a `#!/usr/bin/swift` here comes back mangled from
// every `pre-commit run --all`. Invoked through `swift` instead, which is how the corpus
// script is invoked through `python3` anyway.
//
// Writes App/Assets.xcassets/AppIcon.appiconset/*.png (the Xcode app's icon, macOS
// ladder plus the iOS 1024) and Sources/ShakespeareReader/Resources/AppIcon.png (the
// Dock tile an unbundled `swift run` sets at launch). Every output is checked in, so
// a clone builds with no run of this script; it exists to make the crop and the
// masking reproducible, the same way tools/build_corpus.py does for the corpus.
//
// Swift rather than Python, breaking the language precedent in this directory
// deliberately: build_corpus.py is standard-library-only so it needs no install step,
// and the equivalent image script would need Pillow. CoreGraphics is already on every
// machine that can build this app, and it is what ships the high-quality downsampling
// the 16pt icon needs.

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

// MARK: - Geometry

/// The square cut from the 960x1224 source, in source pixels.
///
/// Chosen against a contact sheet of four candidates at 16, 32 and 128pt. The full
/// canvas leaves the head in the upper third and dissolves to a dark smudge at 16pt;
/// a tight crop on the face alone loses the ruff collar, which is the one high-contrast
/// shape that still reads when the face does not. This keeps head, collar and a hint of
/// the doublet, and needs only a 1.14x upscale to fill the 824pt body.
let crop = CGRect(x: 140, y: 60, width: 720, height: 720)

/// Fraction of the icon canvas the rounded body occupies.
///
/// Apple's macOS icon grid: an 824pt body centred in a 1024pt canvas, with the
/// remaining margin carrying the shadow. iOS is full-bleed instead — the system
/// applies its own mask — so this applies to the macOS ladder only.
let bodyFraction = 824.0 / 1024.0

/// Superellipse exponent for the corner shape.
///
/// `CGPath(roundedRect:)` is circular arcs, which reads visibly pinched at the corners
/// next to a system icon. Apple's shape is a squircle: continuous curvature, no flat
/// sides. n = 5 is the standard model of it, and sampling the curve densely is simpler
/// to get right than the Bézier control-point approximation.
let superellipseExponent = 5.0

// MARK: - Paths

let toolURL = URL(fileURLWithPath: #filePath)
let packageRoot = toolURL.deletingLastPathComponent()  // tools/icon
    .deletingLastPathComponent()  // tools
    .deletingLastPathComponent()  // ShakespeareReader
let sourceURL = toolURL.deletingLastPathComponent().appendingPathComponent("chandos-portrait.jpg")
let iconSetURL = packageRoot.appendingPathComponent("App/Assets.xcassets/AppIcon.appiconset")
let dockIconURL = packageRoot.appendingPathComponent(
    "Sources/ShakespeareReader/Resources/AppIcon.png")

// MARK: - Drawing

/// The squircle, as a path inscribed in `rect`.
///
/// Sampled rather than fitted: 720 points is finer than a pixel at 1024pt, so the
/// polygon is indistinguishable from the curve once CoreGraphics antialiases it.
func superellipsePath(in rect: CGRect, exponent: Double) -> CGPath {
    let path = CGMutablePath()
    let (a, b) = (rect.width / 2, rect.height / 2)
    let center = CGPoint(x: rect.midX, y: rect.midY)
    let samples = 720
    for i in 0 ..< samples {
        let t = 2 * Double.pi * Double(i) / Double(samples)
        // |cos t|^(2/n) with the sign of cos t, which traces the superellipse.
        let x = pow(abs(cos(t)), 2 / exponent) * (cos(t) < 0 ? -1 : 1) * a
        let y = pow(abs(sin(t)), 2 / exponent) * (sin(t) < 0 ? -1 : 1) * b
        let point = CGPoint(x: center.x + x, y: center.y + y)
        if i == 0 { path.move(to: point) } else { path.addLine(to: point) }
    }
    path.closeSubpath()
    return path
}

func context(size: Int) -> CGContext {
    guard
        let context = CGContext(
            data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { fatalError("could not create a \(size)x\(size) bitmap context") }
    // The default is `.default`, which is bilinear and leaves the 16pt icon mushy.
    context.interpolationQuality = .high
    return context
}

/// The macOS icon at `size` pixels: the art, masked to the squircle, inset in the
/// canvas, over a soft shadow.
func macIcon(art: CGImage, size: Int) -> CGImage {
    let canvas = Double(size)
    let body = (canvas * bodyFraction).rounded()
    let origin = ((canvas - body) / 2).rounded()
    let bodyRect = CGRect(x: origin, y: origin, width: body, height: body)
    let path = superellipsePath(in: bodyRect, exponent: superellipseExponent)

    let context = context(size: size)
    // Drawn as a filled shape first so the shadow comes off the silhouette, not off the
    // art's bounding box — clipping to the path and then drawing the image would put the
    // shadow behind opaque pixels where it is never seen.
    context.saveGState()
    context.setShadow(
        // Negative dy: this context is y-up, and the shadow belongs below the body.
        offset: CGSize(width: 0, height: -canvas * 0.012),
        blur: canvas * 0.022,
        color: CGColor(gray: 0, alpha: 0.35))
    context.addPath(path)
    context.setFillColor(CGColor(gray: 0, alpha: 1))
    context.fillPath()
    context.restoreGState()

    context.saveGState()
    context.addPath(path)
    context.clip()
    context.draw(art, in: bodyRect)
    context.restoreGState()

    guard let image = context.makeImage() else { fatalError("macOS icon \(size) failed") }
    return image
}

/// The iOS icon: full-bleed and opaque. An alpha channel here is an App Store
/// validation error, and the rounding is the system's job.
func iOSIcon(art: CGImage, size: Int) -> CGImage {
    guard
        let context = CGContext(
            data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { fatalError("could not create the iOS bitmap context") }
    context.interpolationQuality = .high
    context.draw(art, in: CGRect(x: 0, y: 0, width: Double(size), height: Double(size)))
    guard let image = context.makeImage() else { fatalError("iOS icon \(size) failed") }
    return image
}

func write(_ image: CGImage, to url: URL) {
    guard
        let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { fatalError("could not open \(url.path) for writing") }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
        fatalError("could not write \(url.path)")
    }
    print("  \(url.lastPathComponent)  \(image.width)x\(image.height)")
}

// MARK: - Run

guard let data = try? Data(contentsOf: sourceURL),
    let source = CGImageSourceCreateWithData(data as CFData, nil),
    let full = CGImageSourceCreateImageAtIndex(source, 0, nil)
else { fatalError("could not read \(sourceURL.path)") }

guard let art = full.cropping(to: crop) else {
    fatalError("crop \(crop) does not lie inside the \(full.width)x\(full.height) source")
}

try? FileManager.default.createDirectory(at: iconSetURL, withIntermediateDirectories: true)

/// One entry per `Contents.json` slot. The 32, 256 and 512 pixel sizes each appear
/// twice — as @1x of one point size and @2x of the one below — and get a file each
/// rather than two entries sharing a name, which is the layout Xcode itself writes.
let macLadder: [(points: Int, scale: Int)] = [
    (16, 1), (16, 2), (32, 1), (32, 2), (128, 1), (128, 2), (256, 1), (256, 2), (512, 1), (512, 2),
]

print("app icon from \(sourceURL.lastPathComponent) (\(full.width)x\(full.height))")
print("appiconset:")
for (points, scale) in macLadder {
    let suffix = scale == 1 ? "" : "@\(scale)x"
    let name = "icon_\(points)x\(points)\(suffix).png"
    write(macIcon(art: art, size: points * scale), to: iconSetURL.appendingPathComponent(name))
}
write(iOSIcon(art: art, size: 1024), to: iconSetURL.appendingPathComponent("icon_ios_1024.png"))

// 512 rather than 1024: the largest tile macOS asks an unbundled process for is
// Mission Control's, well under 256pt, so the extra megabyte in the resource bundle
// buys nothing.
print("dock tile:")
write(macIcon(art: art, size: 512), to: dockIconURL)
