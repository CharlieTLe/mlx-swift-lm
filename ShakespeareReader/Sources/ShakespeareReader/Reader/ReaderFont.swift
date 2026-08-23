import CoreText
import SwiftUI

#if os(macOS)
import AppKit
#else
import UIKit
#endif

/// The face the play text is set in.
///
/// The app is otherwise built to imitate a printed edition — hanging verse indents,
/// stage directions in italic and indented further, a line number every fifth line
/// in a gutter — and SF is the one part of it that does not read like a book.
///
/// Two facts about what macOS actually ships shape the rest of this file:
///
/// - **Big Caslon is a single face** (`BigCaslon-Medium`): no italic, no bold, and
///   macOS ships no other Caslon. `Font.custom(...).weight(.semibold)` resolves to
///   the nearest *available* face, so asking for weight gives Baskerville real
///   contrast and Big Caslon nothing at all — one code path rendering two different
///   visual hierarchies. So weight is not a usable channel here; size, tracking,
///   italic, and colour carry the hierarchy instead.
/// - **Garamond is not installed.** It is in Apple's downloadable font asset
///   catalog, which `ReaderFontLibrary` fetches on demand.
///
/// iOS ships a different set, which is why `offered` exists rather than
/// `allCases`: of the three named faces above only Baskerville is there, Big Caslon
/// has no iOS counterpart at all, and Garamond is not in an iOS downloadable font
/// catalog. Hoefler Text and Palatino take their places, and both were picked from the
/// runtime's `System/Library/Fonts/AppFonts`, which is the set actually exposed to
/// apps, and the reason Iowan Old Style is not here. It was the obvious third choice,
/// it ships with iOS, and `CTFontManagerCopyAvailableFontFamilyNames` does not return
/// it, so the row rendered in SF and offered a download that does not exist.
///
/// The cases are *not* renamed per platform. An honest family name beats a silent
/// substitution, and a `readerFont` of `caslon` carried over from a Mac stays safe
/// because `ReaderTypeface.init` falls back to the system face for a family that is
/// not installed.
enum ReaderFont: String, CaseIterable, Sendable {
    case system, caslon, baskerville, garamond, hoeflerText, palatino

    /// What the typeface menu lists, which is only the families the running platform
    /// can actually resolve. `allCases` stays the full set, because it is also the
    /// decoding surface for a stored preference written on the other platform.
    static var offered: [ReaderFont] {
        #if os(macOS)
        [.system, .caslon, .baskerville, .garamond]
        #else
        [.system, .baskerville, .hoeflerText, .palatino]
        #endif
    }

    var displayName: String {
        switch self {
        case .system: "System"
        case .caslon: "Caslon"
        case .baskerville: "Baskerville"
        case .garamond: "Garamond"
        case .hoeflerText: "Hoefler Text"
        case .palatino: "Palatino"
        }
    }

    /// The CoreText family name, or `nil` for the system face, which is the one this
    /// app has no business naming: `Font.body` already resolves it, including on a
    /// machine where SF has been replaced.
    var familyName: String? {
        switch self {
        case .system: nil
        case .caslon: "Big Caslon"
        case .baskerville: "Baskerville"
        case .garamond: "Garamond"
        case .hoeflerText: "Hoefler Text"
        case .palatino: "Palatino"
        }
    }

    /// The point-size multiplier that makes this family read at the size SF does.
    ///
    /// Per family, not one constant. Baskerville and Garamond both set noticeably
    /// smaller than SF at the same point size and need the same lift. Big Caslon is
    /// a display cut with a large x-height and would read *bigger*, not equal, if it
    /// were scaled with them. Hoefler Text is a text cut with a small x-height and
    /// wants nearly as much lift as Baskerville; Palatino has a large x-height and
    /// needs less.
    var opticalScale: CGFloat {
        switch self {
        case .system: 1.00
        case .caslon: 1.07
        case .baskerville, .garamond: 1.15
        case .hoeflerText: 1.10
        case .palatino: 1.05
        }
    }

    /// Whether the family ships a real italic cut.
    ///
    /// CoreText answers by trying the conversion rather than by being told:
    /// `Baskerville` yields `Baskerville-Italic`, `Big Caslon` yields nil. This is
    /// what decides whether `ReaderTypeface.direction` can ask for `.italic()` or
    /// has to shear the face by hand.
    var hasItalicFace: Bool {
        guard let familyName else { return true }
        let base = CTFontCreateWithName(familyName as CFString, 12, nil)
        return CTFontCreateCopyWithSymbolicTraits(
            base, 0, nil, .traitItalic, .traitItalic) != nil
    }

    /// CoreText rather than `NSFontManager.shared`, which is `@MainActor` by way of
    /// its `NSMenuItemValidation` conformance and so out of reach of `--selftest`,
    /// which runs synchronously and off the main actor.
    static func installedFamilyNames() -> Set<String> {
        let names = CTFontManagerCopyAvailableFontFamilyNames() as NSArray
        return Set(names.compactMap { $0 as? String })
    }

    /// The system's own point size for a text style, which is what a custom face has
    /// to be scaled against. Callable off the main actor: neither `NSFont` nor
    /// `UIFont` is isolated, only `NSFontManager` is.
    ///
    /// Asked for **at `.large`** rather than at the current content size category,
    /// which is the whole reason this is not a one-liner. `custom(_:_:)` below hands
    /// the result to `Font.custom(_:size:relativeTo:)`, which scales it by the ratio
    /// of the current dynamic type size to `.large`, so a size that already had
    /// Dynamic Type applied would be scaled by it twice. macOS never notices, being
    /// pinned at `.large`; on iOS, where Dynamic Type is live, it is the difference
    /// between the verse growing once and growing quadratically.
    fileprivate static func systemSize(_ style: Font.TextStyle) -> CGFloat {
        #if os(macOS)
        PlatformFont.preferredFont(forTextStyle: platformStyle(style), options: [:])
            .pointSize
        #else
        PlatformFont.preferredFont(
            forTextStyle: platformStyle(style),
            compatibleWith: UITraitCollection(preferredContentSizeCategory: .large)
        ).pointSize
        #endif
    }

    /// The same measurement **at the category actually in force**, which is what the
    /// *system* face has to be scaled against at a non-default `ReaderTextSize`.
    ///
    /// Two measurements and not one, because the two font constructors behave
    /// differently and neither is documented as doing so:
    ///
    /// - `Font.custom(_:size:relativeTo:)` scales the size it is handed, so it wants
    ///   the `.large` measurement above.
    /// - `Font.system(size:)` does **not**. Measured on the Simulator across the
    ///   `--selftest` ladder: at `.large` the verse read 22.3pt tall at Default and
    ///   24.0pt at the Large step; at xxxLarge the Default reading grew to 30.3pt and
    ///   the Large one was still 24.0pt. A fixed-size system font is frozen, so
    ///   choosing any size step would have taken the reader's accessibility setting
    ///   away from them. The scaling therefore has to happen here, from the category
    ///   in force.
    ///
    /// Measuring each style separately rather than scaling one body size also means the
    /// system face keeps each style's own metric curve, so the caption-derived speaker
    /// heading holds its proportion against the verse at accessibility sizes instead of
    /// growing at the body's rate.
    fileprivate static func systemSize(
        _ style: Font.TextStyle, at dynamicTypeSize: DynamicTypeSize
    ) -> CGFloat {
        #if os(macOS)
        // macOS has no Dynamic Type at all: `DynamicTypeSize` there is always `.large`,
        // so this is the measurement above, and the two paths cannot diverge.
        systemSize(style)
        #else
        PlatformFont.preferredFont(
            forTextStyle: platformStyle(style),
            compatibleWith: UITraitCollection(
                preferredContentSizeCategory: contentSizeCategory(dynamicTypeSize))
        ).pointSize
        #endif
    }

    #if !os(macOS)
    /// `DynamicTypeSize` and `UIContentSizeCategory` are the same ladder in two types,
    /// and SwiftUI ships no conversion between them.
    private static func contentSizeCategory(
        _ size: DynamicTypeSize
    ) -> UIContentSizeCategory {
        switch size {
        case .xSmall: .extraSmall
        case .small: .small
        case .medium: .medium
        case .large: .large
        case .xLarge: .extraLarge
        case .xxLarge: .extraExtraLarge
        case .xxxLarge: .extraExtraExtraLarge
        case .accessibility1: .accessibilityMedium
        case .accessibility2: .accessibilityLarge
        case .accessibility3: .accessibilityExtraLarge
        case .accessibility4: .accessibilityExtraExtraLarge
        case .accessibility5: .accessibilityExtraExtraExtraLarge
        @unknown default: .large
        }
    }
    #endif

    /// `NSFont.TextStyle` and `UIFont.TextStyle` spell every case the same, so this
    /// mapping is written once against `PlatformFont`.
    private static func platformStyle(_ style: Font.TextStyle) -> PlatformFont.TextStyle {
        switch style {
        case .largeTitle: .largeTitle
        case .title: .title1
        case .title2: .title2
        case .title3: .title3
        case .headline: .headline
        case .subheadline: .subheadline
        case .body: .body
        case .callout: .callout
        case .footnote: .footnote
        case .caption: .caption1
        case .caption2: .caption2
        @unknown default: .body
        }
    }
}

/// The fonts the play text is actually drawn with.
///
/// **Roles, not styles.** One property per place that draws play text, rather than a
/// general `font(_ style:weight:italic:)`. Every "what happens in a family with only
/// one face" decision then lives here, in one place, next to the comment explaining
/// it — and `LineRow` stops restating styling it does not own.
struct ReaderTypeface: Equatable, Sendable {
    let font: ReaderFont

    /// `nil` for `.system` **and** for a family that is not installed yet, which is
    /// what Garamond looks like until its asset lands.
    ///
    /// Resolving availability here rather than leaving it to `Font.custom` is
    /// deliberate: `Font.custom`'s fallback for an unknown family is real but
    /// undocumented, and it would make a typo'd family name indistinguishable from a
    /// font that has simply not downloaded yet. With this nil until the family is
    /// present the not-yet-downloaded state is explicit in the type, and when the
    /// download lands the value changes identity and the reader re-renders with no
    /// extra wiring.
    let familyName: String?

    /// Probed once, here, rather than on every row: `direction` is asked for at each
    /// stage direction in the scene.
    private let hasItalicFace: Bool

    /// The reader's size step, carried here rather than in a second environment key.
    ///
    /// `SceneReaderView` re-anchors the scroll position off `.onChange(of: typeface)`,
    /// precisely because changing the type shifts point-based scroll offsets. Putting
    /// the step on this already-`Equatable` value gets that re-anchor for free; a
    /// separate key would not fire it, and every size change would drift the reader off
    /// their line.
    ///
    /// Named `textSize` and not `size`, so it cannot be confused with the `size(_:)`
    /// method below, which answers in points.
    let textSize: ReaderTextSize

    /// The content size category in force, which only the system face needs and only at
    /// a non-default `textSize`: `Font.system(size:)` is fixed, so nothing else would
    /// scale the verse when the reader moves the Larger Text slider. See
    /// `ReaderFont.systemSize(_:at:)`, which is where that was measured.
    ///
    /// Stored here, alongside `textSize`, for the same reason: `SceneReaderView`
    /// re-anchors off `.onChange(of: typeface)`, and a category change moves the type
    /// exactly as a size step does. Always `.large` on macOS.
    let dynamicTypeSize: DynamicTypeSize

    /// No default argument for `textSize` or `dynamicTypeSize`. There are exactly two
    /// construction sites, so being explicit costs two arguments and buys a compiler
    /// error at any new one — and a silently defaulted `dynamicTypeSize` is precisely
    /// the bug that froze the accessibility control before it was threaded through.
    init(
        _ font: ReaderFont, textSize: ReaderTextSize, dynamicTypeSize: DynamicTypeSize,
        installed: Set<String>
    ) {
        let resolved = font.familyName.flatMap { installed.contains($0) ? $0 : nil }
        self.font = font
        self.familyName = resolved
        self.hasItalicFace = resolved == nil || font.hasItalicFace
        self.textSize = textSize
        self.dynamicTypeSize = dynamicTypeSize
    }

    static let system = ReaderTypeface(
        .system, textSize: .default, dynamicTypeSize: .large, installed: [])

    // MARK: - Roles

    var actSceneHeading: Font {
        // `.system` keeps `.headline` verbatim, so the shipped default is today's
        // look exactly. A custom face cannot use weight (see `ReaderFont`), so a
        // step up in size plus the letterspacing below is what holds the heading
        // apart from the verse under it.
        guard let familyName else {
            // `.semibold` restated by hand: it is part of `.headline` and not part of
            // the point size, so a bare `Font.system(size:)` would draw the act
            // heading in the same weight as the verse.
            return textSize.isDefault ? .headline : system(.headline, weight: .semibold)
        }
        return custom(familyName, .title3)
    }

    /// Letterspacing for the heading, which is a separate value because tracking is
    /// a `Text` modifier and not something a `Font` carries. Zero for the system
    /// face, again so the default is untouched.
    var actSceneTracking: CGFloat {
        (familyName == nil ? 0 : 0.8) * textSize.multiplier
    }

    var sceneSetting: Font {
        guard let familyName else {
            return textSize.isDefault ? .subheadline : system(.subheadline)
        }
        return custom(familyName, .subheadline)
    }

    var speakerHeading: Font {
        // Semibold for the system face only. In a one-face family the caps, the
        // tracking, and `.secondary` are what make this read as a label.
        guard let familyName else {
            return textSize.isDefault
                ? .caption.weight(.semibold) : system(.caption, weight: .semibold)
        }
        return custom(familyName, .caption)
    }

    var verse: Font {
        guard let familyName else { return textSize.isDefault ? .body : system(.body) }
        return custom(familyName, .body)
    }

    var direction: Font {
        guard let familyName else {
            return textSize.isDefault ? .callout.italic() : system(.callout, italic: true)
        }
        guard hasItalicFace else { return Self.oblique(familyName, size: size(.callout)) }
        return custom(familyName, .callout).italic()
    }

    /// The gap *between* speeches, tuned against 13pt SF. It is the one padding in
    /// `LineRow` carrying typographic meaning, so it is the one that scales.
    var speechGap: CGFloat { (6 * scale).rounded() }

    /// Serif capitals are already wide, so they need less letterspacing than SF's to
    /// read as a label rather than as a word.
    var speakerTracking: CGFloat {
        (familyName == nil ? 0.6 : 0.4) * textSize.multiplier
    }

    /// The stage-direction indent, which lives here rather than in `LineRow` because a
    /// printed edition sets it in ems: 28pt against 17pt type is not the same indent as
    /// 28pt against 26pt type.
    var directionIndent: CGFloat { (28 * scale).rounded() }

    /// The line-number gutter's font, which stays on the system face for its
    /// monospaced digits but does scale with the reader's step: a 10pt number beside
    /// 20pt verse reads as a bug rather than as restraint.
    var gutterFont: Font {
        textSize.isDefault ? .caption2.monospacedDigit() : system(.caption2).monospacedDigit()
    }

    /// The gutter's width, which moves with `gutterFont` and never on its own: the
    /// width is a budget for three digits' advances, so scaling the font without it is
    /// the clipping case.
    ///
    /// `textSize.multiplier` and not `scale`, because the gutter is not in the reader's
    /// chosen family and so has no optical correction to apply.
    var gutterWidth: CGFloat { (30 * textSize.multiplier).rounded() }

    // MARK: - Sizing

    /// The product of two independent corrections: `opticalScale` makes a family read
    /// at the size SF does and is none of the reader's business, while
    /// `textSize.multiplier` is the only one they chose. The optical half stays 1.0 for
    /// the system face and for a download in flight; the reader's half applies in both
    /// cases, which is the feature.
    private var scale: CGFloat {
        (familyName == nil ? 1 : font.opticalScale) * textSize.multiplier
    }

    /// Rounded, so the verse keeps landing on the baseline grid the number gutter is
    /// aligned to.
    private func size(_ style: Font.TextStyle) -> CGFloat {
        (ReaderFont.systemSize(style) * scale).rounded()
    }

    /// `size(_:)`'s twin for the system face, which has to do its own Dynamic Type
    /// scaling. Same rounding, different measurement — see `system(_:weight:italic:)`.
    private func systemFaceSize(_ style: Font.TextStyle) -> CGFloat {
        (ReaderFont.systemSize(style, at: dynamicTypeSize) * scale).rounded()
    }

    /// `relativeTo:` is what makes the verse follow Dynamic Type, and it does not
    /// double-scale *because* `systemSize(_:)` is measured at `.large`. See the note
    /// there, which is the load-bearing half of this pair.
    private func custom(_ family: String, _ style: Font.TextStyle) -> Font {
        .custom(family, size: size(style), relativeTo: style)
    }

    /// The system face at a non-default step, which is the only reason this exists:
    /// `Font.body` is a *text style*, not a point size, and there is no arithmetic to
    /// do to it. Two things about it are worth knowing.
    ///
    /// - It is not what the default path uses, and the two are not interchangeable.
    ///   `Font.body` and `Font.system(size: 17)` are different values even where they
    ///   resolve to the same 17 points — the first follows Dynamic Type and the second
    ///   does not — so every role short-circuits to the bare style at `.default` rather
    ///   than routing through here with a multiplier of 1.
    /// - The size comes from `systemFaceSize(_:)` and **not** from `size(_:)`, because
    ///   `Font.system(size:)` will not scale it afterwards. That asymmetry with
    ///   `custom(_:_:)` is the whole content of `ReaderFont.systemSize(_:at:)`'s note.
    ///
    /// Naming SF's private dot-prefixed family to `Font.custom` in order to get
    /// `relativeTo:` here instead is rejected for the reason in `familyName`'s comment:
    /// this app has no business naming the system family.
    private func system(
        _ style: Font.TextStyle, weight: Font.Weight = .regular, italic: Bool = false
    ) -> Font {
        let font = Font.system(size: systemFaceSize(style), weight: weight)
        return italic ? font.italic() : font
    }

    /// A synthetic italic for a family with no italic cut, which is Big Caslon,
    /// and only Big Caslon: Baskerville, Garamond, Hoefler Text and Palatino
    /// all ship real italics, so `hasItalicFace` keeps them out of here.
    ///
    /// The shear sits in the matrix's `c` slot only, so advance widths are untouched
    /// and a stage direction wraps exactly where its upright twin would. The size
    /// goes to `CTFontCreateWithFontDescriptor` and stays **out** of the matrix:
    /// CoreText's Swift shim `CTFont.init(_:transform:)` hard-codes size 1.0 and
    /// expects the matrix to carry the scale, so that convenience initializer
    /// silently yields a one-point font.
    ///
    /// A `Font` built from a `CTFont` does not participate in Dynamic Type, so this is
    /// the one role that ignores it — it still follows the reader's size step, which
    /// arrives baked into `size`. Harmless: Big Caslon is macOS-only, and macOS is
    /// pinned at `.large`.
    private static func oblique(_ family: String, size: CGFloat) -> Font {
        let descriptor = CTFontDescriptorCreateWithAttributes(
            [kCTFontFamilyNameAttribute: family] as CFDictionary)
        var shear = CGAffineTransform(a: 1, b: 0, c: 0.2126, d: 1, tx: 0, ty: 0)  // ~12°
        return Font(CTFontCreateWithFontDescriptor(descriptor, size, &shear))
    }
}

extension EnvironmentValues {
    /// Injected on `SceneReaderView` and nowhere else, and that single injection
    /// point *is* the scope of this feature: the navigator, the commentary pane, the
    /// header chrome, and the status strip all stay on the system face. A `let`
    /// parameter threaded through initializers — how `collapsedActs` is passed —
    /// could not express that boundary nearly as well, which is the reason for the
    /// departure.
    @Entry var readerTypeface: ReaderTypeface = .system
}
