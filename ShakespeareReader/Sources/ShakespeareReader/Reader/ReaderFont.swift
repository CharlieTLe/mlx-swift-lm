import AppKit
import CoreText
import SwiftUI

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
enum ReaderFont: String, CaseIterable, Sendable {
    case system, caslon, baskerville, garamond

    var displayName: String {
        switch self {
        case .system: "System"
        case .caslon: "Caslon"
        case .baskerville: "Baskerville"
        case .garamond: "Garamond"
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
        }
    }

    /// The point-size multiplier that makes this family read at the size SF does.
    ///
    /// Per family, not one constant. Baskerville and Garamond both set noticeably
    /// smaller than SF at the same point size and need the same lift. Big Caslon is
    /// a display cut with a large x-height and would read *bigger*, not equal, if it
    /// were scaled with them.
    var opticalScale: CGFloat {
        switch self {
        case .system: 1.00
        case .caslon: 1.07
        case .baskerville, .garamond: 1.15
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
    /// to be scaled against. Callable off the main actor: `NSFont` itself is not
    /// isolated, only `NSFontManager` is.
    fileprivate static func systemSize(_ style: Font.TextStyle) -> CGFloat {
        NSFont.preferredFont(forTextStyle: nsStyle(style), options: [:]).pointSize
    }

    private static func nsStyle(_ style: Font.TextStyle) -> NSFont.TextStyle {
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

    init(_ font: ReaderFont, installed: Set<String>) {
        let resolved = font.familyName.flatMap { installed.contains($0) ? $0 : nil }
        self.font = font
        self.familyName = resolved
        self.hasItalicFace = resolved == nil || font.hasItalicFace
    }

    static let system = ReaderTypeface(.system, installed: [])

    // MARK: - Roles

    var actSceneHeading: Font {
        // `.system` keeps `.headline` verbatim, so the shipped default is today's
        // look exactly. A custom face cannot use weight (see `ReaderFont`), so a
        // step up in size plus the letterspacing below is what holds the heading
        // apart from the verse under it.
        guard let familyName else { return .headline }
        return custom(familyName, .title3)
    }

    /// Letterspacing for the heading, which is a separate value because tracking is
    /// a `Text` modifier and not something a `Font` carries. Zero for the system
    /// face, again so the default is untouched.
    var actSceneTracking: CGFloat { familyName == nil ? 0 : 0.8 }

    var sceneSetting: Font {
        guard let familyName else { return .subheadline }
        return custom(familyName, .subheadline)
    }

    var speakerHeading: Font {
        // Semibold for the system face only. In a one-face family the caps, the
        // tracking, and `.secondary` are what make this read as a label.
        guard let familyName else { return .caption.weight(.semibold) }
        return custom(familyName, .caption)
    }

    var verse: Font {
        guard let familyName else { return .body }
        return custom(familyName, .body)
    }

    var direction: Font {
        guard let familyName else { return .callout.italic() }
        guard hasItalicFace else { return Self.oblique(familyName, size: size(.callout)) }
        return custom(familyName, .callout).italic()
    }

    /// The gap *between* speeches, tuned against 13pt SF. It is the one padding in
    /// `LineRow` carrying typographic meaning, so it is the one that scales.
    var speechGap: CGFloat { (6 * scale).rounded() }

    /// Serif capitals are already wide, so they need less letterspacing than SF's to
    /// read as a label rather than as a word.
    var speakerTracking: CGFloat { familyName == nil ? 0.6 : 0.4 }

    // MARK: - Sizing

    /// 1.0 whenever the play is being drawn in the system face, whether that is the
    /// reader's choice or a download still in flight.
    private var scale: CGFloat { familyName == nil ? 1 : font.opticalScale }

    /// Rounded, so the verse keeps landing on the baseline grid the number gutter is
    /// aligned to.
    private func size(_ style: Font.TextStyle) -> CGFloat {
        (ReaderFont.systemSize(style) * scale).rounded()
    }

    /// `relativeTo:` is correct here and does not double-scale: it multiplies by the
    /// ratio of the current dynamic type size to `.large`, and macOS is pinned at
    /// `.large` unless something calls `.dynamicTypeSize()`, which nothing here
    /// does.
    private func custom(_ family: String, _ style: Font.TextStyle) -> Font {
        .custom(family, size: size(style), relativeTo: style)
    }

    /// A synthetic italic for a family with no italic cut, which is Big Caslon.
    ///
    /// The shear sits in the matrix's `c` slot only, so advance widths are untouched
    /// and a stage direction wraps exactly where its upright twin would. The size
    /// goes to `CTFontCreateWithFontDescriptor` and stays **out** of the matrix:
    /// CoreText's Swift shim `CTFont.init(_:transform:)` hard-codes size 1.0 and
    /// expects the matrix to carry the scale, so that convenience initializer
    /// silently yields a one-point font.
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
