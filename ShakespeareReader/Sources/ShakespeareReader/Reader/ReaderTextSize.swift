import CoreGraphics

/// How large the play text is set, as a multiplier over the system's own point size
/// for each text style.
///
/// A multiplier and not a point size, so the setting composes with everything else
/// rather than replacing it: the per-family `opticalScale` still applies, Dynamic Type
/// still applies on iOS, and every role keeps the relative hierarchy `ReaderTypeface`
/// gives it. There is no separate ladder per platform the way `ReaderFont.offered`
/// splits per platform — the faces differ because installed families differ, and a
/// multiplier has no such dependency.
///
/// **Five steps and not a slider.** The control lives in the typeface menu, and a menu
/// row on macOS *is* an `NSMenuItem`, which takes a title and one image: it hosts
/// neither a slider nor a live preview of the size it would set. Discrete steps are
/// also what the reader can name afterwards — "Larger" is a thing to go back to, a
/// slider position is not. Five is enough to reach a comfortable size from either
/// direction without the menu turning into a ruler.
enum ReaderTextSize: String, CaseIterable, Sendable {
    // Declaration order is menu order, and `--selftest` asserts the multipliers rise
    // along it.
    case small, `default`, large, larger, largest

    var displayName: String {
        switch self {
        case .small: "Small"
        case .default: "Default"
        case .large: "Large"
        case .larger: "Larger"
        case .largest: "Largest"
        }
    }

    /// `default` is **exactly** 1.0, not 0.99 or 1.01: it is what makes the shipped
    /// rendering reproducible. Every point measure in `ReaderTypeface` runs through
    /// this, so a neutral step that is merely close would move the verse for every
    /// reader who has never opened the menu.
    ///
    /// The rest are geometric-ish rather than evenly spaced, because the steps a
    /// reader wants get coarser as the type gets bigger: 15% is a visible change at
    /// 13pt and a small one at 20pt.
    var multiplier: CGFloat {
        switch self {
        case .small: 0.85
        case .default: 1.00
        case .large: 1.15
        case .larger: 1.30
        case .largest: 1.50
        }
    }

    /// Named rather than compared inline, because "is this the neutral step" is asked
    /// at six call sites and each one is a promise the default rendering is untouched.
    var isDefault: Bool { self == .default }
}
