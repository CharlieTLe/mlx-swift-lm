import SwiftUI

#if os(macOS)
import AppKit
#else
import UIKit
#endif

/// The handful of places where AppKit and UIKit genuinely differ.
///
/// Everything the reader draws is otherwise the same code on both platforms, and the
/// point of this file is to keep it that way: the platform split lives here and in the
/// two places where the *layout* differs (`ContentView`'s container, and
/// `SceneReaderView`'s pointer gestures), not scattered through the view bodies.
///
/// What is deliberately **not** here, because it needed no shim: `.help`,
/// `.controlSize`, `.buttonStyle(.borderless)`, `.menuIndicator`,
/// `.listStyle(.sidebar)`, `.focusEffectDisabled`, `.textSelection`,
/// `.textFieldStyle(.plain)`, `.keyboardShortcut`, and the whole CoreText path in
/// `ReaderFontLibrary`. All of them are available on both.

/// The concrete font class, which is what `ReaderFont` needs to ask the system for a
/// text style's point size. `NSFont.TextStyle` and `UIFont.TextStyle` spell their
/// cases identically, so only the lookup call itself has to be written twice.
#if os(macOS)
typealias PlatformFont = NSFont
#else
typealias PlatformFont = UIFont
#endif

/// Whether shift is held *right now*.
///
/// Read from the event state rather than through `TapGesture().modifiers(.shift)`,
/// which is unreliable, and because SwiftUI does not report modifiers on a move
/// command at all. `false` on iOS: a hardware keyboard can hold shift, but there is
/// no UIKit equivalent of `NSEvent.modifierFlags` to poll outside an event, and the
/// long-press-then-drag in `SceneReaderView` is the touch affordance that replaces
/// shift-click anyway.
var isShiftKeyDown: Bool {
    #if os(macOS)
    NSEvent.modifierFlags.contains(.shift)
    #else
    false
    #endif
}

#if !os(macOS)
/// iOS only. macOS copies through `.onCopyCommand`, which hands the responder
/// chain an `NSItemProvider` rather than writing the pasteboard itself.
func copyToPasteboard(_ text: String) {
    UIPasteboard.general.string = text
}
#endif

/// Whether MLX has a GPU to talk to.
///
/// `false` only on the iOS Simulator, and the reason this has to be asked *before* the
/// fact rather than caught after is that the failure is not recoverable. The first touch
/// of any `MLX.Memory` knob constructs `mlx::core::metal::Device`, which on the Simulator
/// calls `abort()` from C++ by way of `std::__libcpp_verbose_abort`, where no Swift
/// `catch` and no `LoadState.failed` can reach it. Unguarded, the app does not merely
/// fail to annotate on the Simulator: it dies on launch, taking the reader, the corpus
/// and every bit of layout that has nothing to do with the model down with it.
///
/// So the load refuses early and says so in the header instead, which leaves the
/// Simulator good for the UI work it is actually good for.
var hasMLXDevice: Bool {
    #if targetEnvironment(simulator)
    false
    #else
    true
    #endif
}

extension View {
    /// `Menu` ignores `.buttonStyle(.borderless)`, hence `.menuStyle` on macOS. And
    /// `BorderlessButtonMenuStyle` is macOS-only, so iOS keeps the default style and
    /// takes the indicator alone. In a navigation bar the default style is already
    /// the borderless one.
    func borderlessMenu() -> some View {
        #if os(macOS)
        menuStyle(.borderlessButton).menuIndicator(.hidden)
        #else
        menuIndicator(.hidden)
        #endif
    }
}
