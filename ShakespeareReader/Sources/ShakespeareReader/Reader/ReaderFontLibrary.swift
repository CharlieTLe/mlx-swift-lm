import CoreText
import SwiftUI

/// Which of the reader's faces are installed, plus the on-demand download for the
/// one that is not.
///
/// Garamond does not ship with macOS, but it *is* in Apple's downloadable font asset
/// catalog (`/System/Library/AssetsV2/com_apple_MobileAsset_Font7`). Matching a
/// descriptor against it with
/// `CTFontDescriptorMatchFontDescriptorsWithProgressHandler` activates the asset in
/// **session** scope: the family becomes available to this process as soon as the
/// match finishes, and a later launch finds it already there.
@MainActor @Observable
final class ReaderFontLibrary {
    private(set) var installed: Set<String> = ReaderFont.installedFamilyNames()
    private(set) var downloading: Set<ReaderFont> = []
    private(set) var failed: [ReaderFont: String] = [:]

    func typeface(for font: ReaderFont) -> ReaderTypeface {
        ReaderTypeface(font, installed: installed)
    }

    func isAvailable(_ font: ReaderFont) -> Bool {
        guard let family = font.familyName else { return true }
        return installed.contains(family)
    }

    /// Not `async`, because there is nothing for the caller to await: a menu action
    /// starts this, and the result reaches the reader through `installed` changing,
    /// which changes the `ReaderTypeface` identity and re-renders the scene. No
    /// continuation, so no resume-once guard, no lock.
    func download(_ font: ReaderFont) {
        guard let family = font.familyName, !installed.contains(family),
            !downloading.contains(font)
        else { return }

        // Cleared synchronously here, and never cleared on success. Separate
        // `Task { @MainActor }` hops are not ordered relative to each other, so a
        // late `.didFailWithError` write has to still be correct on its own: with
        // the clear on this side of the hop it is, and a success never writes an
        // error at all.
        failed[font] = nil
        downloading.insert(font)

        let descriptor = CTFontDescriptorCreateWithAttributes(
            [kCTFontFamilyNameAttribute: family] as CFDictionary)

        // The same shape as `AnnotationService.load`'s progress callback: an
        // escaping handler from a queue we do not own, hopping to the main actor to
        // write state. This one runs on a private serial queue CoreText owns, and
        // must return `true` to let the download continue.
        //
        // Deliberately *not* observing `kCTFontManagerRegisteredFontsChangedNotification`:
        // session-scope activation posts that to the **distributed** centre, not to
        // `NotificationCenter.default`, so it would never arrive. `.didFinish`
        // refreshes `installed` instead.
        _ = CTFontDescriptorMatchFontDescriptorsWithProgressHandler(
            [descriptor] as CFArray, nil
        ) { state, info in
            switch state {
            case .didFailWithError:
                // Documented as "may be called multiple times", and it is **not**
                // terminal — `.didFinish` still follows — so this only records the
                // message and never ends the download.
                let error = (info as NSDictionary)[kCTFontDescriptorMatchingError]
                let message =
                    (error as? NSError)?.localizedDescription
                    ?? "\(family) could not be downloaded."
                Task { @MainActor in self.failed[font] = message }
            case .didFinish:
                Task { @MainActor in
                    self.installed = ReaderFont.installedFamilyNames()
                    self.downloading.remove(font)
                }
            default:
                break
            }
            return true
        }
    }
}
