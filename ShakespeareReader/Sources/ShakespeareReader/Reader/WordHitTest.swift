import CoreText
import Foundation

#if os(macOS)
import AppKit
#else
import UIKit
#endif

/// A pointer position over a line of the play, turned into a character index.
///
/// SwiftUI's `Text` cannot be asked what is under a point, so the line is laid out a
/// second time with CoreText and hit-tested there. That only works while the second
/// layout matches the first, which is what `ReaderTypeface.versePlatformFont` and its
/// twin are for: same family, same point size, same wrap width. Three things make the
/// match tractable here rather than hopeful.
///
/// - The verse carries no `.kerning` and no `.tracking` — only the speaker and act
///   headings do — so there is no letterspacing to model.
/// - macOS pins Dynamic Type at `.large`, so the resolved point size is exact.
/// - A near-miss is *visible*: `LineRow` marks the word it resolved and the menu names
///   it, so drift shows as a mark beside the word rather than as a silent wrong answer.
///
/// `NSAttributedString.Key.font` is spelled `"NSFont"`, which is exactly
/// `kCTFontAttributeName`, so one attributed string serves both AppKit and CoreText and
/// nothing has to be bridged here.
enum WordHitTest {

    /// The character under `point`, which is measured from the **top-leading corner of
    /// the text itself** — not of the row, and not of the padding around it.
    ///
    /// A point past the line's typographic width answers `nil` rather than the last
    /// character. The verse `Text` spans the full pane width, so without that the mark
    /// would stick to the final word of every line across the whole empty right-hand
    /// half of the reader.
    static func characterIndex(
        at point: CGPoint, in text: String, font: PlatformFont, width: CGFloat
    ) -> String.Index? {
        guard point.x >= 0, point.y >= 0,
            let laid = layout(text, font: font, width: width)
        else { return nil }

        // Bands run from each baseline's ascent down to the next line's, so the gap
        // between two visual lines belongs to the line above it.
        var found: (line: CTLine, baseline: CGPoint)?
        for (line, baseline) in laid {
            var ascent: CGFloat = 0
            CTLineGetTypographicBounds(line, &ascent, nil, nil)
            if found == nil || point.y >= baseline.y - ascent {
                found = (line, baseline)
            }
        }
        guard let (line, baseline) = found else { return nil }

        let x = point.x - baseline.x
        guard x >= 0, x <= CTLineGetTypographicBounds(line, nil, nil, nil) else {
            return nil
        }

        let boundary = CTLineGetStringIndexForPosition(line, CGPoint(x: x, y: 0))
        guard boundary != kCFNotFound else { return nil }
        // `CTLineGetStringIndexForPosition` answers with the nearest *insertion point*,
        // so over the right half of a glyph it names the character after it. Stepping
        // back when the boundary it named sits to the right of the pointer is what turns
        // an insertion index into the character actually under the pointer.
        let start = CTLineGetStringRange(line).location
        var offset = boundary
        if CTLineGetOffsetForStringIndex(line, offset, nil) > x, offset > start {
            offset -= 1
        }
        guard offset >= 0, offset < text.utf16.count else { return nil }

        return String.Index(utf16Offset: offset, in: text)
    }

    /// The baseline origin of one character, in the text's own space.
    ///
    /// This is the coordinate `showDefinition(for:at:)` wants, and it is asked for the
    /// first character of the *word* rather than for the character under the pointer, so
    /// the panel is anchored to the start of the word however far into it the reader
    /// right-clicked.
    static func baselineOrigin(
        of index: String.Index, in text: String, font: PlatformFont, width: CGFloat
    ) -> CGPoint? {
        let offset = index.utf16Offset(in: text)
        guard let laid = layout(text, font: font, width: width) else { return nil }

        for (line, baseline) in laid {
            let range = CTLineGetStringRange(line)
            guard offset >= range.location, offset < range.location + range.length else {
                continue
            }
            return CGPoint(
                x: baseline.x + CTLineGetOffsetForStringIndex(line, offset, nil),
                y: baseline.y)
        }
        return nil
    }

    /// The line's visual lines, each with its baseline origin measured from the top of
    /// the text.
    ///
    /// A framesetter and not a bare `CTLine`, because a prose scene wraps: the
    /// grave-diggers in Hamlet V.i are three visual lines of one `Line`, and a single
    /// `CTLine` would hit-test all three against the first.
    private static func layout(
        _ text: String, font: PlatformFont, width: CGFloat
    ) -> [(CTLine, CGPoint)]? {
        guard !text.isEmpty, width > 0 else { return nil }

        let attributed = NSAttributedString(string: text, attributes: [.font: font])
        let framesetter = CTFramesetterCreateWithAttributedString(attributed)
        // A tall finite height rather than `.greatestFiniteMagnitude`, which CTFrame does
        // not lay out into reliably. No line of the corpus wraps anywhere near this.
        let bounds = CGRect(x: 0, y: 0, width: width, height: 100_000)
        let frame = CTFramesetterCreateFrame(
            framesetter, CFRange(location: 0, length: 0),
            CGPath(rect: bounds, transform: nil), nil)

        guard let lines = CTFrameGetLines(frame) as? [CTLine], !lines.isEmpty else {
            return nil
        }
        var origins = [CGPoint](repeating: .zero, count: lines.count)
        CTFrameGetLineOrigins(frame, CFRange(location: 0, length: 0), &origins)

        // CTFrame's origin is the bottom-left of its path, so a baseline's distance from
        // the top of the text is the frame height minus its origin.
        return zip(lines, origins).map {
            ($0, CGPoint(x: $1.x, y: bounds.height - $1.y))
        }
    }
}
