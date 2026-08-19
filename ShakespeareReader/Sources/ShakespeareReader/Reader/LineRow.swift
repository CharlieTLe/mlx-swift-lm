import SwiftUI

/// One line of the play: number gutter, optional speaker heading, the text.
///
/// Selected styling is deliberately **size-neutral** — a background fill, a left
/// accent rule, and a colour change, with no weight or size change. Row height
/// feeds `RowFramesKey`, which is written into `@State`, which is read back during
/// layout; anything that makes height depend on selection closes that loop.
///
/// Changing the typeface resizes every row and so is *not* that loop: `rowFrames` is
/// written from a preference and read **only** inside the drag gesture in
/// `SceneReaderView.row(index:line:)`, never in `body`, so a font change is a one-shot
/// relayout that settles. Selection is the thing that has to stay size-neutral,
/// because `isSelected` *is* read during layout.
@MainActor
struct LineRow: View {
    let index: Int
    let line: Line
    let display: String?
    let isSelected: Bool
    let isFirstSelected: Bool

    /// Whether the reader pane holds the keyboard. Focus stays in the navigator when a
    /// scene is picked there, so the band goes grey to say the arrows are pointed
    /// somewhere else, the way an unfocused `NSTableView` does. Colour **only**: this is
    /// read during layout just as `isSelected` is, so it is bound by the same
    /// size-neutrality contract above.
    let hasFocus: Bool

    @Environment(\.readerTypeface) private var typeface

    /// Verse indent. Speech lines hang under their heading; directions sit further
    /// in and in italic, the way a printed edition sets them.
    private var indent: CGFloat { line.isDirection ? 28 : 0 }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(numberLabel)
                // Stays on the system face whatever the play is set in: a serif
                // family has no monospaced digits, and the `.frame(width: 30)`
                // below depends on stable advances to keep the gutter aligned.
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.tertiary)
                .frame(width: 30, alignment: .trailing)

            VStack(alignment: .leading, spacing: 2) {
                if line.startsSpeech, let display {
                    Text(display.uppercased())
                        .font(typeface.speakerHeading)
                        .foregroundStyle(.secondary)
                        .kerning(typeface.speakerTracking)
                        .padding(.top, index == 0 ? 0 : typeface.speechGap)
                }
                Text(line.plainText)
                    .font(line.isDirection ? typeface.direction : typeface.verse)
                    .foregroundStyle(line.isDirection ? .secondary : .primary)
                    .padding(.leading, indent)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        // Hit target and chrome, not typography, so neither of these scales with the
        // typeface the way `speechGap` above does.
        .padding(.vertical, 1)
        .padding(.horizontal, 6)
        .background(alignment: .leading) {
            if isSelected {
                ZStack(alignment: .leading) {
                    Rectangle().fill(bandFill)
                    Rectangle()
                        .fill(bandRule)
                        .frame(width: 2)
                }
            }
        }
        .contentShape(Rectangle())
    }

    /// `AnyShapeStyle` because the two branches are different style types, which is the
    /// house idiom (`ContentView.paneToggle`).
    private var bandFill: AnyShapeStyle {
        hasFocus
            ? AnyShapeStyle(Color.accentColor.opacity(0.14)) : AnyShapeStyle(.quaternary)
    }

    private var bandRule: AnyShapeStyle {
        hasFocus ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.tertiary)
    }

    private var numberLabel: String {
        guard let number = line.number else { return "" }
        // Every fifth line, as printed editions do: a number on every line is
        // noise, and none at all makes the citation unverifiable.
        return number % 5 == 0 || isFirstSelected ? "\(number)" : ""
    }
}
