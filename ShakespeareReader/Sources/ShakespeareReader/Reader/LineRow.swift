import SwiftUI

/// One line of the play: number gutter, optional speaker heading, the text.
///
/// Selected styling is deliberately **size-neutral** — a background fill, a left
/// accent rule, and a colour change, with no weight or size change. Row height
/// feeds `RowFramesKey`, which is written into `@State`, which is read back during
/// layout; anything that makes height depend on selection closes that loop. The
/// hovered word's mark is bound by the same contract, and that is why it is an
/// underline: an underline is drawn inside the line box, where a heavier weight or a
/// larger size would reflow the row the pointer is sitting on.
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

    /// A word of this line, and where its dictionary panel should be popped: the word's
    /// baseline origin in `DictionaryAnchor.space`. Both are macOS-only paths today; on
    /// iOS nothing calls them, because there is no context menu to call them from.
    let onLookUpWord: (String, CGPoint) -> Void

    /// A word of this line, to be asked about in the commentary pane. The line index goes
    /// with it, because the passage may not be the one currently glossed.
    let onExplainWord: (String, Int) -> Void

    @Environment(\.readerTypeface) private var typeface

    /// The word under the pointer, as a range of `line.plainText`.
    ///
    /// Local to the row and **not** on `SceneReaderView`, which is the difference between
    /// invalidating one row's body as the pointer crosses a word boundary and invalidating
    /// every visible row's. Written only when the resolved range changes, rather than once
    /// per hover event, for the same reason.
    @State private var marked: Range<String.Index>?

    /// The verse `Text`'s own frame, in `DictionaryAnchor.space`.
    ///
    /// A reference box and deliberately **not** `@State`: it is written from geometry and
    /// read only inside the hover handler, never in `body`, so measuring the row can never
    /// invalidate the row it measured. This is what `rowFrames` does one level up, for the
    /// same reason.
    @State private var verseFrame = FrameBox()

    /// Verse indent. Speech lines hang under their heading; directions sit further
    /// in and in italic, the way a printed edition sets them. The measure itself is
    /// the typeface's, because it is proportional to the type it indents.
    private var indent: CGFloat { line.isDirection ? typeface.directionIndent : 0 }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(numberLabel)
                // The *face* stays the system's whatever the play is set in, because a
                // serif family has no monospaced digits. The *size* does not: it
                // follows the reader's step, so the numbers stay legible beside the
                // verse. Font and width come from the typeface together and never one
                // without the other — the width is a budget for three digits'
                // advances, so a bigger font in a fixed frame is the clipping case.
                .font(typeface.gutterFont)
                .foregroundStyle(.tertiary)
                .frame(width: typeface.gutterWidth, alignment: .trailing)

            VStack(alignment: .leading, spacing: 2) {
                if line.startsSpeech, let display {
                    Text(display.uppercased())
                        .font(typeface.speakerHeading)
                        .foregroundStyle(.secondary)
                        .kerning(typeface.speakerTracking)
                        .padding(.top, index == 0 ? 0 : typeface.speechGap)
                }
                verse
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

    /// The play text itself, and the only thing in the row a word can be looked up from:
    /// right-clicking the number gutter or the speaker heading offers no word, which is
    /// correct — neither is the play.
    @ViewBuilder
    private var verse: some View {
        markedText
            .font(line.isDirection ? typeface.direction : typeface.verse)
            .foregroundStyle(line.isDirection ? .secondary : .primary)
            .padding(.leading, indent)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background {
                GeometryReader { geometry in
                    Color.clear
                        .onAppear {
                            verseFrame.rect = geometry.frame(
                                in: .named(DictionaryAnchor.space))
                        }
                        .onChange(
                            of: geometry.frame(in: .named(DictionaryAnchor.space))
                        ) { _, frame in
                            verseFrame.rect = frame
                        }
                }
            }
            // macOS only, both of them, and for different reasons. Hover is a pointer
            // event a phone does not have; the menu is a *long press* there, which is
            // already how `SweepRecognizer` arms a sweep — the app's primary touch
            // interaction — so wiring word lookup up on iOS is a separate question about
            // which gesture wins, not a matter of dropping these in.
            #if os(macOS)
        // Hover consumes neither clicks nor drags, so nothing about selection changes:
        // click still selects, shift-click extends, double-click takes the speech and
        // a drag still sweeps. That is why the word is resolved here rather than from
        // `NSApp.currentEvent` when the menu opens, which would mean converting a
        // window point across AppKit's flipped Y into a SwiftUI space.
        .onContinuousHover(coordinateSpace: .named(DictionaryAnchor.space)) { phase in
            switch phase {
            case .active(let point):
                let resolved = resolvedWord(at: point)
                if resolved != marked { marked = resolved }
            case .ended:
                if marked != nil { marked = nil }
            }
        }
        .contextMenu { wordMenu }
            #endif
    }

    /// The verse, with the hovered word underlined.
    ///
    /// An `AttributedString` is built **only** when a word is actually marked: with nothing
    /// hovered this is `Text(line.plainText)` on the identical code path the app has always
    /// taken. Same move `ReaderTextSize` makes by short-circuiting to the bare text style
    /// at `.default` — the shipped rendering stays what it is rather than becoming
    /// something equivalent to it.
    private var markedText: Text {
        let text = line.plainText
        guard let marked else { return Text(text) }

        var string = AttributedString(text)
        guard let lower = AttributedString.Index(marked.lowerBound, within: string),
            let upper = AttributedString.Index(marked.upperBound, within: string)
        else { return Text(text) }

        // An underline rather than a background fill, so the mark does not compete with
        // the selection band, which is already a fill. `Text.LineStyle` and not
        // `NSUnderlineStyle`: both scopes spell this attribute `underlineStyle`, and only
        // the SwiftUI one is the attribute a `Text` draws.
        string[lower ..< upper].underlineStyle = Text.LineStyle.single
        return Text(string)
    }

    /// The word under a point, or `nil` — a space, punctuation, or the blank right of a
    /// short line. `nil` marks nothing and offers nothing, which is what makes a
    /// hit-testing near-miss something the reader can see before committing to it.
    private func resolvedWord(at point: CGPoint) -> Range<String.Index>? {
        let frame = verseFrame.rect
        // The indent is part of the row, not of the text: the `Text` starts that far in,
        // so both the point and the wrap width are measured from where it starts.
        let width = frame.width - indent
        let local = CGPoint(x: point.x - frame.minX - indent, y: point.y - frame.minY)
        guard
            let index = WordHitTest.characterIndex(
                at: local, in: line.plainText, font: platformFont, width: width)
        else { return nil }
        return WordTokenizer.word(at: index, in: line.plainText)
    }

    private var platformFont: PlatformFont {
        line.isDirection ? typeface.directionPlatformFont : typeface.versePlatformFont
    }

    /// Look Up, Explain, Copy — and nothing at all where no word resolved, so the menu
    /// never names a word the reader was not pointing at.
    ///
    /// **Right-clicking deliberately does not select the line.** macOS convention says it
    /// should, but selecting here commits a generation 350 ms later, which is far too heavy
    /// a side effect for opening a menu. Only "Explain" moves the selection, and only
    /// because it has to have a passage to ask about.
    @ViewBuilder
    private var wordMenu: some View {
        if let marked {
            let term = WordTokenizer.term(for: marked, in: line.plainText)
            if !term.isEmpty {
                // Gated on the dictionary actually having an entry, so `undiscover’d` is
                // offered no panel rather than an empty one.
                if DictionaryAnchor.hasDefinition(for: term) {
                    Button("Look Up “\(term)”") { lookUp(term, from: marked) }
                }
                Button("Explain “\(term)” in context") { onExplainWord(term, index) }
                Divider()
                Button("Copy “\(term)”") { copyToPasteboard(term) }
            }
        }
    }

    /// The panel is anchored to the start of the word rather than to wherever in it the
    /// reader right-clicked, so it lands the same way twice.
    private func lookUp(_ term: String, from range: Range<String.Index>) {
        let frame = verseFrame.rect
        guard
            let origin = WordHitTest.baselineOrigin(
                of: range.lowerBound, in: line.plainText, font: platformFont,
                width: frame.width - indent)
        else { return }
        onLookUpWord(
            term,
            CGPoint(x: origin.x + frame.minX + indent, y: origin.y + frame.minY))
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

/// A frame, held by reference so that writing it is not a view update.
///
/// The `@State` that holds one of these never changes identity, so the row is never
/// invalidated by measuring itself — which is the whole point, since the thing being
/// measured is the text whose height feeds `RowFramesKey`.
@MainActor
final class FrameBox {
    var rect: CGRect = .zero
}
