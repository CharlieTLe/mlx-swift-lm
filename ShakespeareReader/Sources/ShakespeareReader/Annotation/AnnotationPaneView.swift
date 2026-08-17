import SwiftUI

/// What the reader reads: the citation, the selected lines echoed small, the
/// streamed commentary, the ask rows, and the transcript of anything tapped.
@MainActor
struct AnnotationPaneView: View {
    struct Exchange: Identifiable, Equatable {
        let id = UUID()
        var question: String
        var answer: String
    }

    let citation: String?
    let selectedLines: [PassageContext.Utterance]
    let commentary: String
    let followUps: [String]
    let transcript: [Exchange]
    let isBusy: Bool
    let synopsisIsPartial: Bool
    let onAsk: (String) -> Void

    private static let space = "annotation"
    private static let contentID = "content"

    /// Within two lines of body text of the end still counts as "at the end", so a
    /// reader who flicks down without landing exactly at the bottom gets following
    /// back.
    private static let pinSlack: CGFloat = 40

    /// Layout rounding moves the content top by a fraction of a point; only a real
    /// scroll moves it further than this.
    private static let scrollSlack: CGFloat = 4

    @State private var isFollowing = true
    @State private var contentFrame: CGRect = .zero
    @State private var viewportHeight: CGFloat = 0
    @State private var lastTop: CGFloat = 0
    @State private var lastHeight: CGFloat = 0

    /// Following the stream is opt-in. The first gloss of a selection is meant to be
    /// read from the top, so the pane stays put and only chases the end of the
    /// content once the reader has tapped an ask row.
    @State private var isArmed = false

    /// Set while one of the pane's own jumps is in flight. Those move the content
    /// top exactly the way a reader scrolling up does, so without this they would
    /// turn following off the instant they land.
    @State private var isJumping = false

    var body: some View {
        ScrollViewReader { scroller in
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if let citation {
                        header(citation)
                    } else {
                        placeholder
                    }

                    if !commentary.isEmpty {
                        Text(commentary)
                            .font(.body)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    ForEach(transcript) { exchange in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(exchange.question)
                                .font(.callout.weight(.semibold))
                            Text(exchange.answer)
                                .font(.body)
                                .textSelection(.enabled)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }

                    // Last, always: the ask rows follow whatever was most recently
                    // written. Keeping them in a fixed slot above the transcript put the
                    // next set of questions back where the reader tapped, above the
                    // answer they had just asked for.
                    if !followUps.isEmpty {
                        askRows
                    }
                }
                .padding(14)
                // The scroll target is the whole content with a `.bottom` anchor,
                // which puts the end of the ask rows at the end of the viewport
                // rather than needing a sentinel row and the stack spacing that
                // would come with it.
                .id(Self.contentID)
                .background(
                    GeometryReader { geometry in
                        Color.clear.preference(
                            key: ContentFrameKey.self,
                            value: geometry.frame(in: .named(Self.space)))
                    }
                )
            }
            .coordinateSpace(name: Self.space)
            .background(
                GeometryReader { geometry in
                    Color.clear.preference(
                        key: ViewportHeightKey.self, value: geometry.size.height)
                }
            )
            .onPreferenceChange(ViewportHeightKey.self) { viewportHeight = $0 }
            .onPreferenceChange(ContentFrameKey.self) { track($0) }
            .onChange(of: streamTick) { follow(scroller) }
            .onChange(of: transcript.count) { old, new in
                guard new > old, let asked = transcript.last else { return }
                // The question the reader just tapped goes to the top, so it stays
                // in view while its answer fills the space underneath.
                jump { scroller.scrollTo(asked.id, anchor: .top) }
            }
            .onChange(of: commentary.isEmpty) { _, isEmpty in
                // A new selection, a regenerate and Esc all clear the commentary
                // first, and fresh content should start at its beginning.
                guard isEmpty else { return }
                isArmed = false
                jump { scroller.scrollTo(Self.contentID, anchor: .top) }
            }
        }
    }

    /// One value for "something grew", so following needs a single `onChange` rather
    /// than one per streamed field. The ask rows count too: they arrive in one piece
    /// once the gloss is done, and they are what the reader wants to see next.
    private var streamTick: Int {
        commentary.count + followUps.count + transcript.reduce(0) { $0 + $1.answer.count }
    }

    /// How far the end of the content sits past the end of the viewport.
    private var distanceBelowFold: CGFloat {
        contentFrame.maxY - viewportHeight
    }

    /// Re-arms following and scrolls somewhere other than the end.
    private func jump(_ scroll: () -> Void) {
        isFollowing = true
        isJumping = true
        scroll()
    }

    private func follow(_ scroller: ScrollViewProxy) {
        // `isBusy` is what separates a live stream from a cache hit, which lands the
        // whole gloss in one assignment and should be read from the top.
        guard isArmed, isBusy, isFollowing, distanceBelowFold > 0 else { return }
        // Unanimated, deliberately: this runs once per decoded token, and an
        // animated scroll per token queues 60 overlapping animations a second.
        scroller.scrollTo(Self.contentID, anchor: .bottom)
    }

    /// Content grows downward, so the top edge only moves when someone actually
    /// scrolls. That is what separates the reader scrolling up to re-read from
    /// another token arriving underneath them.
    ///
    /// Two things move the top with no reader involved: one of the pane's own jumps,
    /// and the content getting shorter, which makes the scroll view clamp its
    /// offset. Both are excluded, or following would keep switching itself off.
    private func track(_ frame: CGRect) {
        contentFrame = frame
        defer {
            lastTop = frame.minY
            lastHeight = frame.height
        }

        // Reports arrive a layout pass behind, so a jump is still in flight until
        // something actually changes; that report is the jump landing, and it is the
        // one whose moved top has to be forgiven.
        let landing = isJumping
        if frame.minY != lastTop || frame.height != lastHeight {
            isJumping = false
        }

        if !landing, frame.height >= lastHeight,
            frame.minY > lastTop + Self.scrollSlack
        {
            isFollowing = false
        } else if frame.maxY - viewportHeight <= Self.pinSlack {
            isFollowing = true
        }
    }

    @ViewBuilder
    private func header(_ citation: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Text(citation)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)

                // The only sign that work is in flight when the status strip is
                // hidden, and a cold reasoning phase or a scene summary can run for
                // seconds. Unlabelled, and gone the moment text starts arriving, so it
                // never competes with the stream.
                if isBusy, commentary.isEmpty {
                    ProgressView().controlSize(.small)
                }
            }

            // The selected lines are echoed so the reader does not lose the anchor
            // while reading the gloss.
            VStack(alignment: .leading, spacing: 1) {
                ForEach(Array(selectedLines.enumerated()), id: \.offset) { _, line in
                    Text(line.isDirection ? "[\(line.text)]" : line.text)
                        .font(line.isDirection ? .caption2.italic() : .caption)
                        .foregroundStyle(line.isDirection ? .tertiary : .secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding(.leading, 8)
            .overlay(alignment: .leading) {
                Rectangle().fill(.quaternary).frame(width: 2)
            }

            if synopsisIsPartial {
                Label(
                    "Scene summary covers the first part of this scene only.",
                    systemImage: "info.circle")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    @ViewBuilder
    private var placeholder: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Select lines to annotate")
                .font(.headline)
            Text(
                "Click a line, shift-click to extend, double-click for the whole "
                    + "speech, or drag. Esc clears. ⌘R regenerates.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// Full-width vertical rows with a chevron and dividers, which is what the
    /// reference UI actually is — and it avoids needing a flow layout.
    @ViewBuilder
    private var askRows: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("People also ask")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .padding(.bottom, 6)

            ForEach(Array(followUps.enumerated()), id: \.offset) { index, question in
                Button {
                    // Tapping is the reader asking to be carried along with the
                    // answer; until then the pane leaves the scroll position alone.
                    isArmed = true
                    onAsk(question)
                } label: {
                    HStack(spacing: 8) {
                        Text(question)
                            .font(.callout)
                            .multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Image(systemName: "chevron.right")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    .padding(.vertical, 8)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .disabled(isBusy)

                if index < followUps.count - 1 {
                    Divider()
                }
            }
        }
    }
}

/// The pane's content frame and its viewport height, in the pane's own coordinate
/// space. Following the stream needs to know how far the end of the content is
/// from the end of the viewport, and on macOS 14 a `PreferenceKey` is how you
/// learn that: `onScrollGeometryChange` and `ScrollPosition` are 15+.
private struct ContentFrameKey: PreferenceKey {
    static let defaultValue: CGRect = .zero

    /// Only one view reports a real frame, but sibling subviews — the scroll
    /// view's own background layer among them — all contribute the default, and
    /// they are not visited in any guaranteed order. So keep whichever value is
    /// not the default rather than the one that happens to come last, the same
    /// reason `ViewportHeightKey` reduces with `max`.
    static func reduce(value: inout CGRect, nextValue: () -> CGRect) {
        let next = nextValue()
        if next != .zero { value = next }
    }
}

private struct ViewportHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
