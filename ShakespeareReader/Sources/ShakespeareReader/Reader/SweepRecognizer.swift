#if !os(macOS)
import SwiftUI
import UIKit

/// Press-and-hold-then-drag, as a `UILongPressGestureRecognizer` on the enclosing
/// `UIScrollView` rather than as a SwiftUI gesture on the rows.
///
/// SwiftUI cannot express this. Every shape of `DragGesture` attached to the verse rows
/// takes the vertical pan away from the scroll view and the scene stops scrolling
/// entirely — measured with `minimumDistance` 0 and 4, with `.gesture` and with
/// `.simultaneousGesture`, and behind a `LongPressGesture.sequenced(before:)`. Masking
/// the drag off with `including: .none` until a long press arms it does restore
/// scrolling, but then the sweep never starts: a gesture that was masked when the finger
/// landed is not handed the touch already in flight.
///
/// A long press recognizer has neither problem. It requires stillness to recognize, so
/// an ordinary pan fails it and belongs to the scroll view; and once recognized it keeps
/// reporting `.changed` with the finger's location, which is exactly the sweep. It is
/// added to the scroll view itself, with `cancelsTouchesInView` off so the rows keep
/// their taps, and simultaneous recognition allowed so it never inhibits the pan.
struct SweepRecognizer: UIViewRepresentable {
    /// All three take a point in the enclosing scroll view's *viewport* space, which is
    /// the space `rowFrames` is measured in.
    let onBegan: (CGPoint) -> Void
    let onChanged: (CGPoint) -> Void
    let onEnded: () -> Void

    func makeUIView(context: Context) -> UIView {
        let view = AttachView()
        // The recognizer lives on the scroll view, so this view wants no touches of its
        // own; it is here only for its place in the hierarchy.
        view.isUserInteractionEnabled = false
        view.coordinator = context.coordinator
        return view
    }

    func updateUIView(_ view: UIView, context: Context) {
        context.coordinator.onBegan = onBegan
        context.coordinator.onChanged = onChanged
        context.coordinator.onEnded = onEnded
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var onBegan: ((CGPoint) -> Void)?
        var onChanged: ((CGPoint) -> Void)?
        var onEnded: (() -> Void)?
        weak var scrollView: UIScrollView?

        lazy var recognizer: UILongPressGestureRecognizer = {
            let recognizer = UILongPressGestureRecognizer(
                target: self, action: #selector(handle))
            // The same 0.3 s the SwiftUI `LongPressGesture` used.
            recognizer.minimumPressDuration = 0.3
            // Taps on the rows are still SwiftUI's.
            recognizer.cancelsTouchesInView = false
            recognizer.delegate = self
            return recognizer
        }()

        @objc private func handle(_ recognizer: UILongPressGestureRecognizer) {
            guard let scrollView else { return }
            // `location(in:)` on a scroll view is in content coordinates, since its
            // bounds origin *is* the content offset. `rowFrames` is measured in the
            // named space declared on the `ScrollView`, which is the viewport.
            let inContent = recognizer.location(in: scrollView)
            let point = CGPoint(
                x: inContent.x - scrollView.contentOffset.x,
                y: inContent.y - scrollView.contentOffset.y)

            switch recognizer.state {
            case .began: onBegan?(point)
            case .changed: onChanged?(point)
            case .ended, .cancelled, .failed: onEnded?()
            default: break
            }
        }

        /// Never inhibit the scroll view's own pan. Before this recognizer succeeds the
        /// pan must be free to win; after it succeeds the scroll is disabled from the
        /// SwiftUI side instead.
        func gestureRecognizer(
            _ gestureRecognizer: UIGestureRecognizer,
            shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer
        ) -> Bool { true }
    }

    /// Finds the enclosing scroll view once this view is in a window and hangs the
    /// recognizer on it. There is no SwiftUI modifier that reaches the scroll view a
    /// `ScrollView` builds, so walking up is the only way to it.
    final class AttachView: UIView {
        var coordinator: Coordinator?

        override func didMoveToWindow() {
            super.didMoveToWindow()
            guard window != nil, let coordinator, coordinator.scrollView == nil else {
                return
            }
            var next: UIView? = superview
            while let view = next, !(view is UIScrollView) { next = view.superview }
            guard let scrollView = next as? UIScrollView else { return }
            coordinator.scrollView = scrollView
            scrollView.addGestureRecognizer(coordinator.recognizer)
        }
    }
}
#endif
