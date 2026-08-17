import SwiftUI

/// Row geometry, published up from every visible `LineRow`.
///
/// Drag selection needs to map a point to a line, and only the rows themselves
/// know where they ended up. A `PreferenceKey` of `[Int: CGRect]` is the macOS 14
/// way to collect that: `onScrollGeometryChange` and `ScrollPosition` are 15+.
struct RowFramesKey: PreferenceKey {
    static let defaultValue: [Int: CGRect] = [:]

    static func reduce(value: inout [Int: CGRect], nextValue: () -> [Int: CGRect]) {
        value.merge(nextValue(), uniquingKeysWith: { _, new in new })
    }
}

extension [Int: CGRect] {
    /// The line under `point`, for drag selection.
    ///
    /// Vertical containment first, because rows span the full width and a drag
    /// that wanders sideways should still track the row it is level with. Falling
    /// back to the nearest `midY` is what makes a drag past the last row select
    /// the last row rather than stalling.
    ///
    /// Only visible rows are in the map, which is not a limitation in practice:
    /// you can only drag over what you can see. A long selection is click the
    /// first line, scroll, shift-click the last.
    func line(at point: CGPoint) -> Int? {
        if let hit = first(where: {
            point.y >= $0.value.minY && point.y < $0.value.maxY
        }) {
            return hit.key
        }
        return self.min { lhs, rhs in
            abs(lhs.value.midY - point.y) < abs(rhs.value.midY - point.y)
        }?.key
    }
}

/// Publishes one row's frame in the reader's coordinate space.
struct RowFrameReporter: ViewModifier {
    let index: Int
    let space: String

    func body(content: Content) -> some View {
        content.background(
            GeometryReader { geometry in
                Color.clear.preference(
                    key: RowFramesKey.self,
                    value: [index: geometry.frame(in: .named(space))])
            }
        )
    }
}

extension View {
    func reportRowFrame(index: Int, space: String) -> some View {
        modifier(RowFrameReporter(index: index, space: space))
    }
}
