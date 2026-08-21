import SwiftUI

/// Play / act / scene outline. Selecting a scene is how the reader moves, since
/// only one scene is rendered at a time.
///
/// The acts collapse, but **not** with `DisclosureGroup`. Inside a sidebar `List`
/// that control keeps its own expansion state and overrides whatever binding it is
/// handed: `.constant(true)` left the acts in mixed states, and both a
/// parent-derived binding and a child-owned `@State` initialized to `true` rendered
/// most acts shut and refused to open. Emitting the scene rows conditionally instead
/// leaves the list nothing to disagree with — a collapsed act's rows do not exist.
@MainActor
struct NavigatorView: View {
    let corpus: Corpus
    @Binding var key: SceneKey

    /// Acts the reader has collapsed, by `"<play>-<act>"`. Empty means all open,
    /// which is the useful default: the scene settings are what make this navigable.
    ///
    /// Owned by `ContentView` rather than here because hiding this pane removes the
    /// view, and a `@State` set would come back empty with every act re-expanded.
    @Binding var collapsed: Set<String>

    var body: some View {
        // `List` selection is optional; the reader always has a scene open, so a
        // deselection is ignored rather than allowed to empty the pane.
        let selected = Binding<SceneKey?>(
            get: { key },
            set: { if let new = $0 { key = new } })

        List(selection: selected) {
            ForEach(corpus.plays) { play in
                Section(play.title) {
                    ForEach(play.acts, id: \.number) { act in
                        let id = "\(play.id)-\(act.number)"
                        let isOpen = !collapsed.contains(id)

                        actHeader(act, id: id, isOpen: isOpen)

                        if isOpen {
                            // Keyed on the whole `SceneKey`, not on the scene number.
                            // Now that the act headers and their scenes are flat
                            // siblings in one section, scene numbers collide across
                            // acts — and the list reused Act I's rows for Act II and
                            // III, showing the right scene numbers with the wrong
                            // settings.
                            ForEach(entries(play: play, act: act)) { entry in
                                row(entry)
                            }
                        }
                    }
                }
            }
        }
        .listStyle(.sidebar)
    }

    /// Carries no `.tag`, so the list does not treat it as a selectable scene.
    @ViewBuilder
    private func actHeader(_ act: Act, id: String, isOpen: Bool) -> some View {
        Button {
            if isOpen {
                collapsed.insert(id)
            } else {
                collapsed.remove(id)
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: isOpen ? "chevron.down" : "chevron.right")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.tertiary)
                    .frame(width: 10)
                Text("Act \(RomanNumeral.string(act.number))")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }
            // The whole width is the hit target, not just the glyph.
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    /// A scene paired with the key that addresses it, which is both its list
    /// identity and its selection tag.
    private struct SceneEntry: Identifiable {
        let id: SceneKey
        let scene: Scene
    }

    private func entries(play: Play, act: Act) -> [SceneEntry] {
        act.scenes.map {
            SceneEntry(
                id: SceneKey(playID: play.id, act: act.number, scene: $0.number),
                scene: $0)
        }
    }

    @ViewBuilder
    private func row(_ entry: SceneEntry) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(SceneLabel.string(entry.scene.number))
                .font(.callout)
            // The setting is what makes a scene list navigable — "A churchyard"
            // locates the grave-diggers far faster than "Scene I" does.
            Text(entry.scene.setting)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .padding(.leading, 14)
        .tag(entry.id)
    }
}
