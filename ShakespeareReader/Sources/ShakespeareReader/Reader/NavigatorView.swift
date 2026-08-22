import SwiftUI

/// Play / act / scene outline, with a find field over it. Selecting a scene is how
/// the reader moves, since only one scene is rendered at a time.
///
/// An **accordion**: one play open, one act open inside it, following the reading
/// position. With 35 plays the old shape — a section per play, every act header
/// showing, every act expanded unless the reader collapsed it — was over 900 rows in
/// a 210pt column, so finding a play meant scrolling past a dozen others' scenes.
/// Closed, it is 35 play rows and the one act being read. `NavigatorOutline` holds
/// that state and `NavigatorSearch` is the other way in, by title or by setting.
///
/// The plays and acts collapse, but **not** with `DisclosureGroup`. Inside a sidebar
/// `List` that control keeps its own expansion state and overrides whatever binding
/// it is handed: `.constant(true)` left the acts in mixed states, and both a
/// parent-derived binding and a child-owned `@State` initialized to `true` rendered
/// most acts shut and refused to open. Emitting the rows conditionally instead leaves
/// the list nothing to disagree with — a collapsed act's rows do not exist, and
/// neither do a collapsed play's acts.
@MainActor
struct NavigatorView: View {
    let corpus: Corpus
    @Binding var key: SceneKey

    /// What is open: one play, one act. Owned by `ContentView` rather than here
    /// because hiding this pane removes the view, and a `@State` outline would come
    /// back shut with the reader's own play collapsed under them.
    @Binding var outline: NavigatorOutline

    /// The find field. `@State` and not a preference: a filter is a moment, and it
    /// clearing when the pane is hidden is right.
    @State private var query = ""
    @FocusState private var isQueryFocused: Bool

    private var isSearching: Bool {
        !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(spacing: 0) {
            searchField
            Divider()
            outlineList
        }
        .background {
            // ⌘F, needing a control to hang off. Zero-opacity rather than
            // `.hidden()`, which removes it from the hierarchy along with its
            // shortcut.
            //
            // Live only while this pane is on screen: hidden, the navigator does not
            // exist and ⌘F does nothing, which is the same trade the pane's own state
            // makes by living in `ContentView`.
            Button("Find a play") { isQueryFocused = true }
                .keyboardShortcut("f", modifiers: .command)
                .opacity(0)
                .frame(width: 0, height: 0)
                .accessibilityHidden(true)
        }
    }

    // MARK: - Outline

    @ViewBuilder
    private var outlineList: some View {
        // `List` selection is optional; the reader always has a scene open, so a
        // deselection is ignored rather than allowed to empty the pane.
        let selected = Binding<SceneKey?>(
            get: { key },
            set: { if let new = $0 { key = new } })

        let matches = NavigatorSearch.matches(in: corpus, query: query)

        // The open act can be anywhere in the corpus now that the outline starts
        // closed, so the sidebar scrolls itself. Mirrors `SceneReaderView`: the rows'
        // ids are declared by `ForEach` over `SceneEntry`, so `SceneKey` is the
        // scroll target.
        ScrollViewReader { scroller in
            List(selection: selected) {
                if matches.isEmpty {
                    Text("No plays match")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }

                ForEach(matches) { match in
                    playHeader(match)

                    if isOpen(match) {
                        ForEach(match.acts) { actMatch in
                            actHeader(match, actMatch)

                            if isOpen(match, actMatch) {
                                // Keyed on the whole `SceneKey`, not on the scene
                                // number. The act headers and their scenes are flat
                                // siblings, so scene numbers collide across acts —
                                // and the list reused Act I's rows for Act II and
                                // III, showing the right scene numbers with the wrong
                                // settings. The play rows made the collision wider,
                                // not narrower.
                                ForEach(
                                    entries(
                                        play: match.play, act: actMatch.act,
                                        scenes: actMatch.scenes)
                                ) { entry in
                                    row(entry)
                                }
                            }
                        }
                    }
                }
            }
            .listStyle(.sidebar)
            .onAppear { scroll(scroller) }
            // What catches an arrow key rolling into the next scene from the reader
            // pane. A tap in the sidebar scrolls to a row already on screen, which is
            // a harmless no-op.
            .onChange(of: key) { scroll(scroller) }
        }
    }

    /// Puts the scene being read on screen, **next** main-actor turn.
    ///
    /// Deferred deliberately. A roll into the next act arrives here before
    /// `ContentView` has moved the outline onto it — that follow is an `onChange` of
    /// its own — so at this instant the act holding the new scene is still shut and
    /// the row being aimed at does not exist. `scrollTo` on an id the list has not
    /// declared does nothing at all, silently, which is what left the sidebar sitting
    /// on the play list while the reader had rolled into Act V. By the next turn the
    /// outline has moved and the rows are there.
    private func scroll(_ scroller: ScrollViewProxy) {
        let key = key
        Task { scroller.scrollTo(key, anchor: .center) }
    }

    /// Every result of a search is open, so the hits are on screen; otherwise the
    /// accordion decides.
    private func isOpen(_ match: NavigatorSearch.Match) -> Bool {
        isSearching || outline.isOpen(play: match.play.id)
    }

    /// A setting match forces its act open — the matching scenes are the answer, and
    /// a play whose acts are all shut would be showing none of them.
    private func isOpen(
        _ match: NavigatorSearch.Match, _ actMatch: NavigatorSearch.ActMatch
    ) -> Bool {
        !match.matchedTitle || outline.isOpen(act: actMatch.act.number, in: match.play.id)
    }

    // MARK: - Headers

    /// The play title, in its real casing: a row rather than the `Section(play.title)`
    /// this used to be, because a section header takes no tap and the sidebar style
    /// uppercases it.
    ///
    /// Carries no `.tag`, so the list does not treat it as a selectable scene.
    @ViewBuilder
    private func playHeader(_ match: NavigatorSearch.Match) -> some View {
        let font = Font.subheadline.weight(.semibold)

        if isSearching {
            // Held open by the search, so its chevron has nothing to close — a plain
            // label rather than a `Button`, for the reason under `actHeader`.
            headerRow(match.play.title, isOpen: true, font: font, prominent: true)
        } else {
            Button {
                outline.toggle(play: match.play.id)
            } label: {
                headerRow(
                    match.play.title, isOpen: outline.isOpen(play: match.play.id),
                    font: font, prominent: true)
            }
            .buttonStyle(.plain)
        }
    }

    /// Carries no `.tag` either.
    ///
    /// Drawn as a plain label rather than a `Button` when the search is holding it
    /// open: a chevron whose tap cannot close the row is a lie, and it is the same
    /// disagreement this file's header comment describes with `DisclosureGroup`, from
    /// the other side — a control whose state the view refuses to honour.
    @ViewBuilder
    private func actHeader(
        _ match: NavigatorSearch.Match, _ actMatch: NavigatorSearch.ActMatch
    ) -> some View {
        let title = "Act \(RomanNumeral.string(actMatch.act.number))"
        let font = Font.caption.weight(.semibold)

        Group {
            if match.matchedTitle {
                Button {
                    outline.toggle(act: actMatch.act.number, in: match.play.id)
                } label: {
                    headerRow(
                        title, isOpen: isOpen(match, actMatch), font: font,
                        prominent: false)
                }
                .buttonStyle(.plain)
            } else {
                headerRow(title, isOpen: true, font: font, prominent: false)
            }
        }
        // Under its play, the way the scenes below sit under their act.
        .padding(.leading, 12)
    }

    @ViewBuilder
    private func headerRow(
        _ title: String, isOpen: Bool, font: Font, prominent: Bool
    ) -> some View {
        HStack(spacing: 4) {
            Image(systemName: isOpen ? "chevron.down" : "chevron.right")
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(.tertiary)
                .frame(width: 10)
            Text(title)
                .font(font)
                .foregroundStyle(
                    prominent ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary)
                )
                .lineLimit(2)
            Spacer(minLength: 0)
        }
        // The whole width is the hit target, not just the glyph.
        .contentShape(Rectangle())
    }

    // MARK: - Find field

    /// Hand-rolled and shared, not `.searchable`. On a phone this pane is a
    /// `NavigationSplitView` column and `.searchable` would render into the
    /// navigation bar, but on a Mac the navigator is a bare `List` inside an
    /// `HSplitView` with no toolbar for a search field to go in. The shape is the Ask
    /// field's, in `AnnotationPaneView`.
    @ViewBuilder
    private var searchField: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(.caption)
                .foregroundStyle(.secondary)

            TextField("Find a play", text: $query)
                .textFieldStyle(.plain)
                .font(.callout)
                .focused($isQueryFocused)
                // Return keeps the filter and hands the keyboard back, which is what
                // dismisses the software one on a phone; on a Mac it is the arrows and
                // Esc returning to the reader pane.
                .onSubmit { isQueryFocused = false }
                #if !os(macOS)
            .submitLabel(.search)
            .autocorrectionDisabled()
            .textInputAutocapitalization(.never)
                #else
            // Esc belongs to the field while it holds the keyboard, and giving the
            // keyboard up is the point: focused, the field makes the reader pane's
            // arrows, Esc, ⌘C and ⌘R dead — the same trade the Ask field
            // documents. Esc clears the filter and lets the keyboard go; clicking a
            // line is what hands it back to the reader, since nothing claims focus
            // on its own when a `@FocusState` is dropped.
            .onExitCommand {
                query = ""
                isQueryFocused = false
            }
                #endif

            if !query.isEmpty {
                Button {
                    query = ""
                    isQueryFocused = false
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .accessibilityLabel("Clear the filter")
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 8)
        .background(RoundedRectangle(cornerRadius: 7).fill(.quaternary.opacity(0.5)))
        .padding(.horizontal, 8)
        .padding(.vertical, 8)
    }

    // MARK: - Scene rows

    /// A scene paired with the key that addresses it, which is both its list
    /// identity and its selection tag.
    private struct SceneEntry: Identifiable {
        let id: SceneKey
        let scene: Scene
    }

    /// `scenes` rather than `act.scenes`: a setting match carries only the scenes it
    /// matched, and those are the ones worth showing.
    private func entries(play: Play, act: Act, scenes: [Scene]) -> [SceneEntry] {
        scenes.map {
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
            // locates the grave-diggers far faster than "Scene I" does, and it is
            // the other half of what the find field searches.
            Text(entry.scene.setting)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .padding(.leading, 26)
        .tag(entry.id)
    }
}
