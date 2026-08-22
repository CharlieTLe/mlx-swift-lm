/// What the navigator has open: at most one play, and at most one act inside it.
///
/// An accordion rather than a set of collapsed ids, and the polarity is the point.
/// 35 plays / 175 acts / 707 scenes is over 900 rows in a 210pt column, so a set of
/// *collapsed* acts has the wrong default — empty means everything open, which is
/// exactly what a first launch gets. Recording what is *open* instead, one of each,
/// makes the closed outline free: a cold start is 35 play rows.
///
/// Nothing here is persisted. The reading position implies the whole value, so at
/// launch the outline is whatever the restored `SceneKey` says (see
/// `ContentView`'s `onChange(of: sceneKey)`), and a reader who collapsed the current
/// play before quitting comes back with it open — one tap to redo, and arguably the
/// right thing to forget.
///
/// The invariant, which is why both mutators write both fields: an open act belongs
/// to the open play. There is no state where `act` names an act of a closed one.
struct NavigatorOutline: Equatable, Sendable {
    /// `Play.id`, or nothing open at all.
    private(set) var play: String?
    /// `Act.number` within `play`.
    private(set) var act: Int?

    init(play: String? = nil, act: Int? = nil) {
        self.play = play
        self.act = act
    }

    /// The outline the reading position implies: the play and act of the scene being
    /// read, and nothing else.
    static func following(_ key: SceneKey) -> NavigatorOutline {
        NavigatorOutline(play: key.playID, act: key.act)
    }

    func isOpen(play id: String) -> Bool { play == id }

    func isOpen(act number: Int, in playID: String) -> Bool {
        play == playID && act == number
    }

    /// Closing a play drops its act with it; opening another starts with its acts
    /// shut. That second half is deliberate: opening a play should give the reader
    /// five act rows to choose between, not 28 scene rows.
    mutating func toggle(play id: String) {
        play = isOpen(play: id) ? nil : id
        act = nil
    }

    /// Opens the act's play along with it, since an act cannot be open inside a
    /// closed play. Closing the act leaves the play open.
    mutating func toggle(act number: Int, in playID: String) {
        act = isOpen(act: number, in: playID) ? nil : number
        play = playID
    }
}
