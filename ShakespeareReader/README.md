# Shakespeare Reader

A SwiftUI macOS app that annotates Shakespeare on-device: select lines in a play and
it explains them, in the style of a Genius.com annotation, then offers four
follow-up questions that continue the same conversation about that passage. After
the initial model download there are no network calls.

Reading Shakespeare stalls on two things: the language, and the fact that a passage
only means something in light of what happened twenty lines earlier. Print editions
solve that with footnotes plus a headnote per scene, and the reader still has to jump
around. This collapses the loop — select the lines, and the app gathers the
surrounding context itself: act, scene, setting, who is on stage, the run-up lines, a
summary of the scene so far.

![Hamlet I.iv open in the reader. A speech is double-clicked and the commentary pane
streams a gloss of it, then four follow-up questions. The arrow keys move and extend
the selection, each move re-glossing, until the selection rolls off the end of the
scene into I.v. There the Ghost's speech is glossed, a question is typed into the Ask
Anything field, and the answer streams in under it.](docs/demo.gif)

25 seconds, real time, no cuts: everything after each selection is the model running
on-device.

## Install

```bash
brew tap CharlieTLe/tap
brew trust CharlieTLe/tap
brew install shakespeare-reader
shakespeare-reader &
```

The `brew trust` step is not optional: recent Homebrew refuses to load a formula from
a third-party tap until you trust it, and `brew tap` does not say so.

The Homebrew build is not a `.app` bundle, so the command launches the window and holds
the terminal until you quit it; `&` gives the shell back. For a double-clickable Mac app,
build the Xcode project instead; see [As a Mac app](#as-a-mac-app).

The formula builds from source and the source includes MLX, so this is a compile rather
than a download: about two minutes and 1.6 GB of scratch on an M4 Max, longer on fewer
cores or on a first build, which also clones fourteen packages. Every build here needs
the same two things, whether it comes from Homebrew, `swift run`, or the Xcode project:

- **Apple silicon.** MLX has no Intel path.
- **A full Xcode**, selected with `xcode-select`, with the **Metal toolchain**
  component installed. The Command Line Tools ship a `metal` that is a stub and
  cannot compile mlx-swift's GPU kernels; a build without them links and launches
  and then fails on the first annotation with `Failed to load the default metallib`.
  The formula checks for this before building rather than shipping that binary, and
  the Xcode build does not, so it is worth checking by hand before the first Mac or
  iPhone run.

  ```bash
  xcodebuild -showComponent MetalToolchain      # want: Status: installed
  xcodebuild -downloadComponent MetalToolchain  # if it is not
  sudo xcode-select -s /Applications/Xcode.app  # if xcode-select points at the CLT
  ```

First launch downloads `mlx-community/Qwen3-4B-4bit` (about 2.2 GB) into
`~/.cache/huggingface`, and needs roughly 3 GB of memory while a passage is
annotated. After the download there are no network calls.

### From a checkout

```bash
cd ShakespeareReader
swift run -c release ShakespeareReader
```

Use `-c release`; a debug 4B forward pass is not worth watching.

Both app bundles, Mac and iPhone, come out of one project and one scheme:

```bash
open ShakespeareReader/App/ShakespeareReader.xcodeproj
```

Pick a destination and Run. The checked-in project sets **no** `DEVELOPMENT_TEAM`, so
nobody inherits anyone else's. `App/Signing.xcconfig` is the base configuration for both
build configurations and does nothing but optionally include `App/Local.xcconfig`, which
is gitignored. Put your team there to keep it out of `git status` entirely, and to build
from the command line:

```
DEVELOPMENT_TEAM = ABCDE12345
```

Only the iPhone build *requires* a team. My Mac needs none, for the reason the next
section gives.

### As a Mac app

Choose **My Mac** and Run. macOS 14.0 and up.

**No Apple Developer team, and no bundle-id change.** macOS signs a local build ad hoc,
so a fresh clone with `DEVELOPMENT_TEAM` unset builds and launches as-is. That is the
sharp contrast with the iPhone section below, where a device build needs both.

**Apple silicon only, explicitly.** The target pins `ARCHS[sdk=macosx*] = arm64`. That is
not belt-and-braces: `ARCHS_STANDARD` for macosx is still `arm64 x86_64`, and
`ONLY_ACTIVE_ARCH` is set only in Debug, so a Release build would otherwise try an x86_64
slice of the vendored MLX C++ and metal-cpp and MLX has no Intel path. The iPhone build
never met this because its `ARCHS_STANDARD` is arm64 alone.

There is an app icon: the Chandos portrait, cropped to the head and the ruff collar,
`App/Assets.xcassets/AppIcon.appiconset` for the ten macOS sizes and the iOS 1024.
`ASSETCATALOG_COMPILER_APPICON_NAME = AppIcon` is the whole of the wiring — actool writes
`CFBundleIconName`, `CFBundleIconFile` and an `AppIcon.icns` into the built bundle itself,
so neither `Info-macOS.plist` nor `Info-iOS.plist` names the icon and neither should. The
PNGs are checked in; `swift tools/icon/make_icon.swift` regenerates them.

The crop is the part that took a decision rather than a build setting. At 128pt any
square of the painting works, and at 16pt the full canvas is a dark smudge: the head sits
in the upper third and the face is a few pixels of it. Cropping to head-plus-collar keeps
the one high-contrast shape that survives the downsample, which is the white ruff, not the
face. The generator carries the four candidates it was chosen from.

Two things to know about the bundled app versus the Homebrew build, which matter because
both can be installed at once and are then two different Mac apps on one machine:

- **The caches are shared.** Both use `~/.cache/huggingface/hub` for the 2.2 GB of
  weights and the same absolute `~/Library/Application Support/ShakespeareReader/` for
  the annotation cache. Whichever you run first pays the download; the other starts warm.
- **The preferences are not.** Reading position, `readerFont` and `showsCommentary` do
  **not** carry across. The unbundled Homebrew executable has no bundle identifier, so
  CFPreferences falls back to the process name and it writes
  `~/Library/Preferences/ShakespeareReader.plist`; the bundled app uses its bundle id,
  `com.charliele.ShakespeareReader`. So "it forgot my place but kept my annotations" is
  expected, not a bug:

  ```bash
  defaults read ShakespeareReader               # the Homebrew / swift run build
  defaults read com.charliele.ShakespeareReader # the bundled Mac app
  ```

**App Sandbox is deliberately off**, and the Mac build signs against its own empty
`App/ShakespeareReader-macOS.entitlements` rather than the iPhone file. Sandboxing would
redirect `~/.cache/huggingface` into `~/Library/Containers/` and re-download 2.2 GB that
is already on disk; that file's own comment records this, along with why hardened
runtime needs no JIT exception. `FoundationModelsIntegration` is on here for the same
reason it is on for iPhone, described under [On iPhone and iPad](#on-iphone-and-ipad) below.

### On iPhone and iPad

Pick your team under **Signing & Capabilities**, change the bundle id
(`com.charliele.ShakespeareReader`) to one your team owns, and Run. One universal build,
iOS 18.0 and up; there is no visionOS layout.

The iPhone entitlements file, `App/ShakespeareReader-iOS.entitlements`, asks for one
thing, `com.apple.developer.kernel.increased-memory-limit`,
which raises the jetsam ceiling a 4B model needs and which a free personal team can grant.
It does *not* ask for `extended-virtual-addressing`: personal teams cannot sign that one at
all, and it only buys an address space past roughly 4 GB, where the measured peak working
set here is 3.31 GB. Note that a personal team's profile expires in about seven days, after
which the app stops launching until you rebuild it. None of this applies to the Mac build,
which asks for nothing.

**The 4B default wants an 8 GB device.** `AnnotationService.tuneMemory()` caps MLX at
`min(6 GB, physicalMemory / 2)`, so that MLX applies backpressure — waiting for buffers to
free — rather than the app being jetsam-killed. That was a flat 6 GB until it was run on a
6 GB iPad (an iPad Pro 11-inch 2nd generation, `iPad8,10`), where a ceiling equal to total
RAM is one MLX can never reach: it never waits, and the kernel arrives first. The symptom is
not an error but a disappearance, with `MTLCompiler ... XPC_ERROR_CONNECTION_INTERRUPTED` in
the log as the shader compiler service goes down alongside the app. Half of physical pins
8 GB and larger devices to the same 6 GB as before and gives smaller ones a budget they can
actually stay inside; at 6 GB of RAM that budget lands near the 3.31 GB peak, so expect slow
or a reported failure there rather than success.

Same code, same corpus, same model. First launch downloads the same 2.2 GB of
`mlx-community/Qwen3-4B-4bit`, and the header's percentage is the only thing to watch
until it lands.

The layout forks on **`horizontalSizeClass` at runtime**, and deliberately not on
`UIDevice.userInterfaceIdiom`. An iPad in Slide Over, in a 1/3 Split View or in a narrow
Stage Manager window is compact and has to behave like a phone, and it crosses back and
forth *while the app is running*, so a decision made once from the device would leave a
three-pane layout stretched across 320 points. `ContentView.isRegularWidth` is the one
place that is read, and it is hardcoded `true` on macOS — where
`ShakespeareReaderApp`'s 1080 floor makes a window narrower than the three panes
unreachable — so the panes, the toolbar and `revealReader()` read the same on both
platforms instead of being `#if`-gated at each use.

At a **regular width** an iPad gets the Mac's behaviour: the navigator is a column that
stays put rather than something a tap pushes past, the commentary is the persistent
trailing column the `.inspector` presents at that size class, and both have toggles —
`NavigationSplitView`'s own sidebar button, which drives the `columnVisibility:` binding
this file provides, and a ⌘2 `sidebar.trailing` toggle that is the Mac's own button routed
through the same `setCommentary(_:)`, so closing the column stops live generation and
reopening picks the pending selection back up. `revealReader()` is inert there, which is
load-bearing rather than tidy: `openScene(_:in:)` calls it *above* its "is this a different
scene" guard, so without the guard every tap in the scene list would collapse the sidebar.
The determinate download bar comes back too, because the constraint on it was ever the
bar's width and not the platform. The annotation state machine itself needs no fork at all:
at a regular width the inspector reads the persisted `showsCommentary` exactly as the Mac's
third pane does, and `showsGloss` becomes an inert `Bool` exactly as it already was there.

At a **compact width** — every iPhone, and an iPad multitasking narrow — the layout is the
desktop one folded up, unchanged: the scene list and the reader are the two columns of a
`NavigationSplitView`, which that size class renders collapsed, so the
navigator is a push and the system back button is ⌘1. `columnVisibility:` is ignored once
the split view collapses, so `preferredCompactColumn:` is what pushes and is what SwiftUI
writes `.sidebar` back into on the pop, and a tap on the scene already open has to be an
explicit reveal rather than the no-op it is on the Mac. The two bindings sit over the one
stored `showsNavigator` flag, and the compact setter guards on `!isRegularWidth` so a write
arriving *during* a size-class transition cannot set the flag from a column preference the
split view is no longer acting on. The find field sits at the top of
that `Plays` column, the same one the Mac has. The commentary is an `.inspector`,
which at this size class presents as a **sheet**, pinned to `.medium` so the verse stays
on screen above the gloss, which is the part of the three-pane layout worth keeping, and
draggable to `.large` to read a long one. Selection is by touch: tap a line, double-tap
for the whole speech, and **press and hold, then drag** to sweep a passage. That last one
is not a flourish. A touch pan *is* a `DragGesture`, so the bare per-row drag the Mac uses
would win the vertical gesture against the enclosing `ScrollView` and the scene would not
scroll at all. On a touch device the sweep is therefore not a SwiftUI gesture: it is a
`UILongPressGestureRecognizer` on the scroll view itself, in `SweepRecognizer`, which
recognizes only after the finger holds still and keeps reporting its location afterwards.
No composition of SwiftUI gestures does both jobs — every shape of `DragGesture` on the
rows stopped the scene scrolling, including behind `LongPressGesture.sequenced(before:)`
and including with `.simultaneousGesture`, and masking the drag off with
`including: .none` until a long press armed it restored scrolling but never started a
sweep, because a gesture that was masked when the finger landed is not handed the touch
already in flight. Only hardware and a Simulator *click-drag* show this: a trackpad scroll
on the Simulator is a wheel event and never contends for the touch, which is why this
survived a Simulator pass. ⌘R, ⌘C and Esc become Regenerate, Copy passage and
Clear selection in the reader's `⋯` menu. There is no ⌘2 and no commentary toggle at this
width: the sheet rises when a passage is selected, so there is no "is the pane showing"
preference to keep — which is exactly what stops being true when the pane becomes a
sibling column and the toggle comes back. Swiping the sheet away only lowers it — the
gloss, its transcript and its
live `ChatSession` stay, so a re-tap of the still-highlighted passage brings the whole thing
back with no model work. A swipe is not Esc; Clear selection is. The one thing it does stop
is generation still in flight, because cancelling that discards the session the follow-ups
would run on. The back button lowers it on exactly those terms, which it has to: the sheet
is attached to the split view rather than to the reader column, so a pop that left it up
would strand it over the scene list, citing a passage from the scene just left.

Both widths get a **reading measure**, `ReaderTypeface.measure`: 620pt at the default size
step, scaled by the same `scale` the type is, so it grows with the reader's size step, with
their chosen family's optical correction, and with Dynamic Type — a measure held fixed
while the verse grows would get narrower in ems, which is the thing it exists to avoid. It
caps the heading and the verse content *separately*, inside the `ScrollView` and never on
the pane: capping the pane or the scroller itself would put the scroll indicator at the
measure's edge and leave the margins outside the scroller, so a pan beside the verse on a
wide iPad would not scroll the scene. It applies on macOS too rather than under an `#if`,
where at the reader pane's 640 `idealWidth` it is a no-op — 620 clears the ~590pt of text
left after the gutter and the trailing padding — and only bites once the window is widened
with the side panes hidden, which is the same problem for the same reason.

Divergences worth knowing about before they look like bugs:

- **`FoundationModelsIntegration` is on.** The Xcode project model has no way to express
  `traits: []`, so unlike the SwiftPM build described under Notes below, *both* Xcode
  destinations compile `MLXFoundationModels`. The app never calls into it and the whole
  target is behind `@available(iOS 27.0, macOS 27.0, visionOS 27.0, *)` plus
  `#if canImport(FoundationModels)`, so the cost is build time, not behaviour.
- **The typefaces are not the same three.** Of Caslon, Baskerville and Garamond, iOS
  ships only Baskerville: there is no iOS Big Caslon, and Garamond is not in an iOS
  downloadable font catalog. The menu offers Baskerville, Hoefler Text and Palatino
  instead, both replacements taken from the runtime's `System/Library/Fonts/AppFonts`,
  which is the set actually exposed to apps. Iowan Old Style was the obvious third choice
  and is **not** in it: iOS ships the file, `CTFontManagerCopyAvailableFontFamilyNames`
  does not return it, and the row rendered in SF while offering a download that does not
  exist. The cases are not renamed per platform, so a `readerFont` of `caslon` synced from
  a Mac falls back to the system face rather than silently substituting a face it is not.
- **The reader pane is not focusable on iOS.** `.focusable()` exists there only to receive
  the three responder-chain commands macOS uses, and asking for focus without them made
  the reader first responder with no input view, which raised the software keyboard over
  the bottom third of the play every time a navigation-bar menu opened.
- **The pointer and keyboard interactions stay macOS-only, on iPad included.** The layout
  is what the size class forks; interaction on an iPad is interaction on an iPhone, at
  either width. So hover word marking, the word context menu, `DictionaryLookup`,
  arrow-key selection, shift-click extend and the pointer `DragGesture` are all still
  behind `#if os(macOS)`, even on an iPad with a Magic Keyboard and a trackpad attached.
  Un-gating them is a question about which gesture wins against `SweepRecognizer`, which
  is the part most likely to take touch scrolling down with it, so it is deliberately not
  a layout change.

On the **Simulator** everything except the model works: all 35 plays load from the bundled
`Plays/`, the navigator pushes, tap and double-tap and press-and-hold-then-drag all select,
scrolling still works — but scroll it by *click-dragging*, not with the trackpad, since a
trackpad scroll there is a wheel event and proves nothing about a touch pan, as the gesture
note above records — the typeface menu changes the verse, and Dynamic Type at the largest
accessibility size scales the custom faces once rather than twice. It does the same for the
compound case, largest accessibility size *and* the Largest size step, on the system face
too — measured there, and the measurement is why the system face has its own sizing path:
`Font.system(size:)` is fixed and ignores Dynamic Type entirely, so before that path
existed, picking any size step froze the reader's Larger Text setting. The header shows a
red **Load failed** instead of a model, and that is deliberate: MLX has no Metal device on
the Simulator, and finding that out is not a recoverable error. The first touch of any
`MLX.Memory` knob constructs `mlx::core::metal::Device`, which `abort()`s from C++ where no
Swift `catch` can reach it. Unguarded the app dies on launch, taking the whole UI with it, so
`AnnotationService.load()` asks `hasMLXDevice` first and refuses with a message. Annotating
needs a device.

## Latency

Measured on an M4 Max with `mlx-community/Qwen3-4B-4bit`, over the thirteen sample
passages `--benchmark` walks:

| passage | prompt tok | prefill tok/s | time to first token | decode tok/s | words |
|---|---|---|---|---|---|
| Hamlet · I.i.1-6 | 526 | 1374 | 0.40 s | 140.2 | 108 |
| Hamlet · I.v.88-91 | 680 | 1541 | 0.45 s | 138.2 | 116 |
| Hamlet · II.ii.250-262 | 978 | 1550 | 0.64 s | 132.5 | 164 |
| Hamlet · III.i.62-96 | 1023 | 1613 | 0.65 s | 131.2 | 176 |
| Hamlet · IV.iv.20-24 | 703 | 1607 | 0.45 s | 136.3 | 116 |
| Hamlet · V.i.1-12 | 699 | 1591 | 0.45 s | 136.5 | 166 |
| Macbeth · I.iii.38-48 | 768 | 1622 | 0.48 s | 136.5 | 119 |
| Macbeth · II.iii.1-20 | 865 | 1569 | 0.56 s | 133.6 | 166 |
| Macbeth · V.v.17-28 | 710 | 1512 | 0.48 s | 136.0 | 105 |
| Macbeth · I.vii.1-28 | 835 | 1396 | 0.61 s | 134.3 | 124 |
| Romeo and Juliet · I.Pro.1-14 | 523 | 1435 | 0.37 s | 139.5 | 129 |
| Romeo and Juliet · I.v.96-109 | 870 | 1537 | 0.58 s | 133.3 | 133 |
| Romeo and Juliet · II.ii.1-10 | 557 | 1503 | 0.38 s | 137.6 | 113 |

Every row prefills at 1,370-1,620 tok/s, mean 1,527. Peak memory 2.98 GB, resident
around 2.6-3 GB.

**A ~200 tok/s prefill figure from a 30B MoE VLM does not transfer.** That was 30B
through a 52-layer stack; this is a 4B dense model prefilling
about eight times faster. The app was designed to shed context if prefill
disappointed — preceding 15 lines → 8, personae limited to the selection, drop the
synopsis — and none of that was needed. At 0.5 s to first token the annotation
appears about as fast as a footnote you look down at.

Prompt length runs 523-1,023 tokens, mean 749 without a scene summary and roughly
850 with one. Early modern verse runs about **1.4 Qwen3 tokens per word** — elisions
(`o'er`, `on't`) and curly apostrophes split more than modern prose — so do not
budget this at 0.75 words per token. `--show-prompt` prints the assembled prompt
with its exact count from
`tokenizer.applyChatTemplate(messages:tools:additionalContext:)`.

### Turn 2 is cheap, but not as cheap as it looks like it should be

The commentary, the follow-up list, and every tapped question share one
`ChatSession` per passage, so the scene context is prefilled once. Turn 2 prefills
259-366 tokens against a 523-1,023-token turn 1:

| turn 1 prompt | commentary | turn 2 prompt |
|---|---|---|
| 523 | 129 words | 295 |
| 710 | 105 words | 259 |
| 870 | 133 words | 294 |
| 1023 | 176 words | 366 |

Turn 2's cost tracks the **commentary length**, not the turn-1 prompt length: the
600-900 token context block is reused, and what gets re-prefilled is the assistant's
own answer. The likely cause is that Qwen3 emits an empty `<think></think>` block
even under `enable_thinking: false`, which the framework strips from the recorded
text — so the re-rendered transcript diverges from the cached tokens at the assistant
turn and everything from there on is replayed. It costs about 0.2 s, so it was not
worth chasing further; a tapped question still answers in well under half a second.

A cache hit costs **1 ms** and no model work at all.

## What it does

- **Corpus.** 35 plays, parsed from Project Gutenberg into checked-in JSON by
  `tools/build_corpus.py`. Nothing at runtime depends on the script.
- **Selection is a range of line indices**, not characters. SwiftUI's `Text` does not
  expose a selected character range, and a play is line-structured anyway — line
  numbers are what the context window, the cache key, and the citation are all built
  on. Click, shift-click to extend, double-click for the whole speech, drag, arrows,
  Esc to clear, ⌘C to copy with the citation, ⌘R to regenerate. The arrows roll into the
  adjacent scene of the same play at either edge of one, and shift-arrow stops at the
  edge instead, since a selection is scene-scoped. Clicking a line is what hands the
  keyboard back to the reader after the navigator has it — the selection band is grey
  while the arrows are pointed somewhere else.
- **A single word is hover-and-right-click**, macOS only for now. Moving the pointer along
  a line underlines the word under it, and right-clicking that word offers the system
  dictionary panel over it (only when the dictionary actually has an entry — `undiscover’d`
  and `well-a-day` are offered nothing rather than an empty panel, while `quietus`, `argal`
  and even `’tis` have real entries), an *Explain “word” in context* that rides
  the same follow-up turn a tapped question does, and Copy. The word is resolved by laying
  the line out a second time with CoreText and hit-testing that, which works because the
  verse carries no tracking and macOS pins Dynamic Type at `.large`; the underline is there
  partly so a near-miss is visible before you commit to it. What this is *not* is
  character-level drag-highlighting of the verse: `.textSelection(.enabled)` there would
  put system character selection in direct competition with the per-row drag that sweeps a
  passage, which is the app's primary interaction, so hovering a word is what stands in for
  highlighting one. The echoed lines in the commentary pane *are* selectable, on both
  platforms, since nothing is competing for the drag there. Right-clicking deliberately
  does not select the line: selecting commits a generation 350 ms later, which is too heavy
  a side effect for opening a menu, so only Explain moves the selection — and when it has
  to, the word question waits for the passage to be glossed and then asks itself.
- **Context is deterministic**, from the play's own structure: no embeddings, because
  the act/scene/speaker hierarchy is a better index here and it is exact.
- **One scene at a time.** This bounds rows to under 1,000 (Love's Labour's Lost V.ii
  is the worst case at 967, then The Winter's Tale IV.iv at 894), makes every selection
  intrinsically scene-scoped, and keeps the cache key trivial. The rows are in a
  `LazyVStack`, so only the visible ones are built. The navigator is how you move, and
  it is an **accordion**: 35 play rows, one play open at a time, and inside it the act
  you are reading, so the scene you are on is always the one on screen. ⌘F focuses the
  find field at the top of the pane, which filters by play title (`macb`, `henry iv`,
  `loves labours`) or by scene setting (`churchyard` finds the grave-diggers); Esc
  clears it, and clicking a line is what hands the keyboard back to the play.
- **Scene summaries** are generated in the background when a scene opens, in their
  own throwaway session. A selection cancels the prewarm rather than queueing behind
  it, and proceeds without a summary — the summary never blocks an annotation.
- **Four typefaces and five sizes for the play**: the system face, Caslon, Baskerville,
  Garamond, and a Small / Default / Large / Larger / Largest ladder, both picked from the
  same `Aa` menu in the header and both remembered between launches. Default reproduces
  the shipped rendering exactly, by taking the same code path rather than an
  arithmetically equivalent one. It sets the **play text only** — the navigator, the
  commentary, the header and the status strip stay on the system face. So does the
  line-number gutter, which is the one part that only half opts out: a serif family has
  no monospaced digits so the *face* stays the system's, but the *size* follows the
  reader's step, and the font and the frame width move together, because that width is a
  budget for three digits' advances.
- **Diagnostics are off by default.** The model capsule, the load check and the latency
  numbers under the commentary are for working on the app, not for reading a play; the
  header's `⋯` menu turns them on and the preference is remembered between launches.
  Two things show regardless, because hiding them makes a working app look broken: the
  first-launch download progress, and failures.
- **Your place is remembered between launches**: the scene you were reading comes back
  with the passage you had selected still highlighted and scrolled into view, and
  whichever panes you had hidden. The navigator comes back open at the play and act
  being read, scrolled to that scene, which it works out from the position rather than
  storing. A restored passage is not annotated on its own: click it or press ⌘R for
  that. A corpus rebuilt under a stored position keeps the scene and drops the
  highlight rather than putting it over different lines.

All 35 plays are in the reader; the model has clearly read the famous ones before,
which is worth remembering when judging output. `--model mlx-community/Qwen3-8B-4bit`
swaps in a larger model for comparison — test that on a *deliberately obscure*
passage, because "To be or not to be" is in every training corpus on earth and tells
you nothing. Timon of Athens and the Henry VI plays are the useful end of the range for
that. For scale, `--model mlx-community/Qwen3-0.6B-4bit` prefills at 8,426
tok/s and decodes at 394 tok/s in 0.78 GB, and writes noticeably vaguer annotations.

## Corpus

```bash
python3 tools/build_corpus.py --all --out Sources/ShakespeareReader/Resources/Plays
python3 tools/build_corpus.py --all --verify              # stats only, writes nothing
python3 tools/build_corpus.py --slug romeo-and-juliet --from-file /tmp/pg1513.txt --dump-scene 1.0
```

`--all` is 35 sequential requests, and gutenberg.org answers a few of them with `504
Gateway Time-out` on most runs, so each fetch retries with backoff and rejects a
response missing the PG end marker. Without that the run died partway and left the
resource directory half updated, with the exit code as the only sign. A clean rebuild
takes well under a minute.

Run it a few times in quick succession, though, and gutenberg.org starts answering `503
Service Unavailable` for several minutes — 35 files per run is enough to get rate
limited, and no retry budget survives that. `--from-file` with a `--slug` is the way to
work on the parse itself without refetching; the checked-in JSON means nobody needs the
network to build or run the app.

The generated JSON is **checked in**, so the app builds and runs with no network for
anyone else. The script exists to make the parse reproducible and auditable, not as a
build step. Every pattern lives in one `PATTERNS` dict at the top of the file.

All 35 plays come from Project Gutenberg's **1500-1542 series**, one transcription
lineage, which is what makes a single set of patterns viable at all. The catalog's
other Shakespeare families are not interchangeable with it: 1100-1137 and 1765-1802 are
separate transcriptions, 2235-2270 is the First Folio, and 100 is the complete works in
one file. Three plays in the series are still missing — Troilus and Cressida, Pericles
and The Two Noble Kinsmen — and `Resources/Plays/NOTICE.md` records why for each.

What the parse gets, measured on the real files:

| | plays | acts | scenes | speech lines | speech headings | directions | personae |
|---|---|---|---|---|---|---|---|
| whole corpus | 35 | 175 | 707 | 96,902 | 29,067 | 5,902 | 933 |
| Hamlet | | 5 | 20 | 3,817 | 1,137 | 243 | 25 |
| Macbeth | | 5 | 28 | 2,329 | 649 | 168 | 27 |
| Romeo and Juliet | | 5 | 26 | 3,015 | 840 | 199 | 27 |

0.00% of body lines are unclassified in **any** of the 35, and every speech heading in
the body produces a speech. `--selftest` asserts the act, scene and speech-heading
counts for every play by name, so a parser change that moves any of them fails there
rather than in the reader; the three above additionally assert their collective
speakers, which is a reading of the play rather than a count.

Seven scene-0 chorus blocks exist across three plays: Romeo and Juliet's two, King
Henry V's four, and King Henry VIII's Prologue. They are labelled `Prologue` in the
navigator and cited `I.Pro`.

Details that decide the parser, all of them observed rather than assumed:

- **The files are CRLF.** Every `$`-anchored pattern breaks silently otherwise.
- **Scene headers are not reliably at column 0.** Hamlet I.i is; I.ii through I.v
  carry a leading space.
- **The Contents block cannot be skipped by column position.** Macbeth's Contents has
  `ACT I` at column 0, identical to its body header 85 lines later. The body scan is
  anchored after the `Dramatis Personæ` line instead, at the first act **or chorus**
  header below it. Twelfth Night is why that header may carry a trailing period: its
  body sets `ACT I.` where its Contents sets `ACT I`, so requiring a bare header matched
  only the Contents — which in that file sits *above* the personae block, leaving
  nothing to match after it and failing the play outright.
- **The personae block does not always end where it says it does.** Hamlet closes it
  with `SCENE. Elsinore.` and Macbeth with `SCENE: In the end of the Fourth Act`, but
  The Winter's Tale sets that summary in title case and As You Like It has no summary
  line at all, only the sentence `The scene lies first near Oliver's house`. Both used
  to run on into the body and file the entire play as cast: 2,930 personae entries for
  As You Like It and 3,250 for The Winter's Tale, against 25 and 28 real ones. This is
  the quietest failure in the parse — a cast list holding the whole play still decodes,
  still renders, and poisons only the prompt's `WHO THEY ARE` block — so the block now
  stops at the body header as well as at the summary.
- **Speakers are not gated on Dramatis Personæ.** The text has collective and
  numbered speakers that never appear there — `ALL.`, `BOTH MURDERERS.`, `DANES.`,
  `FIRST CLOWN.`, `APPARITION.` Gating would have silently dropped those speeches, so
  the pattern is accepted and `--verify` *reports* unresolved tokens instead of
  failing on them (21 in Hamlet, 22 in Macbeth, 12 in Romeo and Juliet, all of them
  genuinely absent from the personae list). Across the whole corpus this runs from 0 in
  As You Like It to 62 in Coriolanus and 60 in Richard III; the histories are full of
  numbered Soldiers, Messengers and Lords. Only the first three plays have curated
  `ALIASES`, so elsewhere a speaker like Coriolanus's `AEDILE` is named in the prompt
  from its speech token but carries no personae blurb.
- **A heading is not always one all-caps word.** Hamlet and Macbeth each share a line
  between speakers (`HORATIO and MARCELLUS.`, `MACBETH, LENNOX.`), set three
  collectives in title case (`All.`, `Both.`, `Danes.`), and drop the period off one
  `BARNARDO`. The bare token is honoured only for a name already seen speaking, so the
  rule cannot invent a speaker. Missing these is worse than it looks — see the next
  point.
- **A blank line does not close a speech.** The text interrupts a speech with a
  blank-delimited unbracketed direction (`Re-enter Ghost.`) and then resumes the
  *same* speech with no repeated heading. Clearing the speaker at a blank filed
  everything after such an interruption as one stage direction: 64 paragraphs in
  Hamlet and 37 in Macbeth, about 251 and 78 verse lines, taking Claudius's prayer,
  "How all occasions do inform against me", "Is this a dagger", "The raven himself is
  hoarse" and Ophelia's songs with them — unnumbered, unattributed, uncitable, and
  labelled a direction in the annotation prompt. So the speaker survives a blank, and
  an unbracketed line is a direction only if it opens with the `DIRECTION_OPENERS`
  vocabulary and is not the first line under a heading. That guard is what keeps the
  two verse lines in these plays that open with the vocabulary — Ophelia's `The King
  rises.` and Siward's `Enter, sir, the castle.` — as verse; `--verify` lists every
  speech line matching the vocabulary so extending it stays deliberate.
- **The opener boundary is not `\b`.** A curly apostrophe is a word boundary, so
  `Alarum\b` matches `Alarum’d by his sentinel, the wolf,` and takes the last eight
  lines of "Is this a dagger" with it; the plural has to be spelled out for the
  reverse reason, since `Alarums. Enter Macduff.` is a direction.
- **The PG footer has to be stripped.** Leave it in and `DAMAGE.` parses as a speaker
  in every one of the 35 files, from the license text. `--selftest` checks for that
  speaker per play, since the footer is identical in all of them.
- **A bracketed aside can open a verse line.** `[_Aside._] A little more than kin,
  and less than kind.` Treating the whole line as a stage direction drops the verse
  entirely, which is what happened to 22 speeches in Hamlet and 4 in Macbeth before
  the leading-direction split existed — including that line and every one of
  Ophelia's `[_Sings._]` songs. Mid-line asides stay in the verse, because splitting
  those would fragment the line the reader selects.
- **A chorus block has no scene header, and one of them sits outside its act.** Romeo
  and Juliet's `THE PROLOGUE` is between `Dramatis Personæ` and `ACT I`; the Act II
  Chorus is under `ACT II` but above `SCENE I.` Anchoring the body on the first act
  header dropped "Two households, both alike in dignity" so completely it was not even
  counted as unclassified, and the Act II sonnet was the entire 0.53% of body lines
  this play used to leave unclassified: one sonnet, not noise. Both are now scene 0 of
  the act they open, which the reader labels `Prologue` and cites `I.Pro.1-14`. The
  Prologue is filed under Act I, as printed editions do, which is also why an `ACT`
  header for the act already open is a no-op rather than a sixth act.
- **A heading is not always on its own line.** Two lines in Romeo and Juliet put the
  heading and its first verse line together: `ROMEO. Nurse, commend me to thy lady and
  mistress.` and `THIRD WATCH. Here is a Friar that trembles, sighs, and weeps.` Both
  were attributed to whoever spoke last, with the heading left sitting inside that
  speaker's own text, and `THIRD WATCH` has no other heading anywhere in the play, so
  the Watch's third man never entered the cast at all. The rule is tested only at the
  start of a paragraph and after the scene header has had its turn, since `SCENE I. A
  public place.` is the same shape.
- **The direction vocabulary is per transcription, not per parser.** `DIRECTION_OPENERS`
  had to grow two entries for Romeo and Juliet: ` Juliet appears above at a window.`,
  which is the balcony scene's own direction and became Romeo's speech line 2 without
  it, shifting every citation in II.ii by one; and ` Musicians waiting. Enter
  Servants.` Both carry their second word deliberately. `Juliet` alone would take
  `Juliet, the County stays.`, which is Lady Capulet's verse. `--verify` prints
  `dir-shaped verse` for exactly this reason: for these three plays the expected output
  is one line each (Ophelia's `The King rises.`, Siward's `Enter, sir, the castle.`, and
  nothing in Romeo and Juliet), so a real direction read as verse shows up as a fourth.

  **This is the one known defect in the 32 plays added since.** The vocabulary was not
  extended for them, and the statistics cannot show what it misses: an unbracketed
  direction with an unlisted opener, arriving while a speaker is still open, is filed as
  that speaker's verse — numbered, citable, and never counted as unclassified. A scan
  puts it at roughly 34 lines, concentrated in the histories' battle directions
  (`Dead March.`, `March.`, `Drum and colours.`, `Tucket.`, `Noise within.`, `Fight.
  Excursions.`, `Trumpet sounds.`, `Music plays.`). Adding them means minding the same
  trap: bare `Music` would take Titania's `Music, ho, music, such as charmeth sleep.`
  and bare `Within` would take `Within two hours.`, so those need their second word the
  way `Juliet appears` does.

Line numbers are assigned **sequentially within each scene over speech lines only**.
These are not Folger or Arden numbers — those count a verse line shared between two
speakers once — so the JSON records `"numbering": "sequential-within-scene"` and every
citation in the UI reads `Hamlet · III.i.62-96 (this edition)`.

## Verifying

```bash
swift run -c release ShakespeareReader --selftest    # no model, no network
swift run -c release ShakespeareReader --metal-check # one array on the GPU; no model, no network
swift run -c release ShakespeareReader --show-prompt # assembled prompts + exact token counts
swift run -c release ShakespeareReader --benchmark   # the latency table above
swift run -c release ShakespeareReader --greedy      # temperature 0, for prompt A/B work
swift run -c release ShakespeareReader --diagnostics # the capsule, the load check and the
                                                     # status strip on for this launch only
swift run -c release ShakespeareReader --benchmark --passage hamlet:3.1:62-96
```

An installed copy takes the same flags, `shakespeare-reader --selftest` and so on. The
two that need neither the model nor the network are what `brew test` runs:

```bash
shakespeare-reader --selftest      # "selftest: all checks passed"
shakespeare-reader --metal-check   # "metal: ok"
```

So does the bundled Mac app, by running the executable inside it rather than `open`ing it:

```bash
ShakespeareReader.app/Contents/MacOS/ShakespeareReader --selftest
ShakespeareReader.app/Contents/MacOS/ShakespeareReader --metal-check
```

Every flag works unchanged in a bundle. `EntryPoint.main()` handles them and exits before
`ShakespeareReaderApp.main()`, so no window opens, and `Bundle.main` still resolves to the
enclosing `.app` because CFBundle walks up from the executable path to find it.

`--diagnostics` does not write the preference, so it is the way to look at the numbers
once without turning them on for good; `--benchmark` is still how the table above is
produced.

`--metal-check` evaluates one three-element array and prints `metal: ok`. It exists
because `--selftest` never touches the GPU, so it passes on a build whose Metal
kernels were never compiled; that failure otherwise waits for the first annotation.

`--selftest` is model-free and covers what breaks silently: every play decodes with
contiguous per-scene numbering and the exact counts above; 14 `LineSelection` cases
(shift-click backwards, drag reversal, double-click across an interleaved direction,
clamping at scene edges); the on-stage scan against four real scenes that each broke a
naive version of it; the three Romeo and Juliet cases that guard the parser edits above
(both Chorus blocks present as scene 0 with 14 `CHORUS` lines each and the right first
line, the balcony direction sitting between verse lines 1 and 2, `THIRD WATCH` owning
its recovered inline heading); 11 follow-up parser cases; one **golden `PassageContext`
render** compared against a checked-in string, which is what catches prompt drift; and
the typeface picker's inputs, including the CoreText italic probe that decides whether
a stage direction gets a real italic cut or a synthetic one; and the reader's size
ladder — that every step round-trips through its stored raw value, that exactly one step
is neutral and that its multiplier is exactly 1, that the multipliers rise along the
declaration order the menu lists them in, that every role at Default returns the *same
`Font` value* the app returned before there was a size setting — comparing rather than
measuring, which is the one place the no-`Font` rule above is worth breaking — while a
non-default step returns a different one, that the point measures Default derives come out
byte-for-byte unchanged, and that two size steps or two Dynamic Type categories of the same
face compare *unequal*, which is what makes the reader keep their place when the type
resizes under them. A deliberate prompt change
means regenerating that string alongside a `Prompts.version` bump — the self test
prints the replacement.

## Notes

- **`enable_thinking: false`** on every session, via
  `additionalContext: ["enable_thinking": false]`. Without it Qwen3 reasons at length
  before the first visible token — a silent window with nothing on screen to account for
  it. With it there is no visible `<think>` block.
- **Sampling** follows Qwen3's own recommendation for non-thinking mode
  (`temperature: 0.7, topP: 0.8, topK: 20`), except the scene summary, which runs at
  0.3 because it is meant to be dull and accurate.
- **`Memory.cacheLimit` is 256 MB**, not the 2 GB a 30B VLM wants — that figure is sized
  for a 20 GB model churning hundred-MB image activations. 256 MB is what
  `MLXFoundationModels` picks for a model this size.
- **Cancellation.** A new selection cancels the previous generation, waits for it, and
  then waits again on `session.synchronize()` for the cache lock; the session is then
  discarded entirely, because a cancelled generation invalidates `ChatSession`'s token
  ledger. Partial output is never cached.
- **On stage is approximate** and the prompt says so. It comes from scanning `Enter` /
  `Exit` / `Exeunt` directions, which are written for actors, not parsers. A speech
  line adds its speaker (someone speaking is necessarily present) and a later exit
  removes them again. It follows the text faithfully even when the text is coy: in
  III.i the King and Polonius withdraw under an `Exeunt`, so the scan drops them,
  though they are in fact eavesdropping.
- **The prompt is ordered scene-invariant sections first**, then the passage window,
  which is what would make a per-scene prefix cache possible later
  (`ChatSession(cache:state:)`) without rewriting the prompts.
- **The annotation cache is one file per passage**, under
  `~/Library/Application Support/ShakespeareReader/`. A hit requires the schema
  version, the prompt version, the model id **and** a SHA-256 of the selected line
  text to match. The digest is what stops a re-parsed corpus with shifted line
  indices from serving an annotation of different lines — the one failure mode that
  would otherwise be invisible. The path does not include the model, so switching
  models overwrites entries rather than keeping both; nothing wrong is ever served,
  you just pay for a regeneration.
- **Quality.** A 4B model glosses famous passages well and occasionally reaches for a
  plausible-sounding etymology on an obscure word. The instructions tell it to say so
  rather than invent, ⌘R regenerates, and `--model` makes a larger comparison one
  flag. It also overshoots the 90-150 word rule on about a third of passages (up to
  176 words); prompt iteration is where the remaining time would go.
- **Reader typefaces are what the system actually ships**, which is less than it looks.
  **Garamond is not installed on macOS** — it is a downloadable Apple font asset, so
  picking it runs `CTFontDescriptorMatchFontDescriptorsWithProgressHandler`, which
  activates the asset in *session* scope: the family is available to the process within
  seconds and a later launch finds it already there (measured landing in
  `/System/Library/AssetsV2/com_apple_MobileAsset_Font8/`). Until it lands the play
  renders in the system face, because the resolved family name stays nil rather than
  leaning on `Font.custom`'s undocumented fallback. **Weight is not a usable channel**,
  because Big Caslon is a single face (`BigCaslon-Medium`, no italic and no bold) and is
  the only Caslon on macOS: `.weight(.semibold)` resolves to the nearest available face,
  so one code path would give Baskerville real contrast and Caslon none. Size, tracking,
  italic and colour carry the hierarchy instead, and stage directions in Big Caslon get
  an explicit ~12° shear in the font matrix's `c` slot — verified necessary, since
  SwiftUI's `.italic()` leaves that face upright — which slants the glyphs without
  touching advance widths, so a direction wraps exactly where its upright twin would.
  The picker is a `Menu` of `Toggle`s rather than `Button`s because an `NSMenuItem` has
  one image slot: a hand-drawn checkmark would displace the download or error glyph on
  the row the reader just picked, which is the one row whose state matters.
- **The size ladder multiplies the system's point sizes**, and the awkward half is the
  system face, not the serifs. `Font.body` is a *text style*, not a point size: there is no
  arithmetic to do to it, so Default short-circuits to the bare style and only a
  non-default step computes a number and hands it to `Font.system(size:)`. That costs two
  things. The semibold in `.headline` is part of the style and not part of the size, so it
  has to be restated by hand or the act heading loses its weight and reads as verse. And
  `Font.system(size:)` turns out to be **fixed** — it does not follow Dynamic Type, which is
  not documented anywhere and does not match `Font.custom(_:size:relativeTo:)`. Measured on
  the Simulator: with the size step at Large the verse stayed 24pt tall from the `.large`
  category all the way to AX5, while the Default reading grew from 22pt to 69pt. So the
  system face is measured at the category actually in force rather than at `.large`, from a
  `DynamicTypeSize` threaded onto the typeface, and the custom faces keep the `.large`
  measurement because `relativeTo:` is what scales theirs and would otherwise scale it
  twice. Measuring each style separately also keeps each one's own metric curve, so the
  caption-derived speaker heading holds its proportion against the verse at accessibility
  sizes. There is deliberately **no ⌘+ / ⌘−**: the app has no menu bar of its own, a
  `keyboardShortcut` inside a borderless-button `Menu` is not reliably registered, and the
  shape that does work — hidden zero-opacity buttons in the header — takes three of them
  (⌘-, ⌘= and ⌘+) for the least frequently changed preference in the app.
- **This is a sibling SwiftPM package** with a local path dependency on the enclosing
  checkout. It sets `traits: []` on that dependency, which turns off the default
  `FoundationModelsIntegration` trait: the app never touches Apple's FoundationModels
  adapter, and `MLXHuggingFace` pulls that target in only when the trait is on.
- **The Xcode app target compiles the same sources directly**, from
  `App/ShakespeareReader.xcodeproj`. It is **one multiplatform target**, not one per
  platform: `SDKROOT = auto`, `SUPPORTED_PLATFORMS = "iphoneos iphonesimulator macosx"`,
  `TARGETED_DEVICE_FAMILY = "1,2"` for iPhone and iPad, and the whole platform difference
  carried by per-SDK `INFOPLIST_FILE` and
  `CODE_SIGN_ENTITLEMENTS` (plus `ARCHS[sdk=macosx*]`). `SUPPORTS_MACCATALYST`,
  `SUPPORTS_MAC_DESIGNED_FOR_IPHONE_IPAD` and `SUPPORTS_XR_DESIGNED_FOR_IPHONE_IPAD` are
  all `NO`, and the first two for the same reason: the target already builds natively for
  macOS, so either would ship a second, worse Mac app. A second target would have
  duplicated all three build phases, the four synchronized groups and the six package
  product dependencies, and needed a second scheme, to express a difference that is
  four build settings. The sources were already shared, since `#if os(macOS)` has
  carried both layouts from the start. Note that the *sibling* `Package.swift` is what
  is outside this project's build graph; the *root* package of the enclosing checkout is
  very much inside it, as a local package reference, which is why the Xcode builds
  compile `MLXFoundationModels` and the SwiftPM one does not. `Annotation/`, `Corpus/`,
  `Reader/` and `Platform/` are
  file-system-synchronized groups, so adding a file to any of them needs no project edit.
  The four top-level `.swift` files are listed individually, though, so a *new* top-level
  file or a new subdirectory does need one. That split is deliberate. A single synchronized group over
  `Sources/ShakespeareReader` also swallows `Resources/Plays`, and Xcode has no way to
  exclude a plain directory from one: `membershipExceptions` works for a bundle like
  `Documentation.docc` but recurses into an ordinary folder, so every play was copied
  **twice**: once flattened into the `.app` root by the synchronized group, and once as
  the `Plays/` folder reference `CorpusLoader` needs. Keeping the corpus out of the
  synchronized tree is what preserves "adding a play is dropping a file in", which is the
  ethos that matters most here.
- **`Bundle.module` in an executable target** resolves to a `.bundle` beside the
  binary in `.build/release/`. That is correct under `swift run`; a binary copied out
  on its own leaves its corpus behind, and `CorpusLoader` reports that rather than
  showing an empty library. mlx-swift's `default.metallib` is found the same way, from
  a sibling `mlx-swift_Cmlx.bundle`. Under Homebrew that directory is
  `$(brew --prefix)/opt/shakespeare-reader/libexec`, which is why the formula installs
  the binary and all four resource bundles there and puts a wrapper script in `bin`
  rather than a symlink: a symlink would make the app look for its corpus and its GPU
  kernels in `bin`. In the bundled Mac app neither is beside the executable: both
  `Plays/` and `mlx-swift_Cmlx.bundle` land in `Contents/Resources` while the binary
  sits in `Contents/MacOS`, so `CorpusLoader` goes through `Bundle.main` instead, and
  mlx finds its `default.metallib` through its own `Bundle.allBundles` fallback rather
  than by colocation.
- **The unbundled build sets its Dock tile from code.** An asset catalog is only ever
  read out of a bundle, so the `.app` gets its icon with the process uninvolved and a
  bare SwiftPM binary — `swift run`, or the Homebrew install — gets the generic
  unbundled-executable tile no matter how complete the catalog is.
  `ShakespeareReaderApp.init()` therefore loads `Resources/AppIcon.png` out of
  `Bundle.module` and assigns `NSApplication.shared.applicationIconImage`, next to the
  `setActivationPolicy(.regular)` that gives the process a tile to put it on in the first
  place. The visible cost is that the icon appears a beat after the tile does. It is
  guarded on `Bundle.main.bundleIdentifier == nil` rather than on `#if SWIFT_PACKAGE`
  alone: the `#if` is what makes `Bundle.module` exist, but a SwiftPM binary run from
  inside a hand-assembled `.app` has a bundle whose catalog icon is the better one.
  Missing identifier is the test that actually means "no bundle behind me", and the same
  one behind the split preferences above.

## Corpus provenance

Hamlet is Project Gutenberg ebook 1524, Macbeth is 1533, Romeo and Juliet is 1513;
all three are public domain in the United States. Each JSON file records the ebook id,
the URL, the retrieval date, and the SHA-256 of the source text as downloaded. See
`Sources/ShakespeareReader/Resources/Plays/NOTICE.md`, which also records the scene-0
decision behind the two Prologues.

## Icon provenance

The icon is the Chandos portrait, attributed to John Taylor, c. 1600–1610, National
Portrait Gallery NPG 1. The painting is long out of copyright and a faithful photographic
reproduction of a flat public-domain work carries no separate copyright in the United
States, which is why the scan is checked in at `tools/icon/chandos-portrait.jpg` rather
than fetched at build time. `tools/icon/NOTICE.md` records its dimensions and SHA-256, so
the crop is auditable against a specific scan the way each play's is.
