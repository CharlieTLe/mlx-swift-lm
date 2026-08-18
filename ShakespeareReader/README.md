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

## Install

```bash
brew tap CharlieTLe/tap
brew trust CharlieTLe/tap
brew install shakespeare-reader
shakespeare-reader &
```

The `brew trust` step is not optional: recent Homebrew refuses to load a formula from
a third-party tap until you trust it, and `brew tap` does not say so.

There is no `.app` bundle, so the command launches the window and holds the terminal
until you quit it; `&` gives the shell back.

The formula builds from source and the source includes MLX, so this is a compile rather
than a download: about two minutes and 1.6 GB of scratch on an M4 Max, longer on fewer
cores or on a first build, which also clones fourteen packages. It needs:

- **Apple silicon.** MLX has no Intel path.
- **A full Xcode**, selected with `xcode-select`, with the **Metal toolchain**
  component installed. The Command Line Tools ship a `metal` that is a stub and
  cannot compile mlx-swift's GPU kernels; a build without them links and launches
  and then fails on the first annotation with `Failed to load the default metallib`.
  The formula checks for this before building rather than shipping that binary.

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

## Latency

Measured on an M4 Max with `mlx-community/Qwen3-4B-4bit`, over the ten sample
passages `--benchmark` walks:

| passage | prompt tok | prefill tok/s | time to first token | decode tok/s | words |
|---|---|---|---|---|---|
| Hamlet · I.i.1-6 | 526 | 375 | 1.56 s | 140.8 | 100 |
| Hamlet · I.v.88-91 | 680 | 1568 | 0.44 s | 138.6 | 149 |
| Hamlet · II.ii.250-262 | 1006 | 1616 | 0.63 s | 133.4 | 133 |
| Hamlet · III.i.62-96 | 1023 | 1642 | 0.64 s | 133.1 | 137 |
| Hamlet · IV.iv.20-24 | 703 | 1611 | 0.44 s | 138.4 | 108 |
| Hamlet · V.i.1-12 | 699 | 1618 | 0.44 s | 137.8 | 175 |
| Macbeth · I.iii.38-48 | 768 | 1463 | 0.53 s | 137.4 | 140 |
| Macbeth · II.iii.1-20 | 865 | 1588 | 0.56 s | 135.1 | 146 |
| Macbeth · V.v.17-28 | 728 | 1564 | 0.47 s | 138.5 | 99 |
| Macbeth · I.vii.1-28 | 832 | 1631 | 0.52 s | 135.1 | 190 |

The first row is the first request of the process and includes warm-up; every row
after it prefills at 1,460-1,640 tok/s. Peak memory 2.94 GB, resident around
2.6-3 GB.

**The ~200 tok/s prefill figure in `MuseGlimmerDemo/README.md` does not transfer.**
That was a 30B MoE through a 52-layer stack; this is a 4B dense model prefilling
about eight times faster. The app was designed to shed context if prefill
disappointed — preceding 15 lines → 8, personae limited to the selection, drop the
synopsis — and none of that was needed. At 0.5 s to first token the annotation
appears about as fast as a footnote you look down at.

Prompt length runs 526-1,023 tokens, mean 783 without a scene summary and roughly
880 with one. Early modern verse runs about **1.4 Qwen3 tokens per word** — elisions
(`o'er`, `on't`) and curly apostrophes split more than modern prose — so do not
budget this at 0.75 words per token. `--show-prompt` prints the assembled prompt
with its exact count from
`tokenizer.applyChatTemplate(messages:tools:additionalContext:)`.

### Turn 2 is cheap, but not as cheap as it looks like it should be

The commentary, the follow-up list, and every tapped question share one
`ChatSession` per passage, so the scene context is prefilled once. Turn 2 prefills
250-351 tokens against a 526-1,023-token turn 1:

| turn 1 prompt | commentary | turn 2 prompt |
|---|---|---|
| 526 | 100 words | 260 |
| 728 | 99 words | 250 |
| 1023 | 137 words | 300 |
| 832 | 190 words | 351 |

Turn 2's cost tracks the **commentary length**, not the turn-1 prompt length: the
600-900 token context block is reused, and what gets re-prefilled is the assistant's
own answer. The likely cause is that Qwen3 emits an empty `<think></think>` block
even under `enable_thinking: false`, which the framework strips from the recorded
text — so the re-rendered transcript diverges from the cached tokens at the assistant
turn and everything from there on is replayed. It costs about 0.2 s, so it was not
worth chasing further; a tapped question still answers in well under half a second.

A cache hit costs **1 ms** and no model work at all.

## What it does

- **Corpus.** Hamlet and Macbeth, parsed from Project Gutenberg into checked-in JSON
  by `tools/build_corpus.py`. Nothing at runtime depends on the script.
- **Selection is a range of line indices**, not characters. SwiftUI's `Text` does not
  expose a selected character range, and a play is line-structured anyway — line
  numbers are what the context window, the cache key, and the citation are all built
  on. Click, shift-click to extend, double-click for the whole speech, drag, arrows,
  Esc to clear, ⌘C to copy with the citation, ⌘R to regenerate.
- **Context is deterministic**, from the play's own structure: no embeddings, because
  the act/scene/speaker hierarchy is a better index here and it is exact.
- **One scene at a time.** This bounds rows to ~600 (Hamlet II.ii is the worst case),
  makes every selection intrinsically scene-scoped, and keeps the cache key trivial.
  The navigator is how you move.
- **Scene summaries** are generated in the background when a scene opens, in their
  own throwaway session. A selection cancels the prewarm rather than queueing behind
  it, and proceeds without a summary — the summary never blocks an annotation.
- **Four typefaces for the play**: the system face, Caslon, Baskerville, Garamond,
  picked from the `Aa` menu in the header and remembered between launches. It sets the
  **play text only** — the navigator, the commentary, the header and the status strip
  stay on the system face, and so does the line-number gutter: a serif family has no
  monospaced digits, and the gutter is a fixed 30pt frame that depends on stable
  digit advances to stay right-aligned.
- **Diagnostics are off by default.** The model capsule, the load check and the latency
  numbers under the commentary are for working on the app, not for reading a play; the
  header's `⋯` menu turns them on and the preference is remembered between launches.
  Two things show regardless, because hiding them makes a working app look broken: the
  first-launch download progress, and failures.
- **Your place is remembered between launches**: the scene you were reading comes back
  with the passage you had selected still highlighted and scrolled into view, along with
  the acts you had collapsed in the navigator and whichever panes you had hidden. A
  restored passage is not annotated on its own: click it or press ⌘R for that. A corpus
  rebuilt under a stored position keeps the scene and drops the highlight rather than
  putting it over different lines.

Both plays are in the reader; the model has clearly read both of them before, which
is worth remembering when judging output. `--model mlx-community/Qwen3-8B-4bit`
swaps in a larger model for comparison — test that on a *deliberately obscure*
passage, because "To be or not to be" is in every training corpus on earth and tells
you nothing. For scale, `--model mlx-community/Qwen3-0.6B-4bit` prefills at 8,426
tok/s and decodes at 394 tok/s in 0.78 GB, and writes noticeably vaguer annotations.

## Corpus

```bash
python3 tools/build_corpus.py --all --out Sources/ShakespeareReader/Resources/Plays
python3 tools/build_corpus.py --all --verify              # stats only, writes nothing
python3 tools/build_corpus.py --slug hamlet --from-file /tmp/pg1524.txt --dump-scene 3.1
```

The generated JSON is **checked in**, so the app builds and runs with no network for
anyone else. The script exists to make the parse reproducible and auditable, not as a
build step. Every pattern lives in one `PATTERNS` dict at the top of the file.

What the parse gets, measured on the real files and asserted by `--selftest`:

| | acts | scenes | speech lines | speech headings | directions |
|---|---|---|---|---|---|
| Hamlet | 5 | 20 | 3,817 | 1,137 | 243 |
| Macbeth | 5 | 28 | 2,329 | 649 | 168 |

0.00% of body lines are unclassified in either play, and every speech heading in the
body produces a speech: 1,137 of 1,137 and 649 of 649.

Details that decide the parser, all of them observed rather than assumed:

- **The files are CRLF.** Every `$`-anchored pattern breaks silently otherwise.
- **Scene headers are not reliably at column 0.** Hamlet I.i is; I.ii through I.v
  carry a leading space.
- **The Contents block cannot be skipped by column position.** Macbeth's Contents has
  `ACT I` at column 0, identical to its body header 85 lines later. The body scan is
  anchored after the `Dramatis Personæ` line instead.
- **Speakers are not gated on Dramatis Personæ.** The text has collective and
  numbered speakers that never appear there — `ALL.`, `BOTH MURDERERS.`, `DANES.`,
  `FIRST CLOWN.`, `APPARITION.` Gating would have silently dropped those speeches, so
  the pattern is accepted and `--verify` *reports* unresolved tokens instead of
  failing on them (21 in Hamlet, 22 in Macbeth, all of them genuinely absent from the
  personae list).
- **A heading is not always one all-caps word.** Both plays share a line between
  speakers (`HORATIO and MARCELLUS.`, `MACBETH, LENNOX.`), set three collectives in
  title case (`All.`, `Both.`, `Danes.`), and drop the period off one `BARNARDO`. The
  bare token is honoured only for a name already seen speaking, so the rule cannot
  invent a speaker. Missing these is worse than it looks — see the next point.
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
  in both plays, from the license text.
- **A bracketed aside can open a verse line.** `[_Aside._] A little more than kin,
  and less than kind.` Treating the whole line as a stage direction drops the verse
  entirely, which is what happened to 22 speeches in Hamlet and 4 in Macbeth before
  the leading-direction split existed — including that line and every one of
  Ophelia's `[_Sings._]` songs. Mid-line asides stay in the verse, because splitting
  those would fragment the line the reader selects.

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
naive version of it; 11 follow-up parser cases; one **golden `PassageContext`
render** compared against a checked-in string, which is what catches prompt drift; and
the typeface picker's inputs, including the CoreText italic probe that decides whether
a stage direction gets a real italic cut or a synthetic one. A deliberate prompt change
means regenerating that string alongside a `Prompts.version` bump — the self test
prints the replacement.

## Notes

- **`enable_thinking: false`** on every session, via
  `additionalContext: ["enable_thinking": false]`. Without it Qwen3 reasons at length
  before the first visible token — exactly the silent window `MuseGlimmerDemo`'s
  README documents. With it there is no visible `<think>` block.
- **Sampling** follows Qwen3's own recommendation for non-thinking mode
  (`temperature: 0.7, topP: 0.8, topK: 20`), except the scene summary, which runs at
  0.3 because it is meant to be dull and accurate.
- **`Memory.cacheLimit` is 256 MB**, not MuseGlimmer's 2 GB — that figure is sized for
  a 20 GB VLM churning hundred-MB image activations. 256 MB is what
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
  190 words); prompt iteration is where the remaining time would go.
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
- **This is a sibling SwiftPM package** with a local path dependency on the enclosing
  checkout. It sets `traits: []` on that dependency, which turns off the default
  `FoundationModelsIntegration` trait: the app never touches Apple's FoundationModels
  adapter, and `MLXHuggingFace` pulls that target in only when the trait is on.
- **`Bundle.module` in an executable target** resolves to a `.bundle` beside the
  binary in `.build/release/`. That is correct under `swift run`; a binary copied out
  on its own leaves its corpus behind, and `CorpusLoader` reports that rather than
  showing an empty library. mlx-swift's `default.metallib` is found the same way, from
  a sibling `mlx-swift_Cmlx.bundle`. Under Homebrew that directory is
  `$(brew --prefix)/opt/shakespeare-reader/libexec`, which is why the formula installs
  the binary and all four resource bundles there and puts a wrapper script in `bin`
  rather than a symlink: a symlink would make the app look for its corpus and its GPU
  kernels in `bin`.

## Corpus provenance

Hamlet is Project Gutenberg ebook 1524, Macbeth is 1533; both are public domain in
the United States. Each JSON file records the ebook id, the URL, the retrieval date,
and the SHA-256 of the source text as downloaded. See
`Sources/ShakespeareReader/Resources/Plays/NOTICE.md`.
