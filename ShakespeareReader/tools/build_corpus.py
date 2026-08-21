#!/usr/bin/env python3
"""Fetch Project Gutenberg play texts and parse them into the reader's JSON.

Python standard library only, matching the precedent in the repo's own tools/,
so anyone with network access can run this with no install step.

    python3 tools/build_corpus.py --all --out Sources/ShakespeareReader/Resources/Plays
    python3 tools/build_corpus.py --from-file /tmp/pg1524.txt --slug hamlet --out DIR
    python3 tools/build_corpus.py --all --verify              # stats only, writes nothing
    python3 tools/build_corpus.py --dump-scene 3.1 --slug hamlet --from-file /tmp/pg1524.txt

The generated JSON is checked in, so the app builds and runs with no network.
This script exists to make the parse reproducible and auditable, not as a build
step.
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from datetime import date, timezone, datetime

PARSER_VERSION = 3
SCHEMA_VERSION = 1

# Every pattern lives here so re-tuning against a new transcription is one place.
#
# Grounded in the two real files rather than guessed at; the comments record what
# each pattern had to survive.
PATTERNS = {
    # Project Gutenberg wraps the work in these. Everything outside is boilerplate
    # and license text. Skipping this step produces `DAMAGE.` as a speaker in both
    # plays, from the license footer, which is a concrete demonstration that the
    # footer parses as dialogue if left in.
    "pg_start": re.compile(r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK"),
    "pg_end": re.compile(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK"),
    # `Dramatis Personæ` — with the ligature. Match on the prefix so an ASCII
    # transcription still lands.
    "personae": re.compile(r"^\s*Dramatis Person"),
    # Closes the personae block. Hamlet has `SCENE. Elsinore.`, Macbeth has
    # `SCENE: In the end of the Fourth Act, ...` wrapping onto a second line.
    "personae_end": re.compile(r"^\s*SCENE[.:]"),
    # Not reliably at column 0, and *not* distinguishable from the Contents block
    # by column: Macbeth's Contents has `ACT I` at column 0 identical to its body
    # header. The body scan is anchored after the personae block instead.
    "act": re.compile(r"^\s*ACT ([IVXLC]+)\s*$"),
    # Hamlet I.i is at column 0 but I.ii through I.v carry a single leading space,
    # so leading whitespace has to be tolerated. Upper-case `SCENE` also separates
    # the body from the Contents block, which uses title-case `Scene I.`.
    "scene": re.compile(r"^\s*SCENE ([IVXLC]+)\.\s*(.*)$"),
    # All-caps with a trailing period, on its own line. Deliberately *not* gated on
    # the personae list: the real text has collective and numbered speakers that
    # never appear in Dramatis Personæ (ALL., BOTH., BOTH MURDERERS., DANES.,
    # FIRST CLOWN., FIRST AMBASSADOR., APPARITION.). Rejecting unresolved tokens
    # would silently drop those speeches, so --verify reports them instead.
    "speaker": re.compile(r"^[A-Z][A-Z’' .]{1,30}\.$"),
    # Two or more speakers sharing one line, which the pattern above rejects
    # because of the lowercase `and`: `HORATIO and MARCELLUS.`, `MACBETH, LENNOX.`
    # Both joins are covered by the one pattern. Requiring every token to be
    # all-caps is what keeps it off the personae entry `DUNCAN, King of Scotland.`
    # and off directions like `Thunder and Lightning.` or `Enter Ross and Angus.`,
    # where the opening token is not.
    "joint_speaker": re.compile(r"^[A-Z][A-Z’']*(?:[ ]?(?:,|and)[ ][A-Z][A-Z’']*)+\.$"),
    # A heading that lost its period to a transcription slip: `BARNARDO` in Hamlet
    # I.i. Only honoured for a token already seen as a speaker in this play, so
    # the rule cannot invent one -- see `heading_for`.
    "bare_speaker": re.compile(r"^[A-Z][A-Z’']*$"),
    # A stage direction proper, e.g. `[_Exeunt._]`.
    "direction": re.compile(r"^\["),
    # A direction *opening* a verse line, e.g. `[_Aside._] A little more than kin,
    # and less than kind.` Splitting these out matters more than it looks: treating
    # the whole line as a direction drops the verse entirely, and 22 speeches in
    # Hamlet and 4 in Macbeth begin this way — including that line, `[_Within._]
    # Lord Hamlet.`, and every one of Ophelia's `[_Sings._]` songs.
    "leading_direction": re.compile(r"^(?P<direction>\[[^\]]*\])\s+(?P<text>\S.*)$"),
    # A direction closing a verse line, e.g. `How now? A rat? [_Draws._]`. Only
    # brackets at the very start or the very end are split off; the 13 mid-line
    # asides in these two plays (`We will bestow ourselves.—[_To Ophelia._] Read on
    # this book,`) stay in the verse, because splitting them would fragment the
    # line the reader selects.
    "trailing_direction": re.compile(r"^(?P<text>.*?\S)\s+(?P<direction>\[[^\]]*\])$"),
    # The uppercase run inside a personae entry is the token that matches a speech
    # heading: `The GHOST of the late king` speaks as `GHOST.`
    "personae_key": re.compile(r"\b([A-Z][A-Z’']*(?: [A-Z][A-Z’']*)*)\b"),
    # A chorus block's heading. Romeo and Juliet's `THE PROLOGUE` sits *above*
    # `ACT I` (pg1513.txt:121 against 147), so the body cannot be anchored on the
    # first act header alone or the sonnet is never seen. No trailing period is
    # allowed, which is what keeps this off the Contents entry `THE PROLOGUE.`
    # (line 41) and off Hamlet's `PROLOGUE.` speech heading in III.ii.
    "prologue": re.compile(r"^\s*(?:THE )?PROLOGUE\s*$"),
    # A heading the transcription left on the same line as its first verse line:
    # `ROMEO. Nurse, commend me to thy lady and mistress. I protest unto`
    # (pg1513.txt:2288) and `THIRD WATCH. Here is a Friar that trembles, sighs,
    # and weeps.` (5107). Without this both are attributed to whoever spoke last
    # with the heading left inside their own text, and THIRD WATCH -- who has no
    # other heading in the play -- never enters the cast at all.
    #
    # Across all three files this also matches the `SCENE I. A public place.`
    # headers, which is harmless: `scene` has already consumed those by the point
    # in the loop where this is tested.
    "inline_speaker": re.compile(
        r"^(?P<speaker>[A-Z][A-Z’' ]{1,30})\.[ ](?P<text>\S.*)$"),
}

# Collective headings the transcription sets in title case: `All.`, `Both.`,
# `Danes.` An explicit allowlist rather than a pattern, because a title-case word
# plus a period is shape-identical to the direction `Enter Ghost.` Normalising to
# upper case merges these with the `ALL.`/`BOTH.`/`DANES.` speeches elsewhere in
# the same plays rather than inventing a second speaker for the same voice.
TITLE_CASE_SPEAKERS = {"All.": "ALL", "Both.": "BOTH", "Danes.": "DANES"}

# How an unbracketed stage direction opens. Gutenberg interrupts a speech with one
# of these and then resumes the *same* speech with no repeated heading, so an
# unbracketed line cannot be read as a direction merely because no heading came
# immediately before it -- that is what filed 64 paragraphs of Hamlet and 37 of
# Macbeth, Claudius's prayer and the dagger speech among them, as directions.
#
# Grounded in the two real files rather than guessed at, like the patterns above.
# The optional leading underscore is the dumb-show direction `_The Ghost of ...`.
# `Exit`/`Exeunt` are deliberately absent: they never appear unbracketed in either
# play, and every bracketed one is already caught by `direction`.
#
# The opener has to be followed by a space, a period or a comma, which is what
# every unbracketed direction in both files does -- including Macbeth's `Enter,
# with drum and colours ...`. `\b` will not do it: a curly apostrophe is a word
# boundary, so `Alarum\b` matches `Alarum’d by his sentinel, the wolf,` and takes
# the last eight lines of "Is this a dagger" with it. The plural is spelled out for
# the same reason in reverse: `Alarums. Enter Macduff.` is a direction, and without
# it that line parses as Macbeth's verse.
#
# The last two entries are Romeo and Juliet's, and are the demonstration that this
# vocabulary is per-transcription: ` Juliet appears above at a window.` (1530) and
# ` Musicians waiting. Enter Servants.` (1144) are unbracketed directions no other
# entry reaches. `Juliet` alone will not do -- `Juliet, the County stays.` (952) is
# Lady Capulet's verse -- so both openers carry their second word.
DIRECTION_OPENERS = re.compile(
    r"^_?(Enter|Re-enter|Alarums?|Flourish|Thunder|Hautboys|Sennet|Danish march"
    r"|Retreat|Trumpets|A banquet|Ghost rises|The Ghost of|The King rises"
    r"|Juliet appears|Musicians waiting)(?=[ .,])"
)

# What ends a direction. A wrapped direction breaks mid-clause, so an open
# direction whose last line ends in none of these is still being written -- and
# one that ends in `_]` is finished, which is what stops `[_Aside._]` on its own
# line from swallowing the verse underneath it.
DIRECTION_TERMINATORS = (".", "!", "?", "_", "]", ")")

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}

SOURCES = {
    "hamlet": {
        "ebook_id": 1524,
        "title": "Hamlet, Prince of Denmark",
        "url": "https://www.gutenberg.org/cache/epub/1524/pg1524.txt",
    },
    "macbeth": {
        "ebook_id": 1533,
        "title": "Macbeth",
        "url": "https://www.gutenberg.org/cache/epub/1533/pg1533.txt",
    },
    "romeo-and-juliet": {
        "ebook_id": 1513,
        "title": "Romeo and Juliet",
        "url": "https://www.gutenberg.org/cache/epub/1513/pg1513.txt",
    },
}

# Speech tokens that belong to a personae entry filed under another name, because
# Dramatis Personæ names the character and the dialogue labels the role: Hamlet's
# Claudius and Gertrude speak throughout as `KING.` and `QUEEN.`
#
# Deliberately short. An alias is only worth its tokens when the target entry has
# a real blurb to contribute, so the collective and numbered speakers (ALL., BOTH
# MURDERERS., FIRST WITCH., DANES.) are left unresolved rather than attached to a
# nearby entry that would describe them wrongly. --verify lists what is left.
ALIASES = {
    "hamlet": {
        "CLAUDIUS": ["KING"],
        "GERTRUDE": ["QUEEN"],
        "Two Clowns": ["FIRST CLOWN", "SECOND CLOWN"],
    },
    "macbeth": {},
    # The Prince speaks throughout as `PRINCE.`; Dramatis Personæ files him as
    # `ESCALUS, Prince of Verona.` -- the Claudius case exactly. It is also the
    # only alias in this play that earns its tokens: every other unresolved token
    # (the numbered Servants, Watchmen and Musicians, `CITIZENS`) belongs to a
    # personae entry with an empty blurb, so attaching it would say nothing.
    "romeo-and-juliet": {"ESCALUS": ["PRINCE"]},
}


def roman_to_int(text):
    total = 0
    for index, char in enumerate(text):
        value = ROMAN[char]
        nxt = ROMAN[text[index + 1]] if index + 1 < len(text) else 0
        total += -value if value < nxt else value
    return total


def split_lines(text):
    """Split on any line terminator.

    The Gutenberg files are CRLF throughout. This is the one detail that silently
    breaks every `$`-anchored pattern, because a `\\r` left on the end of a line
    means `^\\s*ACT ([IVXLC]+)\\s*$` never matches -- `\\s` does match `\\r`, but a
    speaker heading like `HAMLET.\\r` fails `\\.$` outright. Splitting explicitly
    here rather than relying on the reader's newline mode keeps that out of every
    pattern below.
    """
    return re.split(r"\r\n|\r|\n", text)


# ---------------------------------------------------------------------------
# personae


def title_case(token):
    return " ".join(word.capitalize() for word in token.split())


def parse_personae(lines, start, aliases):
    """Entries between the `Dramatis Personæ` heading and the SCENE summary.

    Blank lines occur *inside* the block (Macbeth groups the women, the crowds,
    and the apparitions with blank lines between), so the block ends at the SCENE
    summary rather than at the first blank line.
    """
    personae = []
    for line in lines[start + 1 :]:
        if PATTERNS["personae_end"].match(line):
            break
        entry = line.strip()
        if not entry:
            continue
        # A wrapped entry continues the previous one: Macbeth's `Lords, Gentlemen,
        # Officers, Soldiers, Murderers, Attendants and` / `Messengers.` Keyed on
        # the dangling conjunction rather than on a missing period, because most
        # entries in Hamlet have no trailing period at all. The semicolon is Romeo
        # and Juliet's `Citizens of Verona; several Men and Women, relations to
        # both houses;` (pg1513.txt:112) -- the one `;`-terminated line in any of
        # the three personae blocks, and without it `Maskers` is read as a 28th
        # person.
        if personae and personae[-1]["_raw"].rstrip().endswith(("and", ",", ";")):
            personae[-1]["_raw"] += " " + entry
            personae[-1]["blurb"] = blurb_for(
                personae[-1]["_raw"], personae[-1]["name"])
            continue

        text = entry.rstrip(".")
        match = PATTERNS["personae_key"].search(text.split(",")[0])
        # Require two characters so a leading `A ` or `I ` is not read as a name.
        key = match.group(1) if match and len(match.group(1)) > 1 else None
        name = key or text.split(",")[0].strip()

        personae.append(
            {
                "name": name,
                "display": title_case(name) if key else name,
                "blurb": blurb_for(entry, name),
                "aliases": aliases.get(name, []),
                "_raw": entry,
            }
        )
    for person in personae:
        del person["_raw"]
    return personae


def blurb_for(entry, name):
    """Everything the entry says about the person beyond their name.

    Cut at the *end* of the name wherever it appears rather than only stripping a
    leading match, so `The GHOST of the late king, Hamlet’s father` yields `of the
    late king, Hamlet’s father` instead of repeating the whole entry back.
    """
    text = entry.rstrip(".")
    at = text.find(name)
    if at >= 0:
        text = text[at + len(name) :]
    return text.lstrip(", ").strip()


# ---------------------------------------------------------------------------
# body


class ParseStats:
    def __init__(self):
        self.speech_lines = 0
        self.direction_lines = 0
        self.unclassified = []
        self.multiline_directions = 0
        self.unbracketed_directions = 0
        self.opener_tokens = set()
        self.resumed_speeches = 0


def parse_body(lines, body_start, stats):
    """Walk the body once, emitting acts, scenes, speech lines and directions.

    Line numbers are assigned sequentially within each scene over *speech* lines
    only. These are not Folger or Arden numbers -- those count a verse line shared
    between two speakers once -- which is why the JSON records the numbering scheme
    and the UI marks every citation "(this edition)".
    """
    acts = []
    act = None
    scene = None
    speaker = None
    number = 0
    # `speechStart` rides on the first speech line of a run rather than getting a
    # record of its own, so the reader view knows where to print a heading.
    speech_starts = False
    # Every token that has introduced a speech in this play. What lets a heading
    # that lost its period be recognised without letting the parser invent one.
    seen = set()
    # Whether the previous line was blank. A resumed speech is verse at the *start*
    # of a paragraph with no heading above it, which is worth counting separately
    # from verse that merely follows a direction on the next line down.
    at_paragraph_start = True
    # Consecutive direction lines with no blank between them are one wrapped
    # direction (`Alarum within. Enter King Duncan, Malcolm, Donalbain, Lennox,
    # with` / `Attendants, meeting a bleeding Captain.`). Joining them keeps the
    # on-stage scan and the prompt reading as one sentence.
    pending = []

    def heading_for(text):
        """The speaker a line introduces, or None if it introduces nobody.

        Three shapes beyond the plain `HAMLET.`, and all three matter: leave them
        out and -- now that a speaker survives a blank line -- their speeches are
        attributed to whoever spoke last, which is worse than filing them as
        directions was.
        """
        if PATTERNS["speaker"].match(text) or PATTERNS["joint_speaker"].match(text):
            return text[:-1]
        if text in TITLE_CASE_SPEAKERS:
            return TITLE_CASE_SPEAKERS[text]
        if PATTERNS["bare_speaker"].match(text) and text in seen:
            return text
        return None

    def flush():
        nonlocal pending
        if not pending:
            return
        if len(pending) > 1:
            stats.multiline_directions += 1
        emit_direction(" ".join(pending))
        pending = []

    def emit_direction(text):
        scene["lines"].append({"kind": "direction", "text": text})
        stats.direction_lines += 1

    def emit_speech(text, is_start):
        nonlocal number
        number += 1
        scene["lines"].append(
            {
                "kind": "speech",
                "speaker": speaker,
                "number": number,
                "text": text,
                "speechStart": is_start,
            }
        )
        stats.speech_lines += 1

    for raw in lines[body_start:]:
        stripped = raw.strip()

        if not stripped:
            flush()
            # A blank line closes the *paragraph*, not the speech, which is why the
            # speaker is deliberately left open here. Gutenberg interrupts a speech
            # with a blank-delimited direction and then resumes it with no repeated
            # heading; clearing the speaker at a blank filed everything after such
            # an interruption as a direction -- the rest of Horatio's "But, soft,
            # behold!" and some 250 other verse lines in Hamlet alone.
            at_paragraph_start = True
            continue

        paragraph_start = at_paragraph_start
        at_paragraph_start = False

        # `THE PROLOGUE` opens the body above `ACT I` (pg1513.txt:121 against 147),
        # so the act it belongs to has no header yet. Printed editions file it
        # under Act I, and so does this.
        if act is None and PATTERNS["prologue"].match(raw):
            flush()
            act = {"number": 1, "scenes": []}
            acts.append(act)
            scene = None
            speaker = None
            continue

        act_match = PATTERNS["act"].match(raw)
        if act_match:
            flush()
            # A fresh local name: `number` is the live line counter. Re-appending
            # on a header for the act already open is what `ACT I`, 26 lines under
            # `THE PROLOGUE`, would otherwise do -- giving the play six acts, the
            # first of them holding nothing but the Prologue.
            act_number = roman_to_int(act_match.group(1))
            if act is None or act["number"] != act_number:
                act = {"number": act_number, "scenes": []}
                acts.append(act)
            scene = None
            speaker = None
            continue

        scene_match = PATTERNS["scene"].match(raw)
        if scene_match and act is not None:
            flush()
            setting = scene_match.group(2).strip()
            scene = {
                "number": roman_to_int(scene_match.group(1)),
                "setting": setting,
                "lines": [],
            }
            act["scenes"].append(scene)
            speaker = None
            number = 0
            continue

        if scene is None:
            if act is None:
                # Anything before the first act: the `SCENE. Elsinore.` summary and
                # blank-padded front matter.
                stats.unclassified.append(stripped)
                continue
            # Inside an act with no `SCENE` header yet: a chorus block. Romeo and
            # Juliet's `ACT II` (pg1513.txt:1425) is followed by `Enter Chorus.`
            # and a 14-line sonnet, with no header until `SCENE I.` 23 lines down,
            # so every one of those lines used to fall to `unclassified` -- one
            # dropped sonnet, and the whole of that play's 0.53%.
            #
            # Scene 0, which `SceneLabel` renders `Prologue` and cites `Pro`.
            # Deliberately falls through rather than continuing, so `Enter Chorus.`
            # is classified as the direction it is.
            scene = {"number": 0, "setting": "", "lines": []}
            act["scenes"].append(scene)
            speaker = None
            number = 0

        heading = heading_for(stripped)
        if heading is not None:
            flush()
            speaker = heading
            seen.add(speaker)
            speech_starts = True
            continue

        inline = (
            PATTERNS["inline_speaker"].match(stripped) if paragraph_start else None
        )
        if inline is not None:
            # A heading the transcription left on the same line as its verse:
            # `ROMEO. Nurse, commend me…` (2288) and `THIRD WATCH. Here is a
            # Friar…` (5107). Opening the speech here and falling through with the
            # rest of the line is what puts THIRD WATCH in the cast at all.
            flush()
            speaker = inline.group("speaker")
            seen.add(speaker)
            speech_starts = True
            stripped = inline.group("text")
            # Not *also* a resumed speech: it carries its own heading.
            paragraph_start = False

        if PATTERNS["direction"].match(stripped):
            # A new bracket always starts a new direction; a line that does not
            # start with one continues the open direction instead.
            flush()
            leading = (
                PATTERNS["leading_direction"].match(stripped)
                if speaker is not None
                else None
            )
            if leading is None:
                pending.append(stripped)
                continue
            # A direction opening a line of an open speech: emit it in reading
            # order, then fall through with the verse that follows it.
            emit_direction(leading.group("direction"))
            stripped = leading.group("text")
        elif pending and not pending[-1].endswith(DIRECTION_TERMINATORS):
            # The open direction broke mid-clause, so this line continues it. All
            # 13 wrapped directions across both plays are this shape.
            pending.append(stripped)
            continue
        elif (opener := DIRECTION_OPENERS.match(stripped)) and not speech_starts:
            # An unbracketed stage direction: Hamlet's column-0 `Enter Francisco
            # and Barnardo, two sentinels.` and Macbeth's indented ` Thunder and
            # Lightning. Enter three Witches.`
            #
            # The `speech_starts` guard is what keeps the two verse lines in these
            # plays that open with the direction vocabulary -- `OPHELIA. / The King
            # rises.` and `SIWARD. / Enter, sir, the castle.` -- as verse. A
            # direction is always its own paragraph, never the first line under a
            # heading. --verify lists any speech line that matches, so extending
            # the vocabulary stays a deliberate act.
            flush()
            stats.unbracketed_directions += 1
            stats.opener_tokens.add(opener.group(1))
            pending.append(stripped)
            continue
        elif speaker is None:
            # Before the first heading of the scene: the opening direction and any
            # unbracketed material under it.
            pending.append(stripped)
            continue
        else:
            # Verse under the open speaker, resuming either after a direction with
            # no blank line under it or after a blank-delimited interruption.
            if paragraph_start:
                stats.resumed_speeches += 1
            flush()

        text = stripped
        trailing = PATTERNS["trailing_direction"].match(text)
        direction_after = None
        if trailing:
            text = trailing.group("text")
            direction_after = trailing.group("direction")

        is_start = speech_starts
        speech_starts = False
        emit_speech(text, is_start)

        if direction_after:
            # Split off so it neither pollutes the verse nor breaks selection.
            emit_direction(direction_after)

    flush()
    return acts


def find_body_start(lines):
    """First act or chorus header after the personae block.

    Column position will not do this: Macbeth's Contents block has `ACT I` at
    column 0, identical to its body header 85 lines later. The personae block sits
    between the two in all three files, so it is the reliable anchor.

    The chorus header counts because Romeo and Juliet's `THE PROLOGUE` is above
    `ACT I`, not below it. Anchoring on the act header alone starts the scan 26
    lines too late and drops "Two households, both alike in dignity" without even
    counting it as unclassified.
    """
    personae_at = next(
        (i for i, line in enumerate(lines) if PATTERNS["personae"].match(line)), None
    )
    if personae_at is None:
        raise SystemExit("no `Dramatis Person` line found; cannot locate the body")
    body_at = next(
        (
            i
            for i in range(personae_at, len(lines))
            if PATTERNS["act"].match(lines[i]) or PATTERNS["prologue"].match(lines[i])
        ),
        None,
    )
    if body_at is None:
        raise SystemExit("no act header after the personae block")
    return personae_at, body_at


def parse_play(slug, text, retrieved):
    source = SOURCES[slug]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    lines = split_lines(text)

    start = next(
        (i for i, line in enumerate(lines) if PATTERNS["pg_start"].match(line)), None
    )
    end = next(
        (i for i, line in enumerate(lines) if PATTERNS["pg_end"].match(line)), None
    )
    if start is None or end is None:
        raise SystemExit(f"{slug}: could not find the Project Gutenberg markers")
    body_lines = lines[start + 1 : end]

    personae_at, body_at = find_body_start(body_lines)
    personae = parse_personae(body_lines, personae_at, ALIASES.get(slug, {}))

    stats = ParseStats()
    acts = parse_body(body_lines, body_at, stats)

    play = {
        "schemaVersion": SCHEMA_VERSION,
        "id": slug,
        "title": source["title"],
        "author": "William Shakespeare",
        "numbering": "sequential-within-scene",
        "source": {
            "kind": "gutenberg",
            "ebookID": source["ebook_id"],
            "url": f"https://www.gutenberg.org/ebooks/{source['ebook_id']}",
            "retrieved": retrieved,
            "textSHA256": digest,
            "parserVersion": PARSER_VERSION,
            "note": "Public domain in the US. PG header/footer stripped.",
        },
        "personae": personae,
        "acts": acts,
    }
    return play, stats


# ---------------------------------------------------------------------------
# reporting


def verify(slug, play, stats):
    """Print per-play statistics and return True if the parse is usable."""
    scenes = [(act, scene) for act in play["acts"] for scene in act["scenes"]]
    speakers = {
        line["speaker"]
        for _, scene in scenes
        for line in scene["lines"]
        if line["kind"] == "speech"
    }
    known = {person["name"] for person in play["personae"]}
    known |= {
        alias for person in play["personae"] for alias in person["aliases"]
    }
    unresolved = sorted(speakers - known)

    classified = stats.speech_lines + stats.direction_lines
    total = classified + len(stats.unclassified)
    unclassified_pct = 100.0 * len(stats.unclassified) / total if total else 0.0

    longest_run = 0
    for _, scene in scenes:
        run = 0
        current = None
        for line in scene["lines"]:
            if line["kind"] != "speech":
                continue
            if line["speaker"] == current:
                run += 1
            else:
                current, run = line["speaker"], 1
            longest_run = max(longest_run, run)

    print(f"{slug}:")
    print(f"  acts             {len(play['acts'])}")
    print(f"  scenes           {len(scenes)}")
    # Acts that opened on a chorus block. Printed because a dropped Prologue is
    # otherwise invisible here: before scene 0 existed those lines were not even
    # counted as unclassified, so nothing in this report moved when they vanished.
    chorus = [
        f"{act['number']}.{scene['number']}"
        for act, scene in scenes
        if scene["number"] == 0
    ]
    print(f"  chorus scenes    {', '.join(chorus) or '-'}")
    print(f"  speech lines     {stats.speech_lines}")
    print(f"  direction lines  {stats.direction_lines}")
    print(f"  wrapped dirs     {stats.multiline_directions}")
    print(f"  unbracketed dirs {stats.unbracketed_directions}")
    print(f"  opener tokens    {', '.join(sorted(stats.opener_tokens)) or '-'}")
    print(f"  resumed speeches {stats.resumed_speeches}")
    # Speech lines that read like a direction. The direction vocabulary decides
    # which unbracketed lines are directions, so this is the review list for that
    # decision: expected output is exactly `The King rises.` in Hamlet and `Enter,
    # sir, the castle.` in Macbeth. Anything else here means a real direction is
    # being read as verse somewhere, and the vocabulary needs extending.
    direction_shaped = [
        line["text"]
        for _, scene in scenes
        for line in scene["lines"]
        if line["kind"] == "speech" and DIRECTION_OPENERS.match(line["text"])
    ]
    print(f"  dir-shaped verse {len(direction_shaped)} (review):")
    for text in direction_shaped[:10]:
        print(f"      {text[:72]!r}")
    print(f"  personae         {len(play['personae'])}")
    print(f"  speakers         {len(speakers)}")
    print(f"  unresolved       {len(unresolved)}: {', '.join(unresolved) or '-'}")
    print(f"  unclassified     {len(stats.unclassified)} ({unclassified_pct:.2f}%)")
    for line in stats.unclassified[:10]:
        print(f"      {line[:72]!r}")
    print(f"  longest run      {longest_run} lines with no speaker change")

    ok = True
    empty = [
        f"{act['number']}.{scene['number']}"
        for act in play["acts"]
        for scene in act["scenes"]
        if not any(line["kind"] == "speech" for line in scene["lines"])
    ]
    if empty:
        print(f"  FAIL scenes with no speech lines: {', '.join(empty)}")
        ok = False
    if unclassified_pct > 2.0:
        print(f"  FAIL {unclassified_pct:.2f}% of lines unclassified (limit 2%)")
        ok = False
    return ok


def dump_scene(play, spec):
    act_number, scene_number = (int(part) for part in spec.split("."))
    for act in play["acts"]:
        if act["number"] != act_number:
            continue
        for scene in act["scenes"]:
            if scene["number"] != scene_number:
                continue
            print(f"{play['title']} {act_number}.{scene_number} — {scene['setting']}")
            for line in scene["lines"]:
                if line["kind"] == "direction":
                    print(f"        | {line['text']}")
                else:
                    head = f"{line['speaker']}." if line["speechStart"] else ""
                    # Wide enough for the longest heading in either play,
                    # `ROSENCRANTZ and GUILDENSTERN.`, which ran into the verse at 16.
                    print(f"{line['number']:5d} | {head:30s}{line['text']}")
            return
    raise SystemExit(f"no scene {spec}")


# ---------------------------------------------------------------------------


def read_source(slug, from_file):
    """Read the source text from disk or fetch it.

    `--from-file` stays a first-class path because gutenberg.org needs a network
    allowlist entry here, and a future reader may not have one.
    """
    if from_file:
        with open(from_file, "rb") as handle:
            return handle.read().decode("utf-8")
    url = SOURCES[slug]["url"]
    print(f"fetching {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="every play in SOURCES")
    parser.add_argument("--slug", choices=sorted(SOURCES), help="one play")
    parser.add_argument("--from-file", help="read the source text from disk")
    parser.add_argument("--out", help="directory to write JSON into")
    parser.add_argument("--verify", action="store_true", help="print stats")
    parser.add_argument("--dump-scene", help="print one scene as parsed, e.g. 3.1")
    parser.add_argument(
        "--retrieved",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="retrieval date recorded in the JSON (default: today, UTC)",
    )
    args = parser.parse_args()

    slugs = sorted(SOURCES) if args.all else ([args.slug] if args.slug else [])
    if not slugs:
        parser.error("pass --all or --slug")
    if args.from_file and len(slugs) > 1:
        parser.error("--from-file applies to a single --slug")

    ok = True
    for slug in slugs:
        text = read_source(slug, args.from_file)
        play, stats = parse_play(slug, text, args.retrieved)

        if args.dump_scene:
            dump_scene(play, args.dump_scene)
            continue
        if args.verify:
            ok = verify(slug, play, stats) and ok
        if args.out and not args.verify:
            path = f"{args.out.rstrip('/')}/{slug}.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(play, handle, ensure_ascii=False, indent=1)
                handle.write("\n")
            print(f"wrote {path}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
