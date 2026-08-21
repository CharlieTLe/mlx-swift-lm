# Corpus provenance

The play texts in this directory are parsed from Project Gutenberg transcriptions
by `tools/build_corpus.py`. Each JSON file records its own `source` block with the
ebook ID, the URL, the retrieval date, and the SHA-256 of the source text as
downloaded, so any file here can be traced back to an exact input.

All 35 come from Project Gutenberg's **1500-1542 series**, which is one transcription
lineage. That is what makes a single set of patterns viable: the parser was tuned
against three of these files and holds for the rest. The catalog's other Shakespeare
families are not interchangeable with it — 1100-1137 and 1765-1802 are separate
transcriptions, 2235-2270 is the First Folio, and 100 is the complete works in one
file — so adding a play from outside this range means re-tuning, not another row here.

| play | ebook | source |
|---|---|---|
| A Midsummer Night's Dream | 1514 | https://www.gutenberg.org/ebooks/1514 |
| All's Well That Ends Well | 1529 | https://www.gutenberg.org/ebooks/1529 |
| Antony and Cleopatra | 1534 | https://www.gutenberg.org/ebooks/1534 |
| As You Like It | 1523 | https://www.gutenberg.org/ebooks/1523 |
| Coriolanus | 1535 | https://www.gutenberg.org/ebooks/1535 |
| Cymbeline | 1538 | https://www.gutenberg.org/ebooks/1538 |
| Hamlet, Prince of Denmark | 1524 | https://www.gutenberg.org/ebooks/1524 |
| Julius Caesar | 1522 | https://www.gutenberg.org/ebooks/1522 |
| King Henry IV, Part 1 | 1516 | https://www.gutenberg.org/ebooks/1516 |
| King Henry IV, Part 2 | 1518 | https://www.gutenberg.org/ebooks/1518 |
| King Henry V | 1521 | https://www.gutenberg.org/ebooks/1521 |
| King Henry VI, Part 1 | 1500 | https://www.gutenberg.org/ebooks/1500 |
| King Henry VI, Part 2 | 1501 | https://www.gutenberg.org/ebooks/1501 |
| King Henry VI, Part 3 | 1502 | https://www.gutenberg.org/ebooks/1502 |
| King Henry VIII | 1541 | https://www.gutenberg.org/ebooks/1541 |
| King John | 1511 | https://www.gutenberg.org/ebooks/1511 |
| King Lear | 1532 | https://www.gutenberg.org/ebooks/1532 |
| King Richard II | 1512 | https://www.gutenberg.org/ebooks/1512 |
| King Richard III | 1503 | https://www.gutenberg.org/ebooks/1503 |
| Love's Labour's Lost | 1510 | https://www.gutenberg.org/ebooks/1510 |
| Macbeth | 1533 | https://www.gutenberg.org/ebooks/1533 |
| Measure for Measure | 1530 | https://www.gutenberg.org/ebooks/1530 |
| Much Ado about Nothing | 1519 | https://www.gutenberg.org/ebooks/1519 |
| Othello | 1531 | https://www.gutenberg.org/ebooks/1531 |
| Romeo and Juliet | 1513 | https://www.gutenberg.org/ebooks/1513 |
| The Comedy of Errors | 1504 | https://www.gutenberg.org/ebooks/1504 |
| The Merchant of Venice | 1515 | https://www.gutenberg.org/ebooks/1515 |
| The Merry Wives of Windsor | 1517 | https://www.gutenberg.org/ebooks/1517 |
| The Taming of the Shrew | 1508 | https://www.gutenberg.org/ebooks/1508 |
| The Tempest | 1540 | https://www.gutenberg.org/ebooks/1540 |
| The Two Gentlemen of Verona | 1509 | https://www.gutenberg.org/ebooks/1509 |
| The Winter's Tale | 1539 | https://www.gutenberg.org/ebooks/1539 |
| Timon of Athens | 1536 | https://www.gutenberg.org/ebooks/1536 |
| Titus Andronicus | 1507 | https://www.gutenberg.org/ebooks/1507 |
| Twelfth Night | 1526 | https://www.gutenberg.org/ebooks/1526 |

Three plays in the series are **absent**, and the gap is deliberate rather than an
oversight. Troilus and Cressida (1528) and Pericles (1537) open a chorus block whose
verse carries no speech heading at all, so the scene parses with no dialogue in it and
`--verify` rejects the play; Romeo and Juliet's choruses never posed the problem
because theirs are headed `CHORUS.` The Two Noble Kinsmen has no usable file here:
1542 is a First Folio transcription outside this series, and the in-series 1506 lists
`PROLOGUE` as its first personae entry, which the body scan mistakes for the body
header and so yields an empty cast.

The titles are Project Gutenberg's own, not a house style. `King Henry VI, Part 1`
next to `Coriolanus` is inconsistent as a shelf, and it is kept anyway: it is what the
source file's `Title:` line says, which keeps each JSON traceable to its input without
a mapping table that could drift.

Shakespeare's works are in the public domain. These particular transcriptions come
from Project Gutenberg and carry no additional copyright in the United States. The
Project Gutenberg header and license footer are stripped during the parse, so no
part of the PG license text is redistributed here — only the play.

One editorial decision is recorded here rather than left implicit in the parse.
Chorus and Prologue blocks are **scene 0** of the act they open, which the reader
labels `Prologue` and cites `I.Pro`. Such a block carries no `SCENE` header in the
transcription, and Romeo and Juliet's first one sits *above* `ACT I` rather than below
it; it is filed under Act I anyway, as printed editions do. Three plays have them, and
the numbered scenes of those acts are unaffected:

| play | scene-0 blocks |
|---|---|
| Romeo and Juliet | Acts I and II |
| King Henry V | Acts II, III, IV and V |
| King Henry VIII | Act I (the Prologue) |

Romeo and Juliet's Act I therefore runs Prologue then Scenes I-V, and Act II Prologue
then Scenes I-VI.

Line numbers in these files are assigned sequentially within each scene over
speech lines only. They are **not** Folger, Arden, or any other standard edition's
numbers, which count a verse line shared between two speakers once. The `numbering`
field says `sequential-within-scene`, and the UI marks every citation
"(this edition)" so nothing implies otherwise.

One known defect, recorded because the statistics cannot show it. `DIRECTION_OPENERS`
in the parser is the vocabulary of unbracketed stage directions, and it was tuned
against Hamlet, Macbeth and Romeo and Juliet only. An unbracketed direction whose
opening word is not in that list, arriving while a speaker is still open, is filed as
that speaker's verse: numbered, citable, and never counted as unclassified. A scan of
the 32 added plays puts this at roughly 34 lines, concentrated in the histories'
battle directions (`Dead March.`, `March.`, `Drum and colours.`, `Tucket.`, `Noise
within.`, `Fight. Excursions.`, `Trumpet sounds.`, `Music plays.`). Extending the
vocabulary is the fix; `--verify` prints the review list per play.
