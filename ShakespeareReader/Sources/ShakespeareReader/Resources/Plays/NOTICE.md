# Corpus provenance

The play texts in this directory are parsed from Project Gutenberg transcriptions
by `tools/build_corpus.py`. Each JSON file records its own `source` block with the
ebook ID, the URL, the retrieval date, and the SHA-256 of the source text as
downloaded, so any file here can be traced back to an exact input.

| play | ebook | source |
|---|---|---|
| Hamlet, Prince of Denmark | 1524 | https://www.gutenberg.org/ebooks/1524 |
| Macbeth | 1533 | https://www.gutenberg.org/ebooks/1533 |
| Romeo and Juliet | 1513 | https://www.gutenberg.org/ebooks/1513 |

Shakespeare's works are in the public domain. These particular transcriptions come
from Project Gutenberg and carry no additional copyright in the United States. The
Project Gutenberg header and license footer are stripped during the parse, so no
part of the PG license text is redistributed here — only the play.

One editorial decision is recorded here rather than left implicit in the parse.
Romeo and Juliet's two Chorus blocks are **scene 0** of the act they open, which
the reader labels `Prologue` and cites `I.Pro`. Neither block carries a `SCENE`
header in the transcription, and the first sits *above* `ACT I` rather than below
it; it is filed under Act I anyway, as printed editions do. Scene 0 exists in no
other play here, and the numbered scenes of those acts are unaffected: Act I runs
Prologue then Scenes I-V, Act II Prologue then Scenes I-VI.

Line numbers in these files are assigned sequentially within each scene over
speech lines only. They are **not** Folger, Arden, or any other standard edition's
numbers, which count a verse line shared between two speakers once. The `numbering`
field says `sequential-within-scene`, and the UI marks every citation
"(this edition)" so nothing implies otherwise.
