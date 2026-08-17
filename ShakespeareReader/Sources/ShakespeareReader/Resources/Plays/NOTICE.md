# Corpus provenance

The play texts in this directory are parsed from Project Gutenberg transcriptions
by `tools/build_corpus.py`. Each JSON file records its own `source` block with the
ebook ID, the URL, the retrieval date, and the SHA-256 of the source text as
downloaded, so any file here can be traced back to an exact input.

| play | ebook | source |
|---|---|---|
| Hamlet, Prince of Denmark | 1524 | https://www.gutenberg.org/ebooks/1524 |
| Macbeth | 1533 | https://www.gutenberg.org/ebooks/1533 |

Shakespeare's works are in the public domain. These particular transcriptions come
from Project Gutenberg and carry no additional copyright in the United States. The
Project Gutenberg header and license footer are stripped during the parse, so no
part of the PG license text is redistributed here — only the play.

Line numbers in these files are assigned sequentially within each scene over
speech lines only. They are **not** Folger, Arden, or any other standard edition's
numbers, which count a verse line shared between two speakers once. The `numbering`
field says `sequential-within-scene`, and the UI marks every citation
"(this edition)" so nothing implies otherwise.
