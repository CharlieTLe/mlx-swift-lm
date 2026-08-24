# Icon source

`chandos-portrait.jpg` is the Chandos portrait of William Shakespeare, attributed to
John Taylor, c. 1600–1610, oil on canvas. It is National Portrait Gallery NPG 1 — the
first portrait the gallery acquired, in 1856 — and the only likeness with a reasonable
claim to having been painted from life.

**Rights.** The painting is centuries out of copyright. A faithful photographic
reproduction of a two-dimensional public-domain work carries no separate copyright in
the United States (*Bridgeman Art Library v. Corel*, S.D.N.Y. 1999), which is why the
scan is checked in here rather than fetched at build time.

**File.** 960 × 1224, sRGB JPEG.

    sha256  723aa35260a5e04d7ec1e9ee9e2c937accb8db747bc4ca9aa68b6c40c5cfcaba

Recorded so the crop in `make_icon.swift` is auditable against a specific scan, the same
reason `Resources/Plays/NOTICE.md` records the SHA-256 of each play's source text.
Replacing the file without updating this hash makes the generated icons unreproducible;
replacing it with a scan of different dimensions also invalidates the crop rectangle,
which is in source pixels and is checked against the image bounds at generation time.
