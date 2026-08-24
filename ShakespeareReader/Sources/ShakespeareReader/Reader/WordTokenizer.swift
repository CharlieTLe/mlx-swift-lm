import Foundation

/// Where one word of a line begins and ends.
///
/// Pure, `Sendable` and deliberately **not** `@MainActor`, so `--selftest` — which runs
/// synchronously and off the main actor — can assert against it. The other half of word
/// lookup, turning a pointer position into a character index, is `WordHitTest`: it needs
/// CoreText and a resolved platform font, and keeping it out of here is what leaves this
/// half testable.
///
/// `String.enumerateSubstrings(options: .byWords)` does most of the work and is right
/// about more than it looks: `o’er`, `on’t`, `Who’s` and `Hamlet’s` all come back whole,
/// straight apostrophe or curly. It is wrong about exactly two shapes, both of which the
/// verse is full of, and the merging below exists for them:
///
/// - **Hyphenated compounds.** `well-a-day` comes back as three words and `to-morrow` as
///   two.
/// - **A leading elision.** `’tis` comes back as `tis`, with the apostrophe outside any
///   word at all, so the mark would sit beside the letter it belongs to.
enum WordTokenizer {

    /// The characters that join two `.byWords` substrings into one word.
    ///
    /// Both apostrophe forms: the corpus is typeset with the curly one and a reader's
    /// own typing is not.
    private static let joiners: Set<Character> = ["'", "\u{2019}", "-"]

    /// Every word of `text`, in order, as ranges that never overlap.
    ///
    /// A joiner merges **only across a single character**. That is the difference
    /// between `to-morrow`, which is one word, and Gutenberg's `death,--and`, which is
    /// two: allowing a run of joiners would fuse every em-dash in the corpus into a
    /// portmanteau.
    static func words(in text: String) -> [Range<String.Index>] {
        var ranges: [Range<String.Index>] = []

        text.enumerateSubstrings(
            in: text.startIndex ..< text.endIndex,
            options: [.byWords, .substringNotRequired]
        ) { _, range, _, _ in
            guard let previous = ranges.last, previous.upperBound < range.lowerBound,
                text.index(after: previous.upperBound) == range.lowerBound,
                joiners.contains(text[previous.upperBound])
            else {
                ranges.append(range)
                return
            }
            ranges[ranges.count - 1] = previous.lowerBound ..< range.upperBound
        }

        // The leading elision, applied after merging so it extends the *merged* range
        // and not an inner fragment. Bounded by the previous word, so two ranges can
        // never be made to overlap.
        for position in ranges.indices {
            let range = ranges[position]
            guard range.lowerBound > text.startIndex else { continue }
            let before = text.index(before: range.lowerBound)
            guard text[before] != "-", joiners.contains(text[before]),
                position == 0 || ranges[position - 1].upperBound <= before
            else { continue }
            ranges[position] = before ..< range.upperBound
        }

        return ranges
    }

    /// The word containing `index`, or `nil` where there is none — a space, a comma, the
    /// blank right of a short line.
    ///
    /// Returning `nil` rather than the nearest word is what makes a hit-test near-miss
    /// visible: nothing is marked, so nothing is offered.
    static func word(at index: String.Index, in text: String) -> Range<String.Index>? {
        words(in: text).first { $0.contains(index) }
    }

    /// What to hand the dictionary and the model: the word without the punctuation that
    /// happened to be leaning on it.
    ///
    /// Edge characters only. `o’er` keeps its apostrophe and `to-morrow` its hyphen,
    /// because those are the word; `there?` and `death,` lose theirs, because those are
    /// the sentence.
    static func term(for range: Range<String.Index>, in text: String) -> String {
        var term = Substring(text[range])
        while let first = term.first, !first.isLetter, !first.isNumber {
            term = term.dropFirst()
        }
        while let last = term.last, !last.isLetter, !last.isNumber {
            term = term.dropLast()
        }
        return String(term)
    }
}
