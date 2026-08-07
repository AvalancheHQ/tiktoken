use std::collections::HashSet;
use std::num::NonZeroU64;
use std::thread;

use fancy_regex::Regex;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use regex::Regex as FastRegex;
use rustc_hash::FxHashMap as HashMap;

#[cfg(feature = "python")]
mod py;

pub type Rank = u32;

use std::collections::BinaryHeap;

#[derive(Eq, PartialEq, Clone, Copy)]
struct Merge {
    start: usize,
    rank: Rank,
}

impl Ord for Merge {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.start.cmp(&self.start))
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct State {
    prev: usize,
    end: usize,
    next_end: usize,
    next_rank: Rank,
    cur_rank: Rank,
}

fn _byte_pair_merge_large(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<Rank> {
    let mut state = Vec::with_capacity(piece.len());
    state.push(State {
        prev: usize::MAX,
        end: 1,
        next_end: 2,
        next_rank: Rank::MAX,
        cur_rank: Rank::MAX,
    });

    let mut heap = BinaryHeap::with_capacity(piece.len());
    for i in 0..piece.len() - 1 {
        if let Some(&rank) = ranks.get(&piece[i..i + 2]) {
            heap.push(Merge { start: i, rank });
            state[i].next_rank = rank;
        }
        // note this is happening offset by 1
        state.push(State {
            prev: i,
            end: i + 2,
            next_end: i + 3,
            next_rank: Rank::MAX,
            cur_rank: Rank::MAX,
        });
    }

    // Repeatedly find the valid merge with smallest rank. We merge the (left) token that
    // starts at `start` and ends at `state[start].end` with the (right) token that starts at
    // `state[start].end` and ends at `state[start].next_end`.  We invalidate the old merges
    // (the ones that started at `state[start].end` and ended at `state[start]`) and add the two
    // new potential merges to the heap.

    let potential_merge = {
        #[inline(always)]
        |state: &mut Vec<State>,
         heap: &mut BinaryHeap<Merge>,
         start: usize,
         next_end_item: usize| {
            state[start].next_end = next_end_item;
            state[start].next_rank = Rank::MAX; // Always invalidate the old merge
            if next_end_item <= piece.len()
                && let Some(&rank) = ranks.get(&piece[start..next_end_item])
            {
                // We have a valid potential merge!
                heap.push(Merge { start, rank });
                state[start].next_rank = rank;
            }
        }
    };

    while let Some(left) = heap.pop() {
        if left.rank == Rank::MAX {
            break;
        }
        if left.rank != state[left.start].next_rank {
            continue; // This merge was invalidated, ignore it
        }

        let left_start = left.start;
        let right_start = state[left_start].end;
        let right_end = state[left_start].next_end;
        debug_assert!(right_end == state[right_start].end);
        let right_next_end = state[right_start].next_end;

        // Merge left and right into a single token
        state[left_start].cur_rank = state[left_start].next_rank;
        state[left_start].end = right_end;
        potential_merge(&mut state, &mut heap, left_start, right_next_end);
        if right_end < state.len() {
            state[right_end].prev = left_start;
        }
        // Update the merge that ends at left_start
        if left_start > 0 {
            let prev_start = state[left_start].prev;
            potential_merge(&mut state, &mut heap, prev_start, right_end);
        }
        // Invalidate the merge starting at right_start, so we ignore it when it comes off the heap
        state[right_start].next_rank = Rank::MAX;
    }

    let mut result = Vec::new();
    let mut i = 0;
    while i < state.len() {
        if state[i].cur_rank != Rank::MAX {
            result.push(state[i].cur_rank);
        } else {
            result.push(ranks[&piece[i..state[i].end]]);
        }
        i = state[i].end;
    }
    result
}

fn _byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    // This is a vector of (start, rank).
    // The rank is of the pair starting at position start.
    let mut parts = Vec::with_capacity(piece.len() + 1);

    // Note that we hash bytes when indexing into `ranks`, not token pairs. As long as we train BPE
    // the way we currently do, this is equivalent. An easy way to break this would be to decouple
    // merge priority from token index or to prevent specific token merges.
    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = {
        #[inline(always)]
        |parts: &Vec<(usize, Rank)>, i: usize| {
            if (i + 3) < parts.len() {
                // Similar to `piece[i..i + 2]` above. The +3 is because we haven't yet deleted
                // parts[i + 1], see comment in the main loop.
                *ranks
                    .get(&piece[parts[i].0..parts[i + 3].0])
                    .unwrap_or(&Rank::MAX)
            } else {
                Rank::MAX
            }
        }
    };

    // If you have n parts and m merges, this does O(mn) work.
    // We could do something with a heap and do O(m log n) work.
    // n is often very small so considerations like cache-locality outweigh the algorithmic
    // complexity downsides of the `parts` vector.
    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;
        // Update parts[i] and parts[i - 1] before removing parts[i + 1], since
        // `parts.remove(i + 1)` will thrash the cache.
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }
    parts
}

pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    let piece_len = piece.len();

    if piece_len == 1 {
        return vec![ranks[piece]];
    }
    if piece_len < 100 {
        return _byte_pair_merge(ranks, piece)
            .windows(2)
            .map(|part| ranks[&piece[part[0].0..part[1].0]])
            .collect();
    }
    _byte_pair_merge_large(ranks, piece)
}

pub fn byte_pair_split<'a>(piece: &'a [u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<&'a [u8]> {
    assert!(piece.len() > 1);
    _byte_pair_merge(ranks, piece)
        .windows(2)
        .map(|part| &piece[part[0].0..part[1].0])
        .collect()
}

// ---------------------------------------------------------------------------
// Tokeniser split
// ---------------------------------------------------------------------------

/// The leading run of top-level alternatives of a split pattern that the `regex`
/// crate can run on its own, compiled as a single automaton.
///
/// `fancy_regex` has to interpret the tokeniser patterns on its backtracking VM,
/// because they use possessive quantifiers and the `\s+(?!\S)` lookahead. It pays
/// the VM's fixed cost — a fresh match state, save/restore bookkeeping and a
/// delegated automaton search per alternative it tries — for *every piece*, and a
/// piece is a handful of bytes.
///
/// The alternatives that produce nearly every piece of real text need none of
/// that: they are plain regex syntax once the possessive markers are dropped.
/// Running them through `regex::Regex::find_iter` walks the whole document in a
/// single streaming search that yields piece after piece, so a regex engine is
/// entered once per *document* instead of once per piece. `fancy_regex` stays the
/// source of truth and decides every position the fast automaton does not match
/// (whitespace runs, the `\s++$` and `\s+(?!\S)` alternatives).
#[derive(Clone)]
struct FastSplit {
    regex: FastRegex,
}

impl FastSplit {
    /// Derives the fast splitter for `pattern`, or `None` if it cannot be shown
    /// to split exactly like `fancy`.
    fn new(pattern: &str, fancy: &Regex) -> Option<FastSplit> {
        // `\G`/`\K` depend on where the search started, and `fancy_regex`'s own
        // iterator feeds extra option flags into the VM for them. Leave any such
        // pattern alone entirely.
        if pattern.contains("\\G") || pattern.contains("\\K") {
            return None;
        }

        // Take the *leading* run of alternatives the `regex` crate can run: since
        // alternation is ordered, if one of them matches at a position then the
        // full pattern matches there with the very same alternative.
        let mut fast: Vec<String> = Vec::new();
        for alt in top_level_alternatives(pattern)? {
            match regex_crate_alternative(alt) {
                Some(relaxed) => fast.push(relaxed),
                None => break,
            }
        }
        if fast.is_empty() {
            return None;
        }

        let regex = FastRegex::new(&fast.join("|")).ok()?;
        // An alternative that can match the empty string would need `find_iter`'s
        // empty-match rules to be mirrored on the fast path too; refuse instead.
        if regex.is_match("") {
            return None;
        }

        let split = FastSplit { regex };
        if !split.splits_like(fancy) {
            return None;
        }
        Some(split)
    }

    /// Checks that the fast path yields exactly the pieces `fancy.find_iter`
    /// yields, over a corpus covering every character class the tokeniser
    /// patterns distinguish.
    fn splits_like(&self, fancy: &Regex) -> bool {
        // Pieces of text that the split patterns treat differently: the three
        // kinds of whitespace run, letters (cased and uncased), digits, the
        // contraction apostrophe, symbols, and non-ASCII of each relevant class.
        const ATOMS: [&str; 14] = [
            " ", "  ", "\t", "\n", "\r\n", "a", "A", "0", "'", "!", "é", "中", "\u{a0}", "\u{0301}",
        ];
        const EXTRA: [&str; 8] = [
            "",
            "The quick brown fox jumps over the lazy dog.",
            "It's a test, isn't it? 42 apples\n\nand 1,000 oranges.",
            "trailing spaces   ",
            "\n\n\n",
            "  leading",
            "ЖЖ ЖЖ 漢字テスト",
            "emoji 🙂🙃 and combining a\u{0301}e\u{0301}",
        ];

        let mut probes: Vec<String> = EXTRA.iter().map(|s| (*s).to_owned()).collect();
        for a in ATOMS {
            probes.push(a.to_owned());
            for b in ATOMS {
                probes.push(format!("{a}{b}"));
                for c in ATOMS {
                    probes.push(format!("{a}{b}{c}"));
                }
            }
        }

        probes.iter().all(|probe| {
            let expected: Result<Vec<&str>, _> = fancy
                .find_iter(probe)
                .map(|res| res.map(|mat| mat.as_str()))
                .collect();
            let actual: Result<Vec<&str>, _> = FastPieces::new(probe, fancy, &self.regex).collect();
            match (expected, actual) {
                (Ok(expected), Ok(actual)) => expected == actual,
                (Err(_), Err(_)) => true,
                _ => false,
            }
        })
    }
}

/// Splits `pattern` into its top-level alternatives, or `None` if the scanner
/// does not fully understand its structure.
fn top_level_alternatives(pattern: &str) -> Option<Vec<&str>> {
    let bytes = pattern.as_bytes();
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0usize;
    let mut in_class = false;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 1,
            b'[' if !in_class => in_class = true,
            b']' if in_class => in_class = false,
            b'(' if !in_class => depth += 1,
            b')' if !in_class => depth = depth.checked_sub(1)?,
            b'|' if !in_class && depth == 0 => {
                parts.push(&pattern[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if in_class || depth != 0 {
        return None;
    }
    parts.push(&pattern[start..]);
    Some(parts)
}

/// Rewrites one top-level alternative into the equivalent form the `regex` crate
/// accepts, or `None` if it uses a construct that has to stay with `fancy_regex`.
///
/// The only rewrite is dropping possessive markers (`++`, `?+`, `*+`, `{n,m}+`),
/// which `regex` has no syntax for. That is not equivalence-preserving in
/// general, which is exactly why the result is verified against `fancy_regex`
/// before it is used (see [`FastSplit::splits_like`]).
fn regex_crate_alternative(alt: &str) -> Option<String> {
    let bytes = alt.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    let mut in_class = false;
    // Whether the token just copied was a quantifier, in which case a following
    // `+` is a possessive marker rather than a quantifier of its own.
    let mut after_quantifier = false;

    while i < bytes.len() {
        let b = bytes[i];
        if in_class {
            if b == b'\\' {
                out.extend_from_slice(&[b, *bytes.get(i + 1)?]);
                i += 2;
                continue;
            }
            if b == b']' {
                in_class = false;
            }
            out.push(b);
            i += 1;
            continue;
        }
        match b {
            b'\\' => {
                let next = *bytes.get(i + 1)?;
                match next {
                    // Anchors, word boundaries and back-references are either
                    // position-dependent or unsupported; give up on them.
                    b'A' | b'z' | b'Z' | b'b' | b'B' | b'G' | b'K' | b'0'..=b'9' => return None,
                    _ => {}
                }
                out.extend_from_slice(&[b, next]);
                i += 2;
                after_quantifier = false;
                // The `{...}` of `\p{..}` is part of the class, not a repetition.
                if (next == b'p' || next == b'P') && bytes.get(i) == Some(&b'{') {
                    while i < bytes.len() {
                        out.push(bytes[i]);
                        i += 1;
                        if bytes[i - 1] == b'}' {
                            break;
                        }
                    }
                }
            }
            b'[' => {
                in_class = true;
                out.push(b);
                i += 1;
                after_quantifier = false;
            }
            // Anchors make the alternative depend on where the haystack ends, and
            // the fast path searches the haystack unanchored.
            b'^' | b'$' => return None,
            b'(' => {
                let rest = &bytes[i..];
                if rest.starts_with(b"(?:") {
                    out.extend_from_slice(b"(?:");
                    i += 3;
                } else if rest.starts_with(b"(?") {
                    // Inline flags, e.g. `(?i:` or `(?i)`. Anything else after
                    // `(?` (lookaround, atomic group, named group, ...) is out.
                    let mut j = i + 2;
                    while matches!(
                        bytes.get(j),
                        Some(b'i' | b'm' | b's' | b'x' | b'u' | b'U' | b'R' | b'-')
                    ) {
                        j += 1;
                    }
                    if j > i + 2 && matches!(bytes.get(j), Some(b':' | b')')) {
                        out.extend_from_slice(&bytes[i..=j]);
                        i = j + 1;
                    } else {
                        return None;
                    }
                } else {
                    // A capturing group would renumber this alternative's groups
                    // when they are joined; not worth reasoning about.
                    return None;
                }
                after_quantifier = false;
            }
            b'*' | b'?' => {
                out.push(b);
                i += 1;
                after_quantifier = true;
            }
            b'+' => {
                if after_quantifier {
                    // Possessive marker: drop it, leaving a greedy quantifier.
                    i += 1;
                    after_quantifier = false;
                } else {
                    out.push(b);
                    i += 1;
                    after_quantifier = true;
                }
            }
            b'{' => {
                let end = i + bytes[i..].iter().position(|&c| c == b'}')?;
                if !bytes[i + 1..end]
                    .iter()
                    .all(|&c| c.is_ascii_digit() || c == b',')
                {
                    return None;
                }
                out.extend_from_slice(&bytes[i..=end]);
                i = end + 1;
                after_quantifier = true;
            }
            b'}' => return None,
            _ => {
                out.push(b);
                i += 1;
                after_quantifier = false;
            }
        }
    }
    if in_class {
        return None;
    }
    String::from_utf8(out).ok()
}

/// Yields the pieces of a text, exactly as `fancy_regex::Regex::find_iter` would.
enum SplitPieces<'r, 't> {
    Fancy(fancy_regex::Matches<'r, 't>),
    Fast(FastPieces<'r, 't>),
}

impl<'t> Iterator for SplitPieces<'_, 't> {
    type Item = fancy_regex::Result<&'t str>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SplitPieces::Fancy(matches) => matches.next().map(|res| res.map(|mat| mat.as_str())),
            SplitPieces::Fast(pieces) => pieces.next(),
        }
    }
}

/// Drives the split with one streaming `regex` search over the whole haystack,
/// handing the positions it does not match to `fancy_regex`.
struct FastPieces<'r, 't> {
    text: &'t str,
    fancy: &'r Regex,
    fast: &'r FastRegex,
    /// Streaming matches of `fast` over `text[base..]`.
    matches: Option<regex::Matches<'r, 't>>,
    base: usize,
    /// The leftmost `fast` match at or after `pos`, once it is known.
    pending: Option<(usize, usize)>,
    /// Where the next piece starts.
    pos: usize,
    /// End of the previously yielded piece, for `find_iter`'s empty-match rule.
    last_match: Option<usize>,
}

impl<'r, 't> FastPieces<'r, 't> {
    fn new(text: &'t str, fancy: &'r Regex, fast: &'r FastRegex) -> FastPieces<'r, 't> {
        FastPieces {
            text,
            fancy,
            fast,
            matches: Some(fast.find_iter(text)),
            base: 0,
            pending: None,
            pos: 0,
            last_match: None,
        }
    }

    /// Restarts the streaming search at `pos` after the `fancy_regex` fallback
    /// stepped over a position the fast search had already passed.
    fn resync(&mut self) {
        match self.pending {
            // Still ahead of us, so still the leftmost fast match from here.
            Some((start, _)) if start >= self.pos => return,
            Some(_) => self.pending = None,
            None => {}
        }
        if self.matches.is_some() && self.pos <= self.text.len() {
            self.base = self.pos;
            self.matches = Some(self.fast.find_iter(&self.text[self.pos..]));
        }
    }
}

impl<'t> Iterator for FastPieces<'_, 't> {
    type Item = fancy_regex::Result<&'t str>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.pos > self.text.len() {
                return None;
            }

            // Pull matches off the streaming search until we know the leftmost one
            // at or after `pos` (if the rest of the haystack has one at all).
            while self.pending.is_none() {
                let Some(matches) = self.matches.as_mut() else {
                    break;
                };
                match matches.next() {
                    Some(mat) => {
                        let (start, end) = (self.base + mat.start(), self.base + mat.end());
                        if start >= self.pos {
                            self.pending = Some((start, end));
                        }
                    }
                    None => self.matches = None,
                }
            }

            if let Some((start, end)) = self.pending
                && start == self.pos
            {
                // The fast alternatives match here, so the full pattern matches
                // here with the same alternative: this is the piece.
                self.pending = None;
                self.pos = end;
                self.last_match = Some(end);
                return Some(Ok(&self.text[start..end]));
            }

            // Nothing here for the fast alternatives: let `fancy_regex` decide
            // this position, exactly as `find_iter` does.
            let mat = match self.fancy.find_from_pos(self.text, self.pos) {
                Ok(Some(mat)) => mat,
                Ok(None) => {
                    self.pos = self.text.len() + 1;
                    return None;
                }
                Err(err) => {
                    self.pos = self.text.len() + 1;
                    return Some(Err(err));
                }
            };
            let (start, end) = (mat.start(), mat.end());
            if start == end {
                // Empty match: start the next search at the smallest possible
                // position of the following match, and never report an empty
                // match sitting where the previous match ended.
                self.pos = end + self.text[end..].chars().next().map_or(1, |c| c.len_utf8());
                self.resync();
                if self.last_match == Some(end) {
                    self.last_match = Some(end);
                    continue;
                }
            } else {
                self.pos = end;
                self.resync();
            }
            self.last_match = Some(end);
            return Some(Ok(&self.text[start..end]));
        }
    }
}

// Various performance notes:
//
// Regex
// =====
// Most of the time is spent in regex. The easiest way to speed this up is by using less fancy
// regex features. For instance, using a regex parse-able by `regex` crate is 3x faster than
// the usual regex we use.
//
// However, given that we're using a regex parse-able by `regex`, there isn't much difference
// between using the `regex` crate and using the `fancy_regex` crate.
//
// There is an important interaction between threading, `regex` and `fancy_regex`.
// When using `fancy_regex`, we hit `regex.find_at`. It turns out that this causes contention on
// some mutable scratch space inside of `regex`. This absolutely kills performance. When using plain
// old `regex`, we don't hit this, because `find_iter` has a different code path.
// Related: https://github.com/rust-lang/regex/blob/master/PERFORMANCE.md
// Anyway, the way we get around this is with having a (mostly) thread local clone of the regex for
// each thread.
//
// Threading
// =========
// I tried using `rayon`. It wasn't really faster than using Python threads and releasing the GIL.
// So goodbye `rayon`! Let thread count etc be in control of our Python users.
//
// Caching
// =======
// The reference tokeniser has an lru cache over the equivalent of `byte_pair_encode`.
// Originally, we had one too! Without it, we were only vaguely faster than Python.
// I used an RWLock to protect the cache. This didn't seem to hurt single threaded performance
// noticeably, but it did affect multi-threaded performance. Weirdly, it seemed to affect
// multi-threaded performance even when I only had readers (maybed I messed something up?).
// Anyway, I realised that we could get rid of the cache, if we treat the set of tokens as a cache!
// These are exactly the set or merges that are likely to be hot. And now we don't have to think
// about interior mutability, memory use, or cloning.
//
// Hashing
// =======
// We use FxHashMap instead of the standard HashMap. This is maybe like a 5-10% win?
// The current implementation ends up doing a lot of hashing of bytes. In theory, this could be made
// to be hashing of two-tuples of ints, which looks like it may also be a couple percent faster.

struct FakeThreadId(NonZeroU64);

fn hash_current_thread() -> usize {
    // It's easier to use unsafe than to use nightly. Rust has this nice u64 thread id counter
    // that works great for our use case of avoiding collisions in our array. Unfortunately,
    // it's private. However, there are only so many ways you can layout a u64, so just transmute
    // https://github.com/rust-lang/rust/issues/67939
    const _: [u8; 8] = [0; std::mem::size_of::<std::thread::ThreadId>()];
    const _: [u8; 8] = [0; std::mem::size_of::<FakeThreadId>()];
    let x = unsafe {
        std::mem::transmute::<std::thread::ThreadId, FakeThreadId>(thread::current().id()).0
    };
    u64::from(x) as usize
}

#[derive(Debug, Clone)]
pub struct DecodeKeyError {
    pub token: Rank,
}

impl std::fmt::Display for DecodeKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Invalid token for decoding: {}", self.token)
    }
}

impl std::error::Error for DecodeKeyError {}

#[derive(Debug, Clone)]
pub struct DecodeError {
    pub message: String,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Could not decode tokens: {}", self.message)
    }
}

impl std::error::Error for DecodeError {}

#[derive(Debug, Clone)]
pub struct EncodeError {
    pub message: String,
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Could not encode string: {}", self.message)
    }
}

impl std::error::Error for EncodeError {}

const MAX_NUM_THREADS: usize = 128;

#[cfg_attr(feature = "python", pyclass(frozen))]
#[derive(Clone)]
pub struct CoreBPE {
    encoder: HashMap<Vec<u8>, Rank>,
    special_tokens_encoder: HashMap<String, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    special_tokens_decoder: HashMap<Rank, Vec<u8>>,
    // Flat, rank-indexed view of every token's bytes (regular + special). Token
    // ranks are dense (`0..=max_rank`), so decoding can resolve a token with a
    // single bounds-checked slice index instead of hashing into `decoder` and
    // then falling back to `special_tokens_decoder`. `decode` is the hottest
    // consumer of this and is dominated by per-token lookups, so replacing the
    // hash lookup with a direct index is a measurable win.
    decoder_flat: Vec<Vec<u8>>,
    regex_tls: Vec<Regex>,
    special_regex_tls: Vec<Regex>,
    // The lookaround-free leading alternatives of the split pattern, compiled as
    // one `regex` automaton, so the split is driven by a single streaming search
    // per document instead of one `fancy_regex` VM run per piece. `None` when the
    // pattern is not one this can be proven equivalent for.
    fast_split: Option<FastSplit>,
    sorted_token_bytes: Vec<Vec<u8>>,
}

impl CoreBPE {
    /// Yields the pieces of `text`, exactly as `regex.find_iter(text)` would.
    fn split_pieces<'a>(&'a self, text: &'a str, regex: &'a Regex) -> SplitPieces<'a, 'a> {
        match &self.fast_split {
            Some(fast) => SplitPieces::Fast(FastPieces::new(text, regex, &fast.regex)),
            None => SplitPieces::Fancy(regex.find_iter(text)),
        }
    }

    fn _get_tl_regex(&self) -> &Regex {
        // See performance notes above for what this is about
        // It's also a little janky, please make a better version of it!
        // However, it's nice that this doesn't leak memory to short-lived threads
        &self.regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    fn _get_tl_special_regex(&self) -> &Regex {
        &self.special_regex_tls[hash_current_thread() % MAX_NUM_THREADS]
    }

    /// Resolves a token id to its decoded bytes, checking the regular decoder
    /// first and then the special-tokens decoder.
    ///
    /// This is the single source of truth for the token lookup order used by
    /// every decode path (both the pure-Rust `decode_bytes` and the fused
    /// Python `list` fast path in `py.rs`), so they cannot diverge.
    pub(crate) fn token_bytes(&self, token: Rank) -> Option<&[u8]> {
        // Ranks are dense, so `decoder_flat` covers every valid token with a
        // direct index. Empty entries mark ranks with no token (there should be
        // none for well-formed vocabularies, but we stay defensive), which we
        // treat as a miss.
        match self.decoder_flat.get(token as usize) {
            Some(bytes) if !bytes.is_empty() => Some(bytes.as_slice()),
            _ => None,
        }
    }

    /// Decodes tokens into a list of bytes.
    ///
    /// The bytes are not gauranteed to be a valid utf-8 string.
    fn decode_bytes(&self, tokens: &[Rank]) -> Result<Vec<u8>, DecodeKeyError> {
        let mut ret = Vec::with_capacity(tokens.len() * 2);
        for &token in tokens {
            let token_bytes = self.token_bytes(token).ok_or(DecodeKeyError { token })?;
            ret.extend(token_bytes);
        }
        Ok(ret)
    }

    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        // This is the core of the encoding logic; the other functions in here
        // just make things complicated :-)
        let regex = self._get_tl_regex();
        let mut ret = vec![];
        for piece in self.split_pieces(text, regex) {
            let piece = piece.unwrap().as_bytes();
            match self.encoder.get(piece) {
                Some(token) => ret.push(*token),
                None => ret.extend(&byte_pair_encode(piece, &self.encoder)),
            }
        }
        ret
    }

    pub fn encode(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
    ) -> Result<(Vec<Rank>, usize), EncodeError> {
        let special_regex = self._get_tl_special_regex();
        let regex = self._get_tl_regex();
        let mut ret = vec![];

        let mut start = 0;
        let mut last_piece_token_len = 0;
        loop {
            let mut next_special;
            let mut start_find = start;
            loop {
                // Find the next allowed special token, if any
                next_special = special_regex.find_from_pos(text, start_find).unwrap();
                match next_special {
                    Some(m) => {
                        if allowed_special.contains(&text[m.start()..m.end()]) {
                            break;
                        }
                        start_find = m.start() + 1;
                    }
                    None => break,
                }
            }
            let end = next_special.map_or(text.len(), |m| m.start());

            // Okay, here we go, compare this logic to encode_ordinary
            for piece_res in self.split_pieces(&text[start..end], regex) {
                let piece = match piece_res {
                    Ok(piece) => piece,
                    Err(e) => {
                        return Err(EncodeError {
                            message: format!("Regex error while tokenizing: {e}"),
                        });
                    }
                };

                let piece = piece.as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    last_piece_token_len = 1;
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                last_piece_token_len = tokens.len();
                ret.extend(&tokens);
            }

            match next_special {
                // And here we push the special token
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end();
                    last_piece_token_len = 0;
                }
                None => break,
            }
        }

        // last_piece_token_len is how many tokens came from the last regex split. This is used
        // for determining unstable tokens, since you can't merge across (stable) regex splits
        Ok((ret, last_piece_token_len))
    }

    fn _increase_last_piece_token_len(
        &self,
        tokens: Vec<Rank>,
        mut last_piece_token_len: usize,
    ) -> (Vec<Rank>, usize) {
        // Unfortunately, the locations where our regex splits can be unstable.
        // For the purposes of determining unstable tokens, unstable regex splitting
        // is only a problem if a split that was present disappears, since this can
        // lead to merging of tokens otherwise thought to be stable.
        // cl100k_base makes our life hard by including the \s*[\r\n]+
        // pattern. This can e.g. cause "\n" + " " to become "\n \n".
        // Here is a quick and dirty fix:
        {
            let token_is_all_space = |token| {
                self.decoder
                    .get(token)
                    .map(|token_bytes| {
                        token_bytes
                            .iter()
                            .rev()
                            .all(|&b| [b' ', b'\n', b'\t'].contains(&b))
                    })
                    .unwrap_or(false)
            };
            if last_piece_token_len > 0
                && token_is_all_space(&tokens[tokens.len() - last_piece_token_len])
            {
                while (last_piece_token_len < tokens.len())
                    && token_is_all_space(&tokens[tokens.len() - last_piece_token_len - 1])
                {
                    last_piece_token_len += 1;
                }
            }
        }
        debug_assert!(last_piece_token_len <= tokens.len());

        (tokens, last_piece_token_len)
    }

    pub fn _encode_unstable_native(
        &self,
        text: &str,
        allowed_special: &HashSet<&str>,
    ) -> (Vec<Rank>, HashSet<Vec<Rank>>) {
        let (tokens, last_piece_token_len) = self.encode(text, allowed_special).unwrap();
        if last_piece_token_len == 0 {
            // If last_piece_token_len is zero, the last token was a special token and we have
            // no unstable bytes
            return (tokens, HashSet::new());
        }
        let (mut tokens, last_piece_token_len) =
            self._increase_last_piece_token_len(tokens, last_piece_token_len);

        let unstable_bytes = self
            .decode_bytes(&tokens[tokens.len() - last_piece_token_len..])
            .unwrap();
        tokens.truncate(tokens.len() - last_piece_token_len);

        // TODO: we should try harder to find additional stable tokens
        // This would reduce the amount of retokenising when determining completions
        // Refer to the logic in an older version of this file

        let mut completions = HashSet::new();
        if unstable_bytes.is_empty() {
            return (tokens, completions);
        }

        // This is the easy bit. Just find all single tokens that start with unstable_bytes
        // (including tokens that exactly match unstable_bytes)
        // Separating this from the loop below helps with performance in a common case.
        let mut point = self
            .sorted_token_bytes
            .partition_point(|x| x.as_slice() < unstable_bytes.as_slice());
        while point < self.sorted_token_bytes.len()
            && self.sorted_token_bytes[point].starts_with(&unstable_bytes)
        {
            completions.insert(vec![
                self.encoder[self.sorted_token_bytes[point].as_slice()],
            ]);
            point += 1;
        }

        // Now apply even more brute force. At every (other) possible position for the straddling
        // token, concatenate additional bytes from that token (if any) to unstable_bytes,
        // and retokenise the whole thing and see what we get.
        for i in 1..unstable_bytes.len() {
            let prefix = &unstable_bytes[..i];
            let suffix = &unstable_bytes[i..];
            let mut point = self
                .sorted_token_bytes
                .partition_point(|x| x.as_slice() < suffix);
            // TODO: Perf optimisation if suffix starts with " "?
            while point < self.sorted_token_bytes.len()
                && self.sorted_token_bytes[point].starts_with(suffix)
            {
                let possibility = [prefix, self.sorted_token_bytes[point].as_slice()].concat();
                let encoded = match std::str::from_utf8(&possibility) {
                    // Morally, this is byte_pair_encode(&possibility, &self.encoder)
                    // But we might have introduced a regex split which would prevent merges.
                    // (particularly possible in the presence of unstable regex splits)
                    // So convert to UTF-8 and do regex splitting.
                    // E.g. with cl100k_base "  !" gets split to " " + " !",
                    // but byte_pair_encode("  !") != byte_pair_encode(" ")
                    Ok(s) => self.encode_ordinary(s),

                    // Technically, whether or not this arm is correct depends on whether there
                    // would be a regex split before the UTF-8 truncation point.
                    // Probably niche enough that no one will ever notice (after all, people didn't
                    // notice all the big holes in the previous unstable token implementation)
                    Err(_) => byte_pair_encode(&possibility, &self.encoder),
                    // Something like the following is intriguing but incorrect:
                    // Err(e) => self.encode_ordinary(unsafe {
                    //     std::str::from_utf8_unchecked(&possibility[..e.valid_up_to()])
                    // }),
                };
                let mut seq = Vec::new();
                let mut seq_len = 0;
                for token in encoded {
                    seq.push(token);
                    seq_len += self.decoder[&token].len();
                    if seq_len >= unstable_bytes.len() {
                        break;
                    }
                }
                completions.insert(seq);
                point += 1;
            }
        }

        // This is also not straightforward. While we generally assume that regex splits are stable,
        // unfortunately, they are not. That is, if adding bytes were to make a split appear in
        // unstable_bytes, this could make tokens possible which our logic would otherwise think
        // would be merged.
        // For example, with gpt2, the use of \s+(?!\S) means that "\n\n" could
        // develop a split, e.g. "\n\n0" splits into "\n"+"\n"+"0", making "\n" a possible token.
        // Here is a quick and dirty fix:
        // This isn't right if we ever remove \s+(?!\S)
        if unstable_bytes.len() > 1 {
            let last_decoded = bstr::decode_last_utf8(unstable_bytes.as_slice());
            if unstable_bytes.len() - last_decoded.1 > 0
                && last_decoded.0.is_some_and(|c| c.is_whitespace())
            {
                let mut reencoded = byte_pair_encode(
                    &unstable_bytes[..unstable_bytes.len() - last_decoded.1],
                    &self.encoder,
                );
                reencoded.extend(byte_pair_encode(
                    &unstable_bytes[unstable_bytes.len() - last_decoded.1..],
                    &self.encoder,
                ));
                completions.insert(reencoded);
            }
        }

        (tokens, completions)
    }

    pub fn new<E, SE, NSE>(
        encoder: E,
        special_tokens_encoder: SE,
        pattern: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>>
    where
        E: IntoIterator<Item = (Vec<u8>, Rank)>,
        SE: IntoIterator<Item = (String, Rank)>,
        NSE: IntoIterator<Item = (String, (Rank, Rank))>,
    {
        Self::new_internal(
            HashMap::from_iter(encoder),
            HashMap::from_iter(special_tokens_encoder),
            pattern,
        )
    }

    fn new_internal(
        encoder: HashMap<Vec<u8>, Rank>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let special_pattern = special_tokens_encoder
            .keys()
            .map(|s| fancy_regex::escape(s))
            .collect::<Vec<_>>()
            .join("|");

        let decoder: HashMap<Rank, Vec<u8>> =
            encoder.iter().map(|(k, v)| (*v, k.clone())).collect();

        assert!(
            encoder.len() == decoder.len(),
            "Encoder and decoder must be of equal length. Encoder length: {}, decoder length: {}.\nMaybe you had duplicate token indices in your encoder?",
            encoder.len(),
            decoder.len()
        );

        let special_tokens_decoder: HashMap<Rank, Vec<u8>> = special_tokens_encoder
            .iter()
            .map(|(k, v)| (*v, k.as_bytes().to_vec()))
            .collect();

        // Build a flat, rank-indexed decode table. Token ranks are dense, so
        // this lets the decode hot path resolve a token with a single indexed
        // slice lookup instead of two hash lookups. Empty slots correspond to
        // ranks with no token (none for well-formed vocabularies).
        let max_rank = decoder
            .keys()
            .chain(special_tokens_decoder.keys())
            .copied()
            .max();
        let decoder_flat: Vec<Vec<u8>> = match max_rank {
            Some(max_rank) => {
                let mut flat = vec![Vec::new(); max_rank as usize + 1];
                for (&rank, bytes) in decoder.iter().chain(special_tokens_decoder.iter()) {
                    flat[rank as usize] = bytes.clone();
                }
                flat
            }
            None => Vec::new(),
        };

        // Clone because I don't know how to tell Rust I'm not going to change the map
        let mut sorted_token_bytes: Vec<Vec<u8>> = encoder.keys().cloned().collect();
        sorted_token_bytes.sort();

        // Compile an independent regex per thread-local slot instead of cloning one. Cloning a
        // `fancy_regex::Regex` shares an `Arc<Prog>` whose regex-automata engines keep their scratch
        // in a single mutex-guarded `Pool`, so cloned slots contend on that pool's slow path during
        // multi-threaded batch encoding. Compiling per slot gives each thread its own pool (fast,
        // lock-free path), paying the compile cost once at construction rather than on the hot path.
        let regex_tls = (0..MAX_NUM_THREADS)
            .map(|_| Regex::new(pattern))
            .collect::<Result<Vec<_>, _>>()?;
        let special_regex_tls = (0..MAX_NUM_THREADS)
            .map(|_| Regex::new(&special_pattern))
            .collect::<Result<Vec<_>, _>>()?;

        // Derive the streaming fast splitter, but only if it is verified at
        // construction to split exactly like `fancy_regex` does; otherwise the
        // tokeniser keeps splitting with `find_iter` verbatim.
        let fast_split = FastSplit::new(pattern, &regex_tls[0]);

        Ok(Self {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            decoder_flat,
            regex_tls,
            special_regex_tls,
            fast_split,
            sorted_token_bytes,
        })
    }

    pub fn special_tokens(&self) -> HashSet<&str> {
        self.special_tokens_encoder
            .keys()
            .map(|s| s.as_str())
            .collect()
    }

    pub fn encode_with_special_tokens(&self, text: &str) -> Vec<Rank> {
        let allowed_special = self.special_tokens();
        self.encode(text, &allowed_special).unwrap().0
    }
}

#[cfg(test)]
mod tests {
    use fancy_regex::Regex;
    use rustc_hash::FxHashMap as HashMap;

    use crate::{Rank, byte_pair_split};

    fn setup_ranks() -> HashMap<Vec<u8>, Rank> {
        HashMap::from_iter([(b"ab".to_vec(), 0), (b"cd".to_vec(), 1)])
    }

    #[test]
    fn test_simple_characters() {
        let ranks = setup_ranks();
        let res = byte_pair_split(b"abcd", &ranks);
        assert_eq!(res, vec![b"ab", b"cd"]);
    }

    #[test]
    fn test_repeated_characters() {
        let ranks = setup_ranks();
        let res = byte_pair_split(b"abab", &ranks);
        assert_eq!(res, vec![b"ab", b"ab"]);
    }

    const R50K_PAT: &str =
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";
    const CL100K_PAT: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";
    const O200K_PAT: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// The shipped patterns must all take the fast path, and it must yield the
    /// same pieces `fancy_regex` does (this is what `FastSplit::new` verifies, so
    /// building it at all is the assertion).
    #[test]
    fn test_fast_split_matches_fancy_for_shipped_patterns() {
        for pattern in [R50K_PAT, CL100K_PAT, O200K_PAT] {
            let fancy = Regex::new(pattern).unwrap();
            assert!(
                crate::FastSplit::new(pattern, &fancy).is_some(),
                "fast split rejected for {pattern}"
            );
        }
    }

    /// Longer, more varied texts than the construction-time corpus, checked piece
    /// for piece against `fancy_regex`.
    #[test]
    fn test_fast_split_matches_fancy_on_texts() {
        let texts = [
            "The quick brown fox jumps over the lazy dog.",
            "It's a test, isn't it? 42 apples\n\nand 1,000 oranges.",
            "  multiple   spaces\tand\ttabs  ",
            "windows\r\nnewlines\r\n\r\nand trailing\r\n",
            "ЖЖ ЖЖ 漢字テスト ١٢٣ Ⅻ",
            "emoji 🙂🙃 combining a\u{0301}e\u{0301} nbsp\u{a0}x",
            "'s 'T 've'll'd '",
            "12345 678 9 0x1F",
            "",
            " ",
            "\n",
            "a",
        ];
        for pattern in [R50K_PAT, CL100K_PAT, O200K_PAT] {
            let fancy = Regex::new(pattern).unwrap();
            let fast = crate::FastSplit::new(pattern, &fancy).unwrap();
            for text in texts {
                let expected: Vec<&str> = fancy
                    .find_iter(text)
                    .map(|res| res.unwrap().as_str())
                    .collect();
                let actual: Vec<&str> = crate::FastPieces::new(text, &fancy, &fast.regex)
                    .map(|res| res.unwrap())
                    .collect();
                assert_eq!(expected, actual, "pattern {pattern} text {text:?}");
            }
        }
    }

    /// Patterns whose first alternative needs `fancy_regex` keep the old path.
    #[test]
    fn test_fast_split_refuses_unsupported_patterns() {
        for pattern in [r"(?=a)b|c", r"\ba+|b", r"(a)\1|b", r"a\G|b", r"^a|b"] {
            let fancy = Regex::new(pattern).unwrap();
            assert!(
                crate::FastSplit::new(pattern, &fancy).is_none(),
                "fast split accepted for {pattern}"
            );
        }
    }

    #[test]
    fn test_possessive_markers_are_dropped_carefully() {
        assert_eq!(
            crate::regex_crate_alternative(r" ?\p{L}++").as_deref(),
            Some(r" ?\p{L}+")
        );
        assert_eq!(
            crate::regex_crate_alternative(r"[^\r\n\p{L}\p{N}]?+\p{L}++").as_deref(),
            Some(r"[^\r\n\p{L}\p{N}]?\p{L}+")
        );
        assert_eq!(
            crate::regex_crate_alternative(r"\p{N}{1,3}+").as_deref(),
            Some(r"\p{N}{1,3}")
        );
        // `\++` is one-or-more literal `+`, not a possessive marker.
        assert_eq!(
            crate::regex_crate_alternative(r"\++").as_deref(),
            Some(r"\++")
        );
        assert_eq!(crate::regex_crate_alternative(r"\s++$"), None);
        assert_eq!(crate::regex_crate_alternative(r"\s+(?!\S)"), None);
    }
}
