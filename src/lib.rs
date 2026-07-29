use std::collections::HashSet;
use std::num::NonZeroU64;
use std::thread;

use fancy_regex::Regex;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use regex_automata::{Input as RaInput, PatternID, meta::Regex as MetaRegex};
use regex_syntax::hir::{Class, ClassUnicode, ClassUnicodeRange, Hir, HirKind, Look};
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

// Splitting the text with the tokeniser pattern is by far the hottest part of encoding: it is
// where ~80% of `encode`/`encode_ordinary` is spent. The patterns shipped with tiktoken use two
// "fancy" constructs that force `fancy_regex` onto its backtracking VM, which allocates a fresh
// state (several `Vec`s) for *every* match it produces:
//
//   * possessive quantifiers (`\p{L}++`, `[^\r\n\p{L}\p{N}]?+`, `\p{N}{1,3}+`, ...)
//   * the negative lookahead in `\s+(?!\S)`
//
// Neither is actually needed: below we rewrite such a pattern into an equivalent *lookaround-free*
// set of patterns that the `regex` engine can run as a single automaton, with no backtracking VM
// and no per-match allocation. The rewrite is only used when every step of it is provably
// equivalence preserving (see `relax_alternative`); anything we cannot prove falls back to
// `fancy_regex`, so behaviour is unchanged for arbitrary user-supplied patterns.

/// The negative-lookahead idiom used by every tiktoken pattern to keep the last whitespace
/// character of a run attached to the following piece.
const WS_LOOKAHEAD: &str = r"\s+(?!\S)";

/// Lookaround-free stand-ins for [`WS_LOOKAHEAD`].
///
/// `\s+(?!\S)` matches a greedy run of whitespace that is followed by whitespace or by the end of
/// the text. Since `\s+` is greedy, the character after the run is never whitespace, so the
/// alternative can only succeed by giving back its last character (or by ending at the end of the
/// text). That is exactly:
///
/// * `\s+$` – the whole run, when it ends the text; and
/// * `\s\s+` – a run of at least two characters, minus its last character (dropped by
///   [`FastMatches`] as a fixup when the match does not end the text).
///
/// A single whitespace character followed by a non-whitespace one matches neither, which mirrors
/// the original alternative failing and the search falling through to the next alternative.
const WS_LOOKAHEAD_REPLACEMENTS: [&str; 2] = [r"\s+$", r"\s\s+"];

/// One `atom quantifier` pair of a regex alternative.
struct Element<'a> {
    atom: &'a str,
    /// `""`, `"*"`, `"+"`, `"?"` or `"{m,n}"`, possibly with a trailing `?` (lazy).
    quantifier: &'a str,
    /// Whether the quantifier carried a possessive `+` marker.
    possessive: bool,
}

/// Splits `pattern` on its top-level `|`, ignoring `|` inside groups, character classes and
/// escapes. Returns `None` if the pattern is not balanced.
fn split_alternatives(pattern: &str) -> Option<Vec<&str>> {
    let b = pattern.as_bytes();
    let mut alts = Vec::new();
    let mut depth = 0usize;
    let mut in_class = false;
    let mut start = 0usize;
    let mut i = 0usize;
    while i < b.len() {
        match b[i] {
            b'\\' => i += 1,
            b'[' if !in_class => in_class = true,
            b']' if in_class => in_class = false,
            b'(' if !in_class => depth += 1,
            b')' if !in_class => depth = depth.checked_sub(1)?,
            b'|' if !in_class && depth == 0 => {
                alts.push(&pattern[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if in_class || depth != 0 {
        return None;
    }
    alts.push(&pattern[start..]);
    Some(alts)
}

/// Tokenises one alternative into `atom quantifier` elements. Returns `None` for anything the
/// simple scanner does not understand (in which case we keep using `fancy_regex`).
fn parse_elements(alt: &str) -> Option<Vec<Element<'_>>> {
    let b = alt.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        let atom_start = i;
        match b[i] {
            b'\\' => {
                i += 1;
                let c = *b.get(i)?;
                i += 1;
                // `\p{...}`, `\P{...}` and `\x{...}` carry a braced argument.
                if matches!(c, b'p' | b'P' | b'x') && b.get(i) == Some(&b'{') {
                    while *b.get(i)? != b'}' {
                        i += 1;
                    }
                    i += 1;
                }
            }
            b'[' => {
                i += 1;
                if b.get(i) == Some(&b'^') {
                    i += 1;
                }
                if b.get(i) == Some(&b']') {
                    i += 1;
                }
                loop {
                    match *b.get(i)? {
                        b'\\' => i += 2,
                        // Nested/POSIX classes: not worth handling, bail out.
                        b'[' => return None,
                        b']' => {
                            i += 1;
                            break;
                        }
                        _ => i += 1,
                    }
                }
            }
            b'(' => {
                let mut depth = 0usize;
                let mut in_class = false;
                loop {
                    match *b.get(i)? {
                        b'\\' => i += 1,
                        b'[' if !in_class => in_class = true,
                        b']' if in_class => in_class = false,
                        b'(' if !in_class => depth += 1,
                        b')' if !in_class => {
                            depth -= 1;
                            if depth == 0 {
                                i += 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
                // A bare flag group like `(?i)` changes the meaning of everything that follows it
                // in the *whole* pattern, which splitting into separate patterns would break.
                let group = &alt[atom_start..i];
                if group.starts_with("(?") && !group[2..].contains(':') {
                    return None;
                }
            }
            _ => {
                i += 1;
                while i < b.len() && (b[i] & 0xC0) == 0x80 {
                    i += 1;
                }
            }
        }
        let atom = &alt[atom_start..i];

        let quant_start = i;
        match b.get(i) {
            Some(b'*') | Some(b'+') | Some(b'?') => i += 1,
            Some(b'{') => {
                let mut j = i + 1;
                while matches!(b.get(j), Some(c) if c.is_ascii_digit() || *c == b',') {
                    j += 1;
                }
                if j > i + 1 && b.get(j) == Some(&b'}') {
                    i = j + 1;
                }
            }
            _ => {}
        }
        let mut quantifier = &alt[quant_start..i];
        let mut possessive = false;
        if !quantifier.is_empty() {
            match b.get(i) {
                Some(b'+') => {
                    possessive = true;
                    i += 1;
                }
                Some(b'?') => {
                    i += 1;
                    quantifier = &alt[quant_start..i];
                }
                _ => {}
            }
        }

        out.push(Element {
            atom,
            quantifier,
            possessive,
        });
    }
    Some(out)
}

fn parse_hir(pattern: &str) -> Option<Hir> {
    regex_syntax::Parser::new().parse(pattern).ok()
}

/// The set of characters `hir` can start with, unioned into `acc`. Returns whether `hir` can match
/// without consuming any character, or `None` if we cannot analyse it.
fn first_set(hir: &Hir, acc: &mut ClassUnicode) -> Option<bool> {
    match hir.kind() {
        HirKind::Empty => Some(true),
        HirKind::Literal(lit) => match std::str::from_utf8(&lit.0).ok()?.chars().next() {
            None => Some(true),
            Some(c) => {
                acc.union(&ClassUnicode::new([ClassUnicodeRange::new(c, c)]));
                Some(false)
            }
        },
        HirKind::Class(Class::Unicode(cls)) => {
            acc.union(cls);
            Some(false)
        }
        HirKind::Class(Class::Bytes(_)) => None,
        // Zero-width: cannot consume a character, so keep looking at what follows.
        HirKind::Look(_) => Some(true),
        HirKind::Repetition(rep) => {
            let sub_nullable = first_set(&rep.sub, acc)?;
            Some(rep.min == 0 || sub_nullable)
        }
        HirKind::Capture(cap) => first_set(&cap.sub, acc),
        HirKind::Concat(subs) => {
            for sub in subs {
                if !first_set(sub, acc)? {
                    return Some(false);
                }
            }
            Some(true)
        }
        HirKind::Alternation(subs) => {
            let mut nullable = false;
            for sub in subs {
                nullable |= first_set(sub, acc)?;
            }
            Some(nullable)
        }
    }
}

/// Is dropping the possessive marker of `elements[idx]` (i.e. turning it into a plain greedy
/// quantifier) guaranteed to preserve the meaning of the alternative?
///
/// A possessive quantifier differs from a greedy one only when the greedy one *backtracks*, i.e.
/// gives characters back so that the rest of the alternative can match. So the rewrite is safe
/// whenever such a backtrack can never succeed (or can never happen at all).
fn possessive_drop_is_safe(elements: &[Element<'_>], idx: usize) -> bool {
    let rest: String = elements[idx + 1..]
        .iter()
        .map(|e| [e.atom, e.quantifier].concat())
        .collect();
    let Some(rest_hir) = parse_hir(&rest) else {
        return false;
    };
    let props = rest_hir.properties();

    // (a) The rest matches the empty string unconditionally, so the greedy quantifier always
    //     succeeds at its longest match and never backtracks.
    if props.minimum_len() == Some(0) && props.look_set().is_empty() {
        return true;
    }
    // (b) The rest is just an end-of-text anchor. Backtracking moves the anchor strictly before
    //     the end of the text, where it can never match.
    if matches!(rest_hir.kind(), HirKind::Look(Look::End)) {
        return true;
    }
    // (c) The rest must consume at least one character, and that character can never be one the
    //     quantifier consumed, so giving characters back can never let it match.
    let Some(atom_hir) = parse_hir(elements[idx].atom) else {
        return false;
    };
    let mut consumed = ClassUnicode::empty();
    match atom_hir.kind() {
        HirKind::Class(Class::Unicode(cls)) => consumed.union(cls),
        HirKind::Literal(_) => {
            if first_set(&atom_hir, &mut consumed) != Some(false) {
                return false;
            }
        }
        // Only single-character atoms let us reason about what a backtrack gives back.
        _ => return false,
    }
    let mut follows = ClassUnicode::empty();
    if first_set(&rest_hir, &mut follows) != Some(false) {
        return false;
    }
    consumed.intersect(&follows);
    consumed.ranges().is_empty()
}

/// Rewrites one alternative into an equivalent lookaround-free one, or `None` if we cannot.
fn relax_alternative(alt: &str) -> Option<String> {
    let elements = parse_elements(alt)?;
    if !elements.iter().any(|e| e.possessive) {
        return Some(alt.to_string());
    }
    for (idx, element) in elements.iter().enumerate() {
        if element.possessive && !possessive_drop_is_safe(&elements, idx) {
            return None;
        }
    }
    Some(
        elements
            .iter()
            .map(|e| [e.atom, e.quantifier].concat())
            .collect(),
    )
}

/// A lookaround-free rewrite of a tokeniser pattern, run as a multi-pattern `regex` automaton.
///
/// Each top-level alternative of the original pattern becomes one pattern; leftmost-first matching
/// over the pattern list is exactly the leftmost-first semantics of the original alternation.
#[derive(Clone)]
struct FastSplitter {
    regex: MetaRegex,
    /// Pattern id of the `\s\s+` stand-in for `\s+(?!\S)`, whose match needs the lookahead fixup.
    ws_fixup: Option<PatternID>,
}

impl FastSplitter {
    fn new(pattern: &str) -> Option<Self> {
        let mut patterns: Vec<String> = Vec::new();
        let mut ws_fixup = None;
        for alt in split_alternatives(pattern)? {
            if alt == WS_LOOKAHEAD {
                ws_fixup = Some(PatternID::new(patterns.len() + 1).ok()?);
                patterns.extend(WS_LOOKAHEAD_REPLACEMENTS.iter().map(|p| p.to_string()));
            } else {
                patterns.push(relax_alternative(alt)?);
            }
        }
        let regex = MetaRegex::new_many(&patterns).ok()?;
        Some(FastSplitter { regex, ws_fixup })
    }

    fn find_iter<'r, 't>(&'r self, text: &'t str) -> FastMatches<'r, 't> {
        FastMatches {
            splitter: self,
            text,
            pos: 0,
            last_match: None,
        }
    }
}

struct FastMatches<'r, 't> {
    splitter: &'r FastSplitter,
    text: &'t str,
    pos: usize,
    last_match: Option<usize>,
}

impl<'t> Iterator for FastMatches<'_, 't> {
    type Item = &'t str;

    fn next(&mut self) -> Option<&'t str> {
        loop {
            if self.pos > self.text.len() {
                return None;
            }
            let input = RaInput::new(self.text).span(self.pos..self.text.len());
            let m = self.splitter.regex.search(&input)?;
            let start = m.start();
            let mut end = m.end();
            if Some(m.pattern()) == self.splitter.ws_fixup && end < self.text.len() {
                // `\s+(?!\S)` gives back the last character of the run whenever the run is
                // followed by a non-whitespace character (which greedy matching guarantees here).
                let last = self.text[start..end].chars().next_back()?;
                end -= last.len_utf8();
            }
            if start == end {
                // Empty match: step over one character so we keep making progress, and never
                // report an empty match right after another match (same as `fancy_regex`).
                self.pos = end + self.text[end..].chars().next().map_or(1, char::len_utf8);
                if self.last_match == Some(end) {
                    continue;
                }
            } else {
                self.pos = end;
            }
            self.last_match = Some(end);
            return Some(&self.text[start..end]);
        }
    }
}

/// The regex used to split text into pieces, either the fast lookaround-free rewrite or the
/// original `fancy_regex` pattern.
#[derive(Clone)]
enum Splitter {
    Fast(FastSplitter),
    Fancy(Regex),
}

impl Splitter {
    fn new(pattern: &str) -> fancy_regex::Result<Self> {
        match FastSplitter::new(pattern) {
            Some(fast) => Ok(Splitter::Fast(fast)),
            None => Ok(Splitter::Fancy(Regex::new(pattern)?)),
        }
    }

    /// Builds an independent copy for another thread-local slot.
    ///
    /// Cloning a [`MetaRegex`] gives the copy its own (lock-free) cache pool, but cloning a
    /// `fancy_regex::Regex` shares one, so the fallback path is recompiled instead.
    fn thread_copy(&self, pattern: &str) -> fancy_regex::Result<Self> {
        match self {
            Splitter::Fast(fast) => Ok(Splitter::Fast(fast.clone())),
            Splitter::Fancy(_) => Ok(Splitter::Fancy(Regex::new(pattern)?)),
        }
    }

    fn find_iter<'r, 't>(&'r self, text: &'t str) -> SplitterMatches<'r, 't> {
        match self {
            Splitter::Fast(fast) => SplitterMatches::Fast(fast.find_iter(text)),
            Splitter::Fancy(regex) => SplitterMatches::Fancy(regex.find_iter(text)),
        }
    }
}

enum SplitterMatches<'r, 't> {
    Fast(FastMatches<'r, 't>),
    Fancy(fancy_regex::Matches<'r, 't>),
}

impl<'t> Iterator for SplitterMatches<'_, 't> {
    type Item = fancy_regex::Result<&'t str>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SplitterMatches::Fast(it) => it.next().map(Ok),
            SplitterMatches::Fancy(it) => it.next().map(|m| m.map(|m| m.as_str())),
        }
    }
}

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
    regex_tls: Vec<Splitter>,
    special_regex_tls: Vec<Regex>,
    sorted_token_bytes: Vec<Vec<u8>>,
}

impl CoreBPE {
    fn _get_tl_regex(&self) -> &Splitter {
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
        for mat in regex.find_iter(text) {
            let piece = mat.unwrap().as_bytes();
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
            for mat_res in regex.find_iter(&text[start..end]) {
                let piece = match mat_res {
                    Ok(m) => m.as_bytes(),
                    Err(e) => {
                        return Err(EncodeError {
                            message: format!("Regex error while tokenizing: {e}"),
                        });
                    }
                };

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

        // Give every thread-local slot a regex with its own scratch pool. Sharing one pool across
        // threads makes the regex-automata engines take a mutex-guarded slow path during
        // multi-threaded batch encoding, so each slot gets an independent copy (see
        // `Splitter::thread_copy`), paying that cost once at construction rather than on the hot
        // path.
        let splitter = Splitter::new(pattern)?;
        let regex_tls = (0..MAX_NUM_THREADS)
            .map(|_| splitter.thread_copy(pattern))
            .collect::<Result<Vec<_>, _>>()?;
        let special_regex_tls = (0..MAX_NUM_THREADS)
            .map(|_| Regex::new(&special_pattern))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            decoder_flat,
            regex_tls,
            special_regex_tls,
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

    /// Every piece produced by the fast, lookaround-free splitter must be identical to the one
    /// `fancy_regex` produces for the original pattern.
    fn assert_same_split(pattern: &str, text: &str) {
        let fast = crate::FastSplitter::new(pattern)
            .unwrap_or_else(|| panic!("expected a fast path for {pattern}"));
        let expected: Vec<&str> = Regex::new(pattern)
            .unwrap()
            .find_iter(text)
            .map(|m| m.unwrap().as_str())
            .collect();
        let actual: Vec<&str> = fast.find_iter(text).collect();
        assert_eq!(actual, expected, "pattern {pattern:?} on text {text:?}");
    }

    #[test]
    fn test_fast_splitter_matches_fancy_regex() {
        let texts = [
            "",
            " ",
            "  ",
            "\n",
            "hello world",
            "Hello  world  ",
            "  leading and trailing   ",
            "a\n\n\nb",
            " \t \n \r\n  ",
            "It's 12345 tokens, isn't it?!",
            "мир 世界 🌍 emoji  test",
            "tabs\tand\t\tspaces  \n\n  mixed",
            "ALLCAPS and MiXeD'S case's",
            "punctuation!!!...---   ###",
            "trailing whitespace at end   ",
            "1234567890 42 007",
        ];
        for pattern in [R50K_PAT, CL100K_PAT, O200K_PAT] {
            for text in texts {
                assert_same_split(pattern, text);
            }
        }
    }

    #[test]
    fn test_unsafe_possessive_falls_back_to_fancy() {
        // Dropping the possessive marker here would change the meaning (` ?+a` cannot backtrack,
        // ` ?a` can), so no fast path must be built.
        assert!(crate::FastSplitter::new(r" ?+ a| ?").is_none());
        // Lookarounds we do not know about are not rewritten either.
        assert!(crate::FastSplitter::new(r"foo(?=bar)").is_none());
    }
}
