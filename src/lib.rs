use std::collections::HashSet;
use std::num::NonZeroU64;
use std::thread;

use fancy_regex::Regex;
#[cfg(feature = "python")]
use pyo3::prelude::*;
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

// Tokeniser split pattern reordering
// =================================
//
// The patterns tiktoken ships are a single top-level alternation whose first
// alternative is the contraction one (`'(?:[sdmt]|ll|ve|re)` for r50k/gpt2/p50k,
// `'(?i:[sdmt]|ll|ve|re)` for cl100k). Because the patterns also contain
// `\s+(?!\S)`, `fancy_regex` matches them with its backtracking VM, which tries
// the alternatives *in order* at every piece start. The contraction alternative
// requires an apostrophe, so in ordinary prose it fails at essentially every
// piece start, and every failure pays a full anchored `regex-automata` delegate
// search plus the VM's stack push/pop and undo bookkeeping.
//
// `reorder_split_alternatives` moves the alternatives that actually match
// ordinary text in front of that leading run. This is sound whenever the two
// groups cannot match at the same position: the leading alternatives all start
// with the same mandatory literal byte, and an alternative is only promoted when
// it is *proven* it can never match a string starting with that byte. At most one
// of the reordered alternatives can then match at any position, so ordered
// alternation picks the same alternative as before, and the relative order inside
// each group is preserved. Nothing follows the alternation, so a match of an
// alternative is a match of the whole pattern.
//
// The analysis is deliberately conservative: anything it does not fully
// understand (capture groups, back-references, `\G`/`\K`, lookaround, negated
// classes or unknown escapes in the alternatives to promote) leaves the pattern
// exactly as written.

/// Splits a pattern into its top-level alternatives, or `None` if the pattern
/// cannot be scanned confidently (unbalanced groups or classes).
fn split_top_level_alternatives(pattern: &str) -> Option<Vec<&str>> {
    let bytes = pattern.as_bytes();
    let mut alternatives = Vec::new();
    let mut start = 0;
    let mut depth = 0usize;
    let mut in_class = false;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                // Skip the escaped byte; escapes never introduce structure here.
                if i + 1 >= bytes.len() {
                    return None;
                }
                i += 1;
            }
            b'[' if !in_class => in_class = true,
            b']' if in_class => in_class = false,
            b'(' if !in_class => depth += 1,
            b')' if !in_class => depth = depth.checked_sub(1)?,
            b'|' if !in_class && depth == 0 => {
                alternatives.push(&pattern[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if depth != 0 || in_class {
        return None;
    }
    alternatives.push(&pattern[start..]);
    Some(alternatives)
}

/// True if `alternative` uses a construct that makes reordering unsafe:
/// a capture group (reordering would renumber it), a back-reference, `\G` or
/// `\K` (both are relative to where matching started).
fn blocks_reordering(alternative: &str) -> bool {
    let bytes = alternative.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                let Some(&next) = bytes.get(i + 1) else {
                    return true;
                };
                if next.is_ascii_digit() || matches!(next, b'G' | b'K' | b'k') {
                    return true;
                }
                i += 1;
            }
            // A `(` that is not immediately followed by `?` opens a capture
            // group, and reordering would renumber it.
            b'(' => match bytes.get(i + 1) {
                Some(b'?') => match bytes.get(i + 2) {
                    // Python-style named group or named back-reference.
                    Some(b'P') => return true,
                    // `(?<name>` captures, while `(?<=`/`(?<!` are lookbehind.
                    Some(b'<') if !matches!(bytes.get(i + 3), Some(b'=') | Some(b'!')) => {
                        return true;
                    }
                    // A bare flag setter such as `(?i)` applies to the rest of
                    // the enclosing group, later alternatives included, so
                    // moving it would change what they match.
                    _ if is_bare_flag_group(&alternative[i..]) => return true,
                    _ => {}
                },
                _ => return true,
            },
            _ => {}
        }
        i += 1;
    }
    false
}

/// True for a flag setter with no body, such as `(?i)` or `(?-u)`, whose effect
/// extends to the rest of the enclosing group.
fn is_bare_flag_group(s: &str) -> bool {
    let Some(rest) = s.strip_prefix("(?") else {
        return false;
    };
    let mut saw_flag = false;
    for c in rest.chars() {
        match c {
            ')' => return saw_flag,
            'i' | 'm' | 's' | 'x' | 'u' | 'U' | 'R' | '-' => saw_flag = true,
            _ => return false,
        }
    }
    false
}

/// The literal character an alternative *must* start with, if it starts with an
/// unquantified literal character. `None` for anything else.
fn required_first_char(alternative: &str) -> Option<char> {
    let (first, len) = literal_at(alternative)?;
    // A quantifier would make the character optional or repeated, and `{` may
    // start a bound we do not want to reason about.
    if matches!(
        alternative.as_bytes().get(len),
        Some(b'?') | Some(b'*') | Some(b'{')
    ) {
        return None;
    }
    Some(first)
}

/// Parses a literal character (plain or backslash-escaped) at the start of `s`,
/// returning the character and how many bytes it occupies.
fn literal_at(s: &str) -> Option<(char, usize)> {
    let mut chars = s.chars();
    match chars.next()? {
        '\\' => {
            let escaped = chars.next()?;
            // Only punctuation escapes are plain literals; `\d`, `\p{…}`, `\b`,
            // … mean something else.
            if escaped.is_ascii_punctuation() {
                Some((escaped, 1 + escaped.len_utf8()))
            } else {
                None
            }
        }
        c if !"\\^$.|?*+()[]{}".contains(c) => Some((c, c.len_utf8())),
        _ => None,
    }
}

/// A guard character we are willing to reason about: ASCII punctuation, which is
/// never matched by `\p{L}`, `\p{N}`, `\p{M}`, `\d`, `\s` or `\w` (`_`, the one
/// punctuation character `\w` accepts, is excluded).
fn is_supported_guard(c: char) -> bool {
    c.is_ascii_punctuation() && c != '_'
}

/// Reorders the top-level alternatives so the alternatives that can never match
/// at the leading run's guard character are tried first. Returns `None` when no
/// such reordering is provably safe.
fn reorder_split_alternatives(pattern: &str) -> Option<String> {
    let alternatives = split_top_level_alternatives(pattern)?;
    if alternatives.len() < 3 || alternatives.iter().any(|a| blocks_reordering(a)) {
        return None;
    }

    // The leading run: alternatives that all require the same literal character.
    let guard = required_first_char(alternatives[0]).filter(|&c| is_supported_guard(c))?;
    let lead_end = alternatives
        .iter()
        .position(|a| required_first_char(a) != Some(guard))
        .unwrap_or(alternatives.len());

    // The alternatives to promote: the ones right after the leading run that
    // provably cannot match anything starting with `guard`.
    let mut promote_end = lead_end;
    while promote_end < alternatives.len() && cannot_start_with(alternatives[promote_end], guard) {
        promote_end += 1;
    }
    if promote_end == lead_end {
        return None;
    }

    let mut reordered = Vec::with_capacity(alternatives.len());
    reordered.extend_from_slice(&alternatives[lead_end..promote_end]);
    reordered.extend_from_slice(&alternatives[..lead_end]);
    reordered.extend_from_slice(&alternatives[promote_end..]);
    Some(reordered.join("|"))
}

/// Proves that `alternative` can only match strings whose first character is not
/// `guard`. Conservative: returns `false` whenever it cannot prove it, including
/// for alternatives that can match the empty string.
fn cannot_start_with(alternative: &str, guard: char) -> bool {
    let mut rest = alternative;
    loop {
        let Some((can_match_guard, atom_len)) = atom_can_match(rest, guard) else {
            return false;
        };
        if can_match_guard {
            return false;
        }
        let after_atom = &rest[atom_len..];
        let Some((quantifier_len, optional)) = quantifier_at(after_atom) else {
            return false;
        };
        if !optional {
            // A mandatory atom that cannot match `guard` starts every match.
            return true;
        }
        rest = &after_atom[quantifier_len..];
        if rest.is_empty() {
            // Everything was optional, so the alternative can match the empty
            // string, which happens at a `guard` position too.
            return false;
        }
    }
}

/// Parses the atom at the start of `s`, returning whether it can match `guard`
/// and its length in bytes. `None` for anything not fully understood.
fn atom_can_match(s: &str, guard: char) -> Option<(bool, usize)> {
    if let Some(rest) = s.strip_prefix('[') {
        // Character class. `[^…]` matches `guard` exactly when the listed set
        // does not contain it.
        let (negated, body) = match rest.strip_prefix('^') {
            Some(body) => (true, body),
            None => (false, rest),
        };
        let end = body.find(']')?;
        let contains = class_contains(&body[..end], guard)?;
        let len = s.len() - body.len() + end + 1;
        Some((contains != negated, len))
    } else if s.starts_with('\\') {
        let (matches_guard, len) = escape_can_match(s, guard)?;
        Some((matches_guard, len))
    } else {
        let (literal, len) = literal_at(s)?;
        Some((literal == guard, len))
    }
}

/// Whether a class escape (`\d`, `\p{L}`, …) or literal escape matches `guard`.
fn escape_can_match(s: &str, guard: char) -> Option<(bool, usize)> {
    let bytes = s.as_bytes();
    match *bytes.get(1)? {
        // Unicode classes. Only the categories that provably exclude ASCII
        // punctuation are understood; `\P{…}` (negated) is not.
        b'p' => {
            let end = s.find('}')?;
            let name = &s[3..end];
            if *bytes.get(2)? != b'{' || !is_letter_or_number_class(name) {
                return None;
            }
            Some((false, end + 1))
        }
        // Perl classes: none of them contains ASCII punctuation other than `_`,
        // which `is_supported_guard` excludes.
        b'd' | b'w' | b's' => Some((false, 2)),
        // Their negations do match punctuation.
        b'D' | b'W' | b'S' => Some((true, 2)),
        _ => {
            let (literal, len) = literal_at(s)?;
            Some((literal == guard, len))
        }
    }
}

/// Unicode general categories (and shorthands) that contain no ASCII
/// punctuation, so `\p{…}` with one of these can never match a guard character.
fn is_letter_or_number_class(name: &str) -> bool {
    matches!(
        name,
        "L" | "Letter"
            | "Lu"
            | "Ll"
            | "Lt"
            | "Lm"
            | "Lo"
            | "N"
            | "Number"
            | "Nd"
            | "Nl"
            | "No"
            | "M"
            | "Mark"
            | "Mn"
            | "Mc"
            | "Me"
            | "Alphabetic"
    )
}

/// Whether a character class body contains `guard`. `None` if the body uses
/// syntax this analysis does not understand.
fn class_contains(body: &str, guard: char) -> Option<bool> {
    let mut rest = body;
    let mut contains = false;
    while !rest.is_empty() {
        if rest.starts_with('\\') {
            let (matches_guard, len) = escape_can_match(rest, guard)?;
            contains |= matches_guard;
            rest = &rest[len..];
            continue;
        }
        if rest.starts_with('[') {
            // Nested classes and POSIX brackets are not understood.
            return None;
        }
        let (low, low_len) = literal_at(rest)?;
        let after_low = &rest[low_len..];
        if let Some(range_rest) = after_low.strip_prefix('-') {
            if range_rest.is_empty() {
                // Trailing `-` is a literal.
                contains |= guard == '-';
                rest = range_rest;
                continue;
            }
            let (high, high_len) = literal_at(range_rest)?;
            contains |= low <= guard && guard <= high;
            rest = &range_rest[high_len..];
        } else {
            contains |= low == guard;
            rest = after_low;
        }
    }
    Some(contains)
}

/// Parses a quantifier at the start of `s`, returning its length and whether it
/// makes the preceding atom optional. `None` for a bound this analysis cannot
/// read.
fn quantifier_at(s: &str) -> Option<(usize, bool)> {
    let bytes = s.as_bytes();
    let (len, optional) = match bytes.first() {
        Some(b'?') | Some(b'*') => (1, true),
        Some(b'+') => (1, false),
        Some(b'{') => {
            let end = s.find('}')?;
            let lower_bound = s[1..end].split(',').next()?.trim();
            (end + 1, lower_bound == "0" || lower_bound.is_empty())
        }
        _ => return Some((0, false)),
    };
    // A lazy (`?`) or possessive (`+`) marker after the quantifier.
    let len = match bytes.get(len) {
        Some(b'?') | Some(b'+') => len + 1,
        _ => len,
    };
    Some((len, optional))
}

/// Texts used to double-check that a reordered pattern splits exactly like the
/// pattern as written before it is adopted.
const SPLIT_PROBES: &[&str] = &[
    "",
    " ",
    "  ",
    "\n",
    " \n\t ",
    "a",
    "A'",
    " the quick brown fox jumps ",
    "don't it's I'LL we've they're 'tis ''x '",
    "'s't're've'm'll'd'",
    "1 23 456 7890 0x12 3.14",
    "hello, world! -- (yes) [no] {maybe}?",
    "  leading and trailing whitespace   ",
    "Ünïcödé çafé 中文字\u{3000}😀 a\u{301}b",
    "tabs\tand\r\nnewlines\r\n\n  ",
    "mixed123abc'sd 'x'y'z_ \u{a0}nbsp\u{2009}thin",
];

/// Whether two compiled patterns produce identical piece streams on the probes.
fn splits_identically(as_written: &Regex, candidate: &Regex) -> bool {
    SPLIT_PROBES.iter().all(|probe| {
        let pieces = |re: &Regex| -> Vec<Option<std::ops::Range<usize>>> {
            re.find_iter(probe)
                .map(|m| m.ok().map(|m| m.range()))
                .collect()
        };
        pieces(as_written) == pieces(candidate)
    })
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
    regex_tls: Vec<Regex>,
    special_regex_tls: Vec<Regex>,
    sorted_token_bytes: Vec<Vec<u8>>,
}

impl CoreBPE {
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
        for mat in regex.find_iter(text) {
            let piece = mat.unwrap().as_str().as_bytes();
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
                let mat = match mat_res {
                    Ok(m) => m,
                    Err(e) => {
                        return Err(EncodeError {
                            message: format!("Regex error while tokenizing: {e}"),
                        });
                    }
                };

                let piece = mat.as_str().as_bytes();
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

        // Compile the pattern as written first, so an invalid pattern is still rejected here, with
        // the same error, before anything else happens.
        let as_written = Regex::new(pattern)?;

        // `fancy_regex` matches a pattern containing lookaround on its backtracking VM, which tries
        // the top-level alternatives in order at every piece start. Both shipped patterns lead with
        // the contraction alternative, which needs an apostrophe, so in ordinary text it fails at
        // (almost) every piece start and each failure costs a delegated automaton search plus VM
        // bookkeeping. Trying the alternatives that actually match first removes that work. The
        // reordering is provably piece-for-piece equivalent (see `reorder_split_alternatives`) and
        // is double-checked here before being adopted.
        let reordered = reorder_split_alternatives(pattern).and_then(|rewritten| {
            let candidate = Regex::new(&rewritten).ok()?;
            splits_identically(&as_written, &candidate).then_some(rewritten)
        });
        let split_pattern: &str = reordered.as_deref().unwrap_or(pattern);

        // Compile an independent regex per thread-local slot instead of cloning one. Cloning a
        // `fancy_regex::Regex` shares an `Arc<Prog>` whose regex-automata engines keep their scratch
        // in a single mutex-guarded `Pool`, so cloned slots contend on that pool's slow path during
        // multi-threaded batch encoding. Compiling per slot gives each thread its own pool (fast,
        // lock-free path), paying the compile cost once at construction rather than on the hot path.
        let regex_tls = (0..MAX_NUM_THREADS)
            .map(|_| Regex::new(split_pattern))
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

    use crate::{Rank, byte_pair_split, reorder_split_alternatives, splits_identically};

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

    const R50K_PATTERN: &str =
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";
    const CL100K_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";
    const O200K_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    #[test]
    fn test_r50k_pattern_is_reordered() {
        assert_eq!(
            reorder_split_alternatives(R50K_PATTERN).as_deref(),
            Some(
                r" ?\p{L}++| ?\p{N}++|'(?:[sdmt]|ll|ve|re)| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"
            )
        );
    }

    #[test]
    fn test_patterns_without_a_provably_safe_promotion_are_left_alone() {
        // cl100k's letter alternative starts with `[^\r\n\p{L}\p{N}]`, which
        // matches an apostrophe, so nothing may be promoted past the
        // contraction alternative. o200k does not lead with a literal at all.
        assert_eq!(reorder_split_alternatives(CL100K_PATTERN), None);
        assert_eq!(reorder_split_alternatives(O200K_PATTERN), None);
    }

    #[test]
    fn test_reordering_refuses_unsupported_constructs() {
        // Capture group (reordering would renumber it), back-reference, `\G`.
        assert_eq!(reorder_split_alternatives(r"'a|(b)+|\p{L}+"), None);
        assert_eq!(reorder_split_alternatives(r"'a|\1|\p{L}+"), None);
        assert_eq!(reorder_split_alternatives(r"'a|\G\p{L}+|x"), None);
        // Unbalanced pattern: not scanned at all.
        assert_eq!(reorder_split_alternatives(r"'a|(?:b|\p{L}+"), None);
        // The promotable candidate can match the guard character.
        assert_eq!(reorder_split_alternatives(r"'a|.+|x"), None);
        assert_eq!(reorder_split_alternatives(r"'a|[',]+|x"), None);
        // Only optional atoms: matches the empty string, so it may match at an
        // apostrophe too.
        assert_eq!(reorder_split_alternatives(r"'a|\p{L}*|x"), None);
        // A bare flag setter applies to every later alternative.
        assert_eq!(reorder_split_alternatives(r"'a|\p{L}+|(?i)x|y"), None);
    }

    #[test]
    fn test_reordered_pattern_splits_identically() {
        let as_written = Regex::new(R50K_PATTERN).unwrap();
        let reordered = Regex::new(&reorder_split_alternatives(R50K_PATTERN).unwrap()).unwrap();
        assert!(splits_identically(&as_written, &reordered));

        let texts = [
            "don't stop believin'",
            "''''s't 'll've're'm'd",
            "It's 12:34, isn't it? -- yes'",
            "  \n\ttabs 'n' spaces  ",
            "Ünïcödé çafé's 中文字 😀 a\u{301}'s",
            "'",
            "'s",
            "x'y'z",
        ];
        for text in texts {
            let expected: Vec<_> = as_written
                .find_iter(text)
                .map(|m| m.unwrap().range())
                .collect();
            let actual: Vec<_> = reordered
                .find_iter(text)
                .map(|m| m.unwrap().range())
                .collect();
            assert_eq!(expected, actual, "piece streams differ for {text:?}");
        }
    }
}
