use std::collections::HashSet;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::thread;

use fancy_regex::Regex;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use regex_automata::{
    Anchored, MatchKind,
    dfa::{Automaton, StartKind, dense},
    util::primitives::StateID,
};
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

// ============================================================================
// Tokeniser split
// ============================================================================
//
// The split is where encoding spends most of its time. `fancy_regex` has to
// interpret the tokeniser patterns on its backtracking VM (they use possessive
// quantifiers and the `\s+(?!\S)` look-ahead), and it pays that VM's fixed cost
// -- a fresh match state, backtrack bookkeeping and one delegated
// `regex-automata` search per alternative it tries -- *once per piece*, for a
// piece that is only a handful of bytes.
//
// Almost every piece of real text, though, is produced by the pattern's leading
// alternatives, none of which need look-around. `SplitDfa` compiles those into
// a single byte-level DFA, so a piece costs one anchored DFA pass over its own
// bytes: no VM, no backtrack stack, no per-piece allocation. The DFA is
// immutable and needs no scratch space, so one copy serves every thread.
//
// `fancy_regex` remains the source of truth: it decides every position the DFA
// does not match (whitespace runs, and anything at all for a pattern the
// analysis below does not fully understand).

/// Byte-level DFA recognising the leading, look-around-free top-level
/// alternatives of a tokeniser split pattern.
struct SplitDfa {
    dfa: dense::DFA<Vec<u32>>,
    /// The anchored start state. None of the alternatives the DFA covers can
    /// look before the piece start (`plain_alternative` rejects `^`, `\A`,
    /// `\b`, ...), so the start state is the same at every position and does
    /// not have to be looked up per piece.
    start: StateID,
}

/// Limits for building the DFA. The tokeniser patterns' Unicode classes make
/// for a sizeable automaton; anything that does not fit keeps `fancy_regex`.
const SPLIT_DFA_SIZE_LIMIT: usize = 8 * (1 << 20);
const SPLIT_DETERMINIZE_SIZE_LIMIT: usize = 32 * (1 << 20);

impl SplitDfa {
    /// Returns the end of the piece the split produces at `at`, or `None` when
    /// none of the alternatives the DFA covers match there.
    ///
    /// This is `regex-automata`'s anchored forward search, specialised to what
    /// a tokeniser split needs: the start state is known up front, there is no
    /// prefilter (the search is anchored) and a match is never reported early,
    /// so the whole search is one transition per byte of the piece. Running the
    /// automaton directly like this keeps the per-piece cost proportional to
    /// the piece, which matters because a piece is only a handful of bytes.
    #[inline]
    fn piece_end(&self, text: &str, at: usize) -> Option<usize> {
        let dfa = &self.dfa;
        let haystack = text.as_bytes();
        let mut sid = self.start;
        // Matches are delayed by one byte: a match state is entered by reading
        // the byte *after* the end of the match, so `mat` records that index.
        let mut mat = None;
        let mut i = at;
        while i < haystack.len() {
            sid = dfa.next_state(sid, haystack[i]);
            if dfa.is_special_state(sid) {
                if dfa.is_match_state(sid) {
                    mat = Some(i);
                } else if dfa.is_dead_state(sid) {
                    return mat;
                } else if dfa.is_quit_state(sid) {
                    // A byte the DFA refuses to handle: `fancy_regex` decides.
                    return None;
                }
            }
            i += 1;
        }
        // End of the haystack: the alternatives may still match there.
        if dfa.is_match_state(dfa.next_eoi_state(sid)) {
            mat = Some(haystack.len());
        }
        mat
    }
}

/// Splits `pattern` into its top-level alternatives.
///
/// Returns `None` for anything this deliberately simple scanner does not fully
/// understand, in which case no fast path is built at all.
fn top_level_alternatives(pattern: &str) -> Option<Vec<&str>> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0usize;
    let mut in_class = false;
    let mut chars = pattern.char_indices();
    while let Some((i, c)) = chars.next() {
        match c {
            '\\' => {
                // Skip the escaped character, whatever it is.
                chars.next()?;
            }
            '[' if !in_class => in_class = true,
            ']' if in_class => in_class = false,
            '(' if !in_class => depth += 1,
            ')' if !in_class => depth = depth.checked_sub(1)?,
            '|' if !in_class && depth == 0 => {
                parts.push(&pattern[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if in_class || depth != 0 {
        return None;
    }
    parts.push(&pattern[start..]);
    Some(parts)
}

/// Rewrites one top-level alternative into a form the `regex-automata` DFA
/// compiler accepts, or returns `None` if the alternative uses a construct
/// whose meaning is not a function of the text at the piece start alone.
///
/// The only rewrite is dropping possessive markers (`++`, `*+`, `?+`,
/// `{m,n}+`), which `regex-automata` has no syntax for. A possessive
/// quantifier differs from a greedy one only when the greedy one would
/// backtrack, so this is not universally sound -- which is why the resulting
/// DFA is only adopted after it is verified to split exactly like
/// `fancy_regex` (see `SplitDfa::agrees_with`).
fn plain_alternative(alt: &str) -> Option<String> {
    let mut out = String::with_capacity(alt.len());
    let mut chars = alt.chars().peekable();
    let mut in_class = false;
    // Whether the last thing written was a quantifier, so that a `+` following
    // it is a possessive marker rather than a quantifier of its own.
    let mut prev_quantifier = false;
    while let Some(c) = chars.next() {
        if in_class {
            match c {
                '\\' => {
                    out.push(c);
                    out.push(chars.next()?);
                }
                // Nested or POSIX classes, and class set operations, are not
                // worth reasoning about here.
                '[' | '&' | '~' => return None,
                ']' => {
                    in_class = false;
                    out.push(c);
                }
                _ => out.push(c),
            }
            continue;
        }
        match c {
            '\\' => {
                let e = chars.next()?;
                match e {
                    // Assertions and references that depend on more than the
                    // text at the piece start, or that `find_iter` handles
                    // specially.
                    'G' | 'K' | 'b' | 'B' | 'A' | 'z' | 'Z' | 'g' | 'k' => return None,
                    // Back-references (and octal escapes).
                    '0'..='9' => return None,
                    _ => {}
                }
                out.push(c);
                out.push(e);
                prev_quantifier = false;
            }
            // Anchors make the piece depend on where the haystack ends.
            '^' | '$' => return None,
            '[' => {
                in_class = true;
                out.push(c);
                prev_quantifier = false;
            }
            '(' => {
                out.push(c);
                if chars.peek() == Some(&'?') {
                    out.push(chars.next()?);
                    match chars.peek() {
                        Some(':') => out.push(chars.next()?),
                        // Inline flags, e.g. `(?i:...)`.
                        Some(f) if f.is_ascii_alphabetic() || *f == '-' => {
                            while let Some(f) = chars.peek() {
                                if f.is_ascii_alphabetic() || *f == '-' {
                                    out.push(chars.next()?);
                                } else {
                                    break;
                                }
                            }
                            match chars.peek() {
                                Some(':') | Some(')') => out.push(chars.next()?),
                                _ => return None,
                            }
                        }
                        // Look-around, capture group names, conditionals, ...
                        _ => return None,
                    }
                }
                prev_quantifier = false;
            }
            '{' => {
                // A `{m,n}` repetition (only then is a following `+`
                // possessive); anything else is a literal brace.
                let mut rep = String::from('{');
                let mut valid = false;
                let mut digits = 0;
                loop {
                    match chars.peek() {
                        Some(d) if d.is_ascii_digit() => {
                            digits += 1;
                            rep.push(chars.next()?);
                        }
                        Some(',') if digits > 0 => rep.push(chars.next()?),
                        Some('}') if digits > 0 => {
                            rep.push(chars.next()?);
                            valid = true;
                            break;
                        }
                        _ => break,
                    }
                }
                out.push_str(&rep);
                prev_quantifier = valid;
            }
            '+' if prev_quantifier => prev_quantifier = false,
            '+' | '*' | '?' => {
                out.push(c);
                prev_quantifier = true;
            }
            _ => {
                out.push(c);
                prev_quantifier = false;
            }
        }
    }
    if in_class {
        return None;
    }
    Some(out)
}

/// Builds a DFA for the leading look-around-free alternatives of `pattern`, if
/// it can be shown to split text exactly like `fancy_regex` does.
fn build_split_dfa(pattern: &str, fancy: &Regex) -> Option<SplitDfa> {
    // `find_iter` passes `\G`/`\K` state between matches; leave such patterns
    // entirely alone.
    if pattern.contains("\\G") || pattern.contains("\\K") {
        return None;
    }
    let alternatives = top_level_alternatives(pattern)?;
    // The fast alternatives must be a *prefix* of the alternation: only then
    // does an ordered-alternation match of the DFA pick the same alternative
    // the full pattern picks.
    let mut plain = Vec::new();
    for alt in &alternatives {
        match plain_alternative(alt) {
            Some(rewritten) => plain.push(rewritten),
            None => break,
        }
    }
    if plain.is_empty() {
        return None;
    }
    let dfa = dense::Builder::new()
        .configure(
            dense::Config::new()
                .match_kind(MatchKind::LeftmostFirst)
                .start_kind(StartKind::Anchored)
                .dfa_size_limit(Some(SPLIT_DFA_SIZE_LIMIT))
                .determinize_size_limit(Some(SPLIT_DETERMINIZE_SIZE_LIMIT)),
        )
        .build(&plain.join("|"))
        .ok()?;
    let start = dfa.universal_start_state(Anchored::Yes)?;
    let split = SplitDfa { dfa, start };
    if !split.agrees_with(fancy) {
        return None;
    }
    Some(split)
}

impl SplitDfa {
    /// Checks that splitting with the DFA (falling back to `fancy_regex`)
    /// yields exactly the pieces `fancy_regex` yields on its own, over a corpus
    /// covering every kind of character the tokeniser patterns distinguish.
    fn agrees_with(&self, fancy: &Regex) -> bool {
        for probe in split_probes() {
            let mut fast = Vec::new();
            for piece in Pieces::new(&probe, self, fancy) {
                match piece {
                    Ok(piece) => fast.push(piece),
                    Err(_) => return false,
                }
            }
            let mut slow = Vec::new();
            for mat in fancy.find_iter(&probe) {
                match mat {
                    Ok(mat) => slow.push(mat.as_str()),
                    Err(_) => return false,
                }
            }
            if fast != slow {
                return false;
            }
        }
        true
    }
}

/// The corpus `SplitDfa::agrees_with` verifies against: every string of up to
/// three characters over a spread of ASCII and non-ASCII characters (letters,
/// digits, marks, spaces, line breaks, punctuation), every pair over a wider
/// set, plus a handful of longer strings.
fn split_probes() -> Vec<String> {
    const TRIPLES: [&str; 10] = ["a", "Z", "0", " ", "\n", "'", ",", "é", "中", "\t"];
    const PAIRS: [&str; 20] = [
        "a", "Z", "0", " ", "\n", "'", ",", "é", "中", "\t", "\r", "s", "9", "\u{a0}", "½", "Ⅷ",
        "０", "\u{301}", "!", "_",
    ];
    const LONGER: [&str; 24] = [
        "",
        " ",
        "  ",
        "   ",
        "hello world",
        "it's a test",
        "I'LL DO IT",
        "don't  stop",
        "123 4567 89",
        "a  b\n\nc",
        "\r\n\r\n",
        " \n \n ",
        "\t\t x",
        "trailing space ",
        "trailing newline\n",
        "café au lait",
        "中文字符测试",
        "🎉🎉 party",
        "a\u{301}e\u{301}",
        "０１２３",
        "½Ⅷ¾",
        "mixed 中文 and ASCII 123",
        "\u{a0}nbsp\u{a0}",
        "under_score-dash.dot!bang?",
    ];

    let mut probes = Vec::new();
    for a in PAIRS {
        probes.push(a.to_string());
        for b in PAIRS {
            probes.push(format!("{a}{b}"));
        }
    }
    for a in TRIPLES {
        for b in TRIPLES {
            for c in TRIPLES {
                probes.push(format!("{a}{b}{c}"));
            }
        }
    }
    probes.extend(LONGER.iter().map(|s| s.to_string()));
    probes
}

/// Returns the smallest index of a valid UTF-8 sequence starting after `i`.
/// Mirrors the equivalent helper in `fancy_regex`'s `Matches` iterator.
fn next_utf8(text: &str, i: usize) -> usize {
    match text.as_bytes().get(i) {
        None => i + 1,
        Some(&b) => {
            i + match b {
                0x00..=0x7F => 1,
                0x80..=0xBF => 1,
                0xC0..=0xDF => 2,
                0xE0..=0xEF => 3,
                _ => 4,
            }
        }
    }
}

/// Iterator over the pieces of `text`, taking each piece from the DFA when it
/// matches and from `fancy_regex` otherwise.
///
/// The piece stream is identical to `fancy_regex`'s `find_iter`, including its
/// handling of empty matches and of positions where the pattern matches only
/// further along.
struct Pieces<'a> {
    text: &'a str,
    split: &'a SplitDfa,
    fancy: &'a Regex,
    pos: usize,
    last_match: Option<usize>,
}

impl<'a> Pieces<'a> {
    fn new(text: &'a str, split: &'a SplitDfa, fancy: &'a Regex) -> Self {
        Pieces {
            text,
            split,
            fancy,
            pos: 0,
            last_match: None,
        }
    }
}

impl<'a> Iterator for Pieces<'a> {
    type Item = fancy_regex::Result<&'a str>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.pos > self.text.len() {
                return None;
            }
            if self.pos < self.text.len()
                && let Some(end) = self.split.piece_end(self.text, self.pos)
                && end > self.pos
            {
                let piece = &self.text[self.pos..end];
                self.pos = end;
                self.last_match = Some(end);
                return Some(Ok(piece));
            }
            // The DFA's alternatives don't match here: `fancy_regex` decides.
            let mat = match self.fancy.find_from_pos(self.text, self.pos) {
                Ok(Some(mat)) => mat,
                Ok(None) => {
                    self.pos = self.text.len() + 1;
                    return None;
                }
                Err(e) => {
                    self.pos = self.text.len() + 1;
                    return Some(Err(e));
                }
            };
            if mat.start() == mat.end() {
                self.pos = next_utf8(self.text, mat.end());
                let after_match = self.last_match == Some(mat.end());
                self.last_match = Some(mat.end());
                // Don't accept an empty match immediately following a match.
                if after_match {
                    continue;
                }
            } else {
                self.pos = mat.end();
                self.last_match = Some(mat.end());
            }
            return Some(Ok(mat.as_str()));
        }
    }
}

/// The pieces of `text`, from either splitter.
enum Split<'a> {
    Fast(Pieces<'a>),
    Fancy(fancy_regex::Matches<'a, 'a>),
}

impl<'a> Iterator for Split<'a> {
    type Item = fancy_regex::Result<&'a str>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Split::Fast(pieces) => pieces.next(),
            Split::Fancy(matches) => matches.next().map(|mat| mat.map(|mat| mat.as_str())),
        }
    }
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
    // Byte-level DFA for the split pattern's leading, look-around-free
    // alternatives, which produce nearly every piece of real text. It is
    // immutable and needs no scratch space, so unlike the `fancy_regex` copies
    // above a single instance serves every thread. `None` when the pattern
    // could not be shown to split identically (see `build_split_dfa`).
    split_dfa: Option<Arc<SplitDfa>>,
    sorted_token_bytes: Vec<Vec<u8>>,
}

impl CoreBPE {
    /// Iterator over the pieces of `text`, exactly as `fancy_regex`'s
    /// `find_iter` on the split pattern would produce them.
    fn split<'a>(&'a self, text: &'a str) -> Split<'a> {
        match &self.split_dfa {
            Some(split) => Split::Fast(Pieces::new(text, split, self._get_tl_regex())),
            None => Split::Fancy(self._get_tl_regex().find_iter(text)),
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
        let mut ret = vec![];
        for piece in self.split(text) {
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
            for piece in self.split(&text[start..end]) {
                let piece = match piece {
                    Ok(piece) => piece.as_bytes(),
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

        // Compile the DFA for the split pattern's look-around-free leading
        // alternatives. It is only adopted if it splits exactly like
        // `fancy_regex`, so an unusual pattern simply keeps today's behaviour.
        let split_dfa = build_split_dfa(pattern, &regex_tls[0]).map(Arc::new);

        Ok(Self {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            decoder_flat,
            regex_tls,
            special_regex_tls,
            split_dfa,
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

    #[test]
    fn test_top_level_alternatives() {
        assert_eq!(
            crate::top_level_alternatives(R50K_PAT).unwrap(),
            vec![
                r"'(?:[sdmt]|ll|ve|re)",
                r" ?\p{L}++",
                r" ?\p{N}++",
                r" ?[^\s\p{L}\p{N}]++",
                r"\s++$",
                r"\s+(?!\S)",
                r"\s",
            ]
        );
        // `|` inside a group or a character class is not a top-level `|`.
        assert_eq!(
            crate::top_level_alternatives(r"a[|]|(b|c)|\|").unwrap(),
            vec![r"a[|]", r"(b|c)", r"\|"]
        );
    }

    #[test]
    fn test_plain_alternative() {
        // Possessive markers are dropped, everything else is kept verbatim.
        assert_eq!(
            crate::plain_alternative(r" ?[^\s\p{L}\p{N}]++[\r\n]*+").unwrap(),
            r" ?[^\s\p{L}\p{N}]+[\r\n]*"
        );
        assert_eq!(
            crate::plain_alternative(r"\p{N}{1,3}+").unwrap(),
            r"\p{N}{1,3}"
        );
        assert_eq!(
            crate::plain_alternative(r"'(?i:[sdmt]|ll|ve|re)").unwrap(),
            r"'(?i:[sdmt]|ll|ve|re)"
        );
        // A `+` after a literal brace is a quantifier, not a possessive marker.
        assert_eq!(crate::plain_alternative(r"a}+").unwrap(), r"a}+");
        // Constructs that don't depend on the piece start alone are rejected.
        assert!(crate::plain_alternative(r"\s+(?!\S)").is_none());
        assert!(crate::plain_alternative(r"\s++$").is_none());
        assert!(crate::plain_alternative(r"^\s+").is_none());
        assert!(crate::plain_alternative(r"\bword").is_none());
        assert!(crate::plain_alternative(r"(a)\1").is_none());
        assert!(crate::plain_alternative(r"(?<=a)b").is_none());
    }

    #[test]
    fn test_split_dfa_matches_fancy_regex() {
        for pattern in [R50K_PAT, CL100K_PAT] {
            let fancy = Regex::new(pattern).unwrap();
            // Built at all (the leading alternatives are look-around-free) and
            // verified against `fancy_regex` over the probe corpus.
            let split = crate::build_split_dfa(pattern, &fancy).expect("fast path");
            // The leading alternatives are covered, the look-around ones are not.
            assert_eq!(split.piece_end(" the", 0), Some(4));
            assert_eq!(split.piece_end("  a", 0), None);
        }
    }

    #[test]
    fn test_piece_end_matches_regex_automata_search() {
        use regex_automata::{Anchored, Input, dfa::Automaton};

        for pattern in [R50K_PAT, CL100K_PAT] {
            let fancy = Regex::new(pattern).unwrap();
            let split = crate::build_split_dfa(pattern, &fancy).expect("fast path");
            for probe in crate::split_probes() {
                for at in 0..=probe.len() {
                    if !probe.is_char_boundary(at) {
                        continue;
                    }
                    let input = Input::new(probe.as_str())
                        .span(at..probe.len())
                        .anchored(Anchored::Yes);
                    let expected = match split.dfa.try_search_fwd(&input) {
                        Ok(m) => m.map(|m| m.offset()),
                        Err(_) => None,
                    };
                    assert_eq!(
                        split.piece_end(&probe, at),
                        expected,
                        "pattern {pattern:?}, probe {probe:?} at {at}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_split_dfa_rejects_unsupported_patterns() {
        // A pattern whose *first* alternative needs look-around gets no DFA.
        let pattern = r"\s+(?!\S)| ?\p{L}+";
        let fancy = Regex::new(pattern).unwrap();
        assert!(crate::build_split_dfa(pattern, &fancy).is_none());
        // Neither does one using `\G`, which `find_iter` treats specially.
        let pattern = r"\G ?\p{L}+|\s";
        let fancy = Regex::new(pattern).unwrap();
        assert!(crate::build_split_dfa(pattern, &fancy).is_none());
    }
}
