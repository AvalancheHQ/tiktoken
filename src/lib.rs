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

/// The leading alternatives of a tokeniser split pattern that can only ever
/// match at one specific literal byte (e.g. cl100k's `'(?i:[sdmt]|ll|ve|re)`,
/// which requires an apostrophe).
///
/// `fancy_regex` runs the split on its backtracking VM, and the VM tries every
/// alternative in order at every piece start. Those leading alternatives
/// therefore cost a delegate automaton call per piece even though the byte at
/// the piece start already rules them out. Keeping them in a separate regex
/// lets the hot path run the remaining alternatives only, and consult this one
/// just at the positions where it can actually match.
#[derive(Clone)]
struct LiteralPrefixAlts {
    /// The literal bytes those alternatives must start with.
    guards: Vec<u8>,
    /// One compiled copy per thread-local slot, see `regex_tls`.
    regex_tls: Vec<Regex>,
}

/// Splits `pattern` into its top-level alternatives, i.e. the `|`-separated
/// branches that are not inside a group or a character class.
fn top_level_alternatives(pattern: &str) -> Option<Vec<&str>> {
    let bytes = pattern.as_bytes();
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut depth: i32 = 0;
    let mut in_class = false;
    let mut class_start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                if i + 1 >= bytes.len() {
                    return None;
                }
                i += 2;
                continue;
            }
            b'[' if !in_class => {
                in_class = true;
                class_start = i;
            }
            b']' if in_class => {
                // A `]` right after `[` or `[^` is a literal, not the class end.
                let literal = i == class_start + 1
                    || (i == class_start + 2 && bytes[class_start + 1] == b'^');
                if !literal {
                    in_class = false;
                }
            }
            b'(' if !in_class => depth += 1,
            b')' if !in_class => {
                depth -= 1;
                if depth < 0 {
                    return None;
                }
            }
            b'|' if !in_class && depth == 0 => {
                parts.push(&pattern[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if depth != 0 || in_class {
        return None;
    }
    parts.push(&pattern[start..]);
    Some(parts)
}

/// The byte an alternative must start with, if it always starts with the same
/// plain literal one.
///
/// Only non-alphanumeric ASCII literals are accepted, so a case-insensitivity
/// flag elsewhere in the pattern cannot make the alternative match a different
/// byte, and the byte can be compared directly against UTF-8 input.
fn mandatory_literal_prefix(alt: &str) -> Option<u8> {
    let bytes = alt.as_bytes();
    let first = *bytes.first()?;
    if !first.is_ascii()
        || first.is_ascii_alphanumeric()
        || first.is_ascii_whitespace()
        || first.is_ascii_control()
        || b"\\^$.[]()*+?{}|".contains(&first)
    {
        return None;
    }
    // The literal must be mandatory: a quantifier that allows zero repetitions
    // would let the alternative match at a different byte.
    match bytes.get(1) {
        Some(b'?') | Some(b'*') | Some(b'{') => None,
        _ => Some(first),
    }
}

/// Whether an alternative can be lifted out of the pattern verbatim: it must
/// not open a capture group (which would renumber the groups of the
/// alternatives that stay behind) nor set flags that would leak into them.
fn alternative_is_self_contained(alt: &str) -> bool {
    let bytes = alt.as_bytes();
    let mut in_class = false;
    let mut class_start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => {
                if i + 1 >= bytes.len() {
                    return false;
                }
                i += 2;
                continue;
            }
            b'[' if !in_class => {
                in_class = true;
                class_start = i;
            }
            b']' if in_class => {
                let literal = i == class_start + 1
                    || (i == class_start + 2 && bytes[class_start + 1] == b'^');
                if !literal {
                    in_class = false;
                }
            }
            b'(' if !in_class => {
                if bytes.get(i + 1) != Some(&b'?') {
                    return false; // capture group
                }
                // Accept only `(?flags:` groups: anything reaching `)` first is
                // a flag setter or a lookaround, which we do not reason about.
                let mut j = i + 2;
                while j < bytes.len() && bytes[j] != b':' && bytes[j] != b')' {
                    j += 1;
                }
                if j >= bytes.len() || bytes[j] == b')' {
                    return false;
                }
            }
            _ => {}
        }
        i += 1;
    }
    true
}

/// Peels the leading alternatives of a split pattern that can only match at a
/// fixed literal byte.
///
/// Returns `(peeled_pattern, remaining_pattern, guard_bytes)`. Because
/// alternation is ordered, an alternative that cannot match at a position never
/// influences the match there, so running `remaining_pattern` alone at
/// positions whose byte is not a guard yields exactly the same pieces.
fn peel_literal_prefix_alternatives(pattern: &str) -> Option<(String, String, Vec<u8>)> {
    // `\G`/`\K` depend on where the search started and backreferences depend on
    // group numbering; both would be affected by splitting the pattern.
    if pattern.contains("\\G") || pattern.contains("\\K") || pattern.contains("\\k") {
        return None;
    }
    if pattern
        .as_bytes()
        .windows(2)
        .any(|w| w[0] == b'\\' && w[1].is_ascii_digit())
    {
        return None;
    }

    let alts = top_level_alternatives(pattern)?;
    let mut guards: Vec<u8> = Vec::new();
    let mut peeled = 0usize;
    for alt in &alts {
        match mandatory_literal_prefix(alt) {
            Some(byte) if alternative_is_self_contained(alt) => {
                if !guards.contains(&byte) {
                    guards.push(byte);
                }
                peeled += 1;
            }
            _ => break,
        }
    }
    if peeled == 0 || peeled == alts.len() {
        return None;
    }
    Some((alts[..peeled].join("|"), alts[peeled..].join("|"), guards))
}

/// Iterates over the tokeniser split pieces of `text`, mirroring
/// `fancy_regex::Regex::find_iter` while skipping the alternatives that cannot
/// match at the current position.
struct Pieces<'t, 'c> {
    text: &'t str,
    regex: &'c Regex,
    prefix: Option<(&'c [u8], &'c Regex)>,
    last_end: usize,
    last_match: Option<usize>,
}

impl<'t> Pieces<'t, '_> {
    /// Position of the next byte in `bytes[from..to]` that one of the peeled
    /// alternatives could start at.
    #[inline]
    fn next_guard(bytes: &[u8], from: usize, to: usize, guards: &[u8]) -> Option<usize> {
        let to = to.min(bytes.len());
        if from >= to {
            return None;
        }
        bytes[from..to]
            .iter()
            .position(|b| guards.contains(b))
            .map(|i| from + i)
    }

    fn find_from(&self, from: usize) -> fancy_regex::Result<Option<(usize, usize)>> {
        let Some((guards, prefix_regex)) = self.prefix else {
            return Ok(self
                .regex
                .find_from_pos(self.text, from)?
                .map(|m| (m.start(), m.end())));
        };

        let bytes = self.text.as_bytes();
        let mut pos = from;
        loop {
            // The peeled alternatives come first in the pattern, so they win at
            // any position where they match.
            if bytes.get(pos).is_some_and(|b| guards.contains(b))
                && let Some(m) = prefix_regex.find_from_pos(self.text, pos)?
                && m.start() == pos
            {
                return Ok(Some((m.start(), m.end())));
            }
            match self.regex.find_from_pos(self.text, pos)? {
                // No match at or after `pos` for the remaining alternatives, but
                // a peeled one may still match at a later guard byte.
                None => match Self::next_guard(bytes, pos + 1, bytes.len(), guards) {
                    Some(next) => pos = next,
                    None => return Ok(None),
                },
                Some(m) => {
                    // A peeled alternative starting anywhere up to and including
                    // the match start would have matched earlier (or won there).
                    match Self::next_guard(bytes, pos + 1, m.start() + 1, guards) {
                        Some(next) => pos = next,
                        None => return Ok(Some((m.start(), m.end()))),
                    }
                }
            }
        }
    }
}

impl<'t> Iterator for Pieces<'t, '_> {
    type Item = fancy_regex::Result<&'t str>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.last_end > self.text.len() {
                return None;
            }
            let (start, end) = match self.find_from(self.last_end) {
                Ok(Some(m)) => m,
                Ok(None) => return None,
                Err(error) => {
                    // Stop on the first error, like `fancy_regex::Matches`.
                    self.last_end = self.text.len() + 1;
                    return Some(Err(error));
                }
            };

            if start == end {
                // Empty match: make progress, and skip one that immediately
                // follows the previous match.
                self.last_end = end + self.text[end..].chars().next().map_or(1, char::len_utf8);
                if Some(end) == self.last_match {
                    continue;
                }
            } else {
                self.last_end = end;
            }
            self.last_match = Some(end);
            return Some(Ok(&self.text[start..end]));
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
    // Holds the split pattern minus the alternatives peeled into
    // `prefix_alts` (the pattern verbatim when nothing was peeled).
    regex_tls: Vec<Regex>,
    prefix_alts: Option<LiteralPrefixAlts>,
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

    /// Iterates over the tokeniser split pieces of `text`, exactly as
    /// `self._get_tl_regex().find_iter(text)` would.
    fn _split_pieces<'t>(&self, text: &'t str) -> Pieces<'t, '_> {
        let slot = hash_current_thread() % MAX_NUM_THREADS;
        Pieces {
            text,
            regex: &self.regex_tls[slot],
            prefix: self
                .prefix_alts
                .as_ref()
                .map(|alts| (alts.guards.as_slice(), &alts.regex_tls[slot])),
            last_end: 0,
            last_match: None,
        }
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
        for mat in self._split_pieces(text) {
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
            for mat_res in self._split_pieces(&text[start..end]) {
                let mat = match mat_res {
                    Ok(m) => m,
                    Err(e) => {
                        return Err(EncodeError {
                            message: format!("Regex error while tokenizing: {e}"),
                        });
                    }
                };

                let piece = mat.as_bytes();
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
        // The leading alternatives that can only match at a fixed literal byte
        // are compiled separately, so the hot path does not ask the backtracking
        // VM about them at every piece start. Anything that cannot be peeled
        // (including any pattern we do not fully understand) keeps the pattern
        // verbatim, and a peeled pattern that fails to compile falls back too.
        let peeled =
            peel_literal_prefix_alternatives(pattern).and_then(|(prefix, rest, guards)| {
                let rest_tls = (0..MAX_NUM_THREADS)
                    .map(|_| Regex::new(&rest))
                    .collect::<Result<Vec<_>, _>>()
                    .ok()?;
                let prefix_tls = (0..MAX_NUM_THREADS)
                    .map(|_| Regex::new(&prefix))
                    .collect::<Result<Vec<_>, _>>()
                    .ok()?;
                Some((
                    rest_tls,
                    LiteralPrefixAlts {
                        guards,
                        regex_tls: prefix_tls,
                    },
                ))
            });

        let (regex_tls, prefix_alts) = match peeled {
            Some((rest_tls, alts)) => {
                // Compile the pattern as written once, so an invalid pattern is
                // still rejected here exactly as before.
                Regex::new(pattern)?;
                (rest_tls, Some(alts))
            }
            None => (
                (0..MAX_NUM_THREADS)
                    .map(|_| Regex::new(pattern))
                    .collect::<Result<Vec<_>, _>>()?,
                None,
            ),
        };
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
            prefix_alts,
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
    const O200K_PAT: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    /// The split must yield exactly what `fancy_regex::find_iter` yields.
    fn assert_same_pieces(pattern: &str, texts: &[&str]) {
        let bpe = crate::CoreBPE::new_internal(
            HashMap::from_iter([(b"a".to_vec(), 0)]),
            HashMap::default(),
            pattern,
        )
        .unwrap();
        let reference = Regex::new(pattern).unwrap();
        for text in texts {
            let expected: Vec<&str> = reference
                .find_iter(text)
                .map(|m| m.unwrap().as_str())
                .collect();
            let actual: Vec<&str> = bpe._split_pieces(text).map(|p| p.unwrap()).collect();
            assert_eq!(actual, expected, "pattern {pattern:?} on text {text:?}");
        }
    }

    #[test]
    fn test_split_matches_fancy_regex() {
        let texts = [
            "",
            "'",
            "'s",
            "'S",
            "don't stop",
            "it's a 'quoted' word",
            "hello world",
            "  leading and trailing  ",
            "line one\nline two\r\n\r\n",
            "123 4567 89",
            "a'b'c''d",
            "punctuation!!! ...and 'more'",
            "héllo wörld — naïve 'test'",
            "日本語のテキスト 'quote' 123",
            "emoji 🙂🙃 and 'contractions' aren't rare",
            "\t\t'tab'\u{a0}nbsp",
            "'''",
            "'ll 've 'd 'm 't 's",
        ];
        for pattern in [R50K_PAT, CL100K_PAT, O200K_PAT] {
            assert_same_pieces(pattern, &texts);
        }
    }

    #[test]
    fn test_peeling_is_conservative() {
        // The two shipped patterns peel their contraction alternative.
        for pattern in [R50K_PAT, CL100K_PAT] {
            let (prefix, rest, guards) = super::peel_literal_prefix_alternatives(pattern).unwrap();
            assert_eq!(guards, vec![b'\'']);
            assert!(prefix.starts_with('\''));
            assert!(!rest.contains("[sdmt]"));
        }
        // o200k starts with a character class, so nothing is peeled.
        assert!(super::peel_literal_prefix_alternatives(O200K_PAT).is_none());
        // Neither are patterns with capture groups, flag setters or `\G`.
        assert!(super::peel_literal_prefix_alternatives(r"'(a)|\p{L}+").is_none());
        assert!(super::peel_literal_prefix_alternatives(r"'(?i)a|\p{L}+").is_none());
        assert!(super::peel_literal_prefix_alternatives(r"\G'a|\p{L}+").is_none());
        assert!(super::peel_literal_prefix_alternatives(r"'a").is_none());
        // An optional leading literal is not a guarantee.
        assert!(super::peel_literal_prefix_alternatives(r"'?a|\p{L}+").is_none());
    }
}
