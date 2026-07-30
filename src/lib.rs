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

/// The tokeniser split patterns tiktoken ships, recognised verbatim.
///
/// Splitting is ~85% of encoding time: both patterns use possessive quantifiers
/// and the `\s+(?!\S)` lookahead, so `fancy_regex` runs them on its backtracking
/// VM — one VM run, with its own allocations and delegate searches, per token.
/// Knowing which pattern we are splitting with lets [`fast_piece_end`] decide
/// the common pieces with a plain byte scan instead.
///
/// `r50k_base`/`p50k_base`/`p50k_edit`/`gpt2` share `R50K_PATTERN`.
const R50K_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";
/// `cl100k_base`.
const CL100K_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";

/// Which split pattern a `CoreBPE` was built with, if it is one we recognise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PatternFamily {
    /// Not a pattern we know: no fast path, `fancy_regex` does all the work.
    Unknown,
    R50k,
    Cl100k,
}

impl PatternFamily {
    fn of(pattern: &str) -> Self {
        if pattern == R50K_PATTERN {
            PatternFamily::R50k
        } else if pattern == CL100K_PATTERN {
            PatternFamily::Cl100k
        } else {
            PatternFamily::Unknown
        }
    }
}

/// True for the ASCII characters `\s` matches (`\s` is `\p{White_Space}`; every
/// other member is non-ASCII).
#[inline]
const fn is_ascii_space(b: u8) -> bool {
    matches!(b, b'\t' | b'\n' | 0x0b | 0x0c | b'\r' | b' ')
}

/// True for the ASCII characters in `[^\s\p{L}\p{N}]`.
#[inline]
const fn is_ascii_symbol(b: u8) -> bool {
    b.is_ascii() && !b.is_ascii_alphanumeric() && !is_ascii_space(b)
}

/// End of the maximal ASCII letter run at `i`, or `None` if it is cut short by a
/// non-ASCII byte that could extend it (a letter, or a mark for some patterns).
#[inline]
fn ascii_letter_run(bytes: &[u8], mut i: usize) -> Option<usize> {
    while i < bytes.len() && bytes[i].is_ascii_alphabetic() {
        i += 1;
    }
    (i >= bytes.len() || bytes[i].is_ascii()).then_some(i)
}

/// End of the maximal ASCII digit run at `i`, capped at `max_len` digits, or
/// `None` if a shorter-than-cap run is cut short by a non-ASCII byte that could
/// be another `\p{N}`.
#[inline]
fn ascii_digit_run(bytes: &[u8], start: usize, max_len: usize) -> Option<usize> {
    let mut i = start;
    while i < bytes.len() && i - start < max_len && bytes[i].is_ascii_digit() {
        i += 1;
    }
    (i - start == max_len || i >= bytes.len() || bytes[i].is_ascii()).then_some(i)
}

/// End of the maximal `[^\s\p{L}\p{N}]` run at `i` followed by `[\r\n]*` when
/// `line_breaks` is set, or `None` if the run is cut short by a non-ASCII byte
/// that could belong to the class.
#[inline]
fn ascii_symbol_run(bytes: &[u8], mut i: usize, line_breaks: bool) -> Option<usize> {
    while i < bytes.len() && is_ascii_symbol(bytes[i]) {
        i += 1;
    }
    if i < bytes.len() && !bytes[i].is_ascii() {
        return None;
    }
    if line_breaks {
        while i < bytes.len() && matches!(bytes[i], b'\r' | b'\n') {
            i += 1;
        }
    }
    Some(i)
}

/// End offset of the piece starting at `start`, when it can be determined
/// without running the regex, or `None` to let `fancy_regex` decide.
///
/// This resolves the alternation by hand for the pieces that dominate real
/// text — letter runs, digit runs and symbol runs, each optionally preceded by
/// the character the pattern's alternative absorbs — and gives up on everything
/// else, so `fancy_regex` remains the source of truth. Both families are read
/// alternative by alternative, in the order the VM tries them:
///
/// * `R50K_PATTERN`: `'(?:[sdmt]|ll|ve|re)` needs a `'`, so a piece that does
///   not start with one is decided by ` ?\p{L}++`, ` ?\p{N}++` or
///   ` ?[^\s\p{L}\p{N}]++`, each an atomic maximal run with an optional leading
///   space. A whitespace-led piece other than `space + non-whitespace` is left
///   to the regex (`\s++$`, `\s+(?!\S)`, `\s`).
/// * `CL100K_PATTERN`: same shape, except that the letter alternative absorbs
///   any non-line-break non-alphanumeric character as a prefix
///   (`[^\r\n\p{L}\p{N}]?+\p{L}++`), digits are capped at three
///   (`\p{N}{1,3}+`), the symbol alternative also swallows trailing line breaks
///   (` ?[^\s\p{L}\p{N}]++[\r\n]*+`), and `space + digit` matches the trailing
///   bare `\s` (a single space) because every alternative in between fails.
///
/// Anything involving a non-ASCII byte bails out: whether such a byte extends a
/// run is a question about Unicode character classes, which is the regex's job.
/// A leading `'` also bails, since the contraction alternative can match just
/// part of a letter run (`'ss` -> `'s`).
#[inline]
fn fast_piece_end(family: PatternFamily, bytes: &[u8], start: usize) -> Option<usize> {
    let first = *bytes.get(start)?;
    if first.is_ascii_alphabetic() {
        return ascii_letter_run(bytes, start);
    }
    if !first.is_ascii() || first == b'\'' {
        return None;
    }
    let cl100k = match family {
        PatternFamily::Cl100k => true,
        PatternFamily::R50k => false,
        PatternFamily::Unknown => return None,
    };

    if first.is_ascii_digit() {
        return ascii_digit_run(bytes, start, if cl100k { 3 } else { usize::MAX });
    }

    // A single character can be absorbed as the prefix of the piece that starts
    // at `start + 1`: a space for r50k, anything but a line break for cl100k.
    let prefix = if cl100k {
        !matches!(first, b'\r' | b'\n')
    } else {
        first == b' '
    };
    // When nothing follows the prefix every alternative that could absorb it
    // fails, so fall through to the unprefixed handling below.
    if let (true, Some(&next)) = (prefix, bytes.get(start + 1)) {
        if !next.is_ascii() {
            return None;
        }
        if next.is_ascii_alphabetic() {
            return ascii_letter_run(bytes, start + 1);
        }
        if first == b' ' {
            if next.is_ascii_digit() {
                return if cl100k {
                    // ` ?[^\s\p{L}\p{N}]++…` cannot start with a digit and the
                    // whitespace alternatives all need more whitespace or the
                    // end of the text, so the trailing bare `\s` wins.
                    Some(start + 1)
                } else {
                    ascii_digit_run(bytes, start + 1, usize::MAX)
                };
            }
            if !is_ascii_space(next) {
                return ascii_symbol_run(bytes, start + 1, cl100k);
            }
        }
    }
    if is_ascii_space(first) {
        // Whitespace runs are the regex's business.
        return None;
    }
    ascii_symbol_run(bytes, start, cl100k)
}

/// The pieces of `text`, as `regex.find_iter(text)` would yield them.
///
/// For a recognised [`PatternFamily`] the common pieces are recognised by
/// [`fast_piece_end`] and every other position is handed to `fancy_regex`,
/// which stays the source of truth. For an unrecognised pattern this is
/// `find_iter` verbatim.
enum Pieces<'a> {
    Fancy(fancy_regex::Matches<'a, 'a>),
    Fast {
        text: &'a str,
        regex: &'a Regex,
        family: PatternFamily,
        pos: usize,
    },
}

impl<'a> Pieces<'a> {
    fn new(regex: &'a Regex, family: PatternFamily, text: &'a str) -> Self {
        match family {
            PatternFamily::Unknown => Pieces::Fancy(regex.find_iter(text)),
            _ => Pieces::Fast {
                text,
                regex,
                family,
                pos: 0,
            },
        }
    }
}

impl<'a> Iterator for Pieces<'a> {
    type Item = fancy_regex::Result<&'a str>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Pieces::Fancy(matches) => matches.next().map(|m| m.map(|m| m.as_str())),
            Pieces::Fast {
                text,
                regex,
                family,
                pos,
            } => {
                if *pos >= text.len() {
                    return None;
                }
                if let Some(end) = fast_piece_end(*family, text.as_bytes(), *pos) {
                    let piece = &text[*pos..end];
                    *pos = end;
                    return Some(Ok(piece));
                }
                match regex.find_from_pos(text, *pos) {
                    Ok(Some(m)) => {
                        // Every alternative of the recognised patterns consumes
                        // at least one character, but stay safe against a
                        // zero-width match spinning forever.
                        *pos = if m.end() > m.start() {
                            m.end()
                        } else {
                            m.end() + codepoint_len_at(text, m.end())
                        };
                        Some(Ok(m.as_str()))
                    }
                    Ok(None) => {
                        *pos = text.len();
                        None
                    }
                    Err(e) => {
                        *pos = text.len();
                        Some(Err(e))
                    }
                }
            }
        }
    }
}

/// Length in bytes of the UTF-8 code point starting at `ix`, or 1 at the end of
/// the string (only used to make forward progress).
#[inline]
fn codepoint_len_at(text: &str, ix: usize) -> usize {
    text[ix..].chars().next().map_or(1, char::len_utf8)
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
    // Which of the shipped split patterns this tokeniser uses, if any. Lets the
    // splitter recognise the common pieces with a byte scan instead of a
    // `fancy_regex` VM run; see `Pieces` and `fast_piece_end`.
    pattern_family: PatternFamily,
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
        for piece in Pieces::new(regex, self.pattern_family, text) {
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
            for piece_res in Pieces::new(regex, self.pattern_family, &text[start..end]) {
                let piece = match piece_res {
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

        Ok(Self {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            decoder_flat,
            regex_tls,
            special_regex_tls,
            sorted_token_bytes,
            pattern_family: PatternFamily::of(pattern),
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

    use crate::{CL100K_PATTERN, PatternFamily, Pieces, R50K_PATTERN, Rank, byte_pair_split};

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

    /// Pieces as `fancy_regex` alone would produce them.
    fn fancy_pieces<'a>(regex: &Regex, text: &'a str) -> Vec<&'a str> {
        regex.find_iter(text).map(|m| m.unwrap().as_str()).collect()
    }

    /// Pieces as the fast splitter produces them.
    fn fast_pieces<'a>(regex: &'a Regex, family: PatternFamily, text: &'a str) -> Vec<&'a str> {
        Pieces::new(regex, family, text)
            .map(|piece| piece.unwrap())
            .collect()
    }

    fn assert_same_pieces(pattern: &str, texts: &[String]) {
        let regex = Regex::new(pattern).unwrap();
        let family = PatternFamily::of(pattern);
        assert_ne!(family, PatternFamily::Unknown, "pattern not recognised");
        for text in texts {
            assert_eq!(
                fast_pieces(&regex, family, text),
                fancy_pieces(&regex, text),
                "split mismatch for {text:?} with pattern {pattern}"
            );
        }
    }

    fn tricky_texts() -> Vec<String> {
        [
            "",
            " ",
            "  ",
            "   word",
            "hello world",
            "The quick brown fox jumps over the lazy dog.",
            "don't 'ss 's 'sss 'll 'LL 'x '",
            "(BPE) is a way\tof.Net converting\r\ntext",
            "a1b2c3 123 1234 12345 007",
            "trailing spaces   ",
            "\n\n\n",
            " \n \r\n\t\u{b}\u{c}",
            "caf\u{e9} na\u{ef}ve \u{e9}t\u{e9}",
            "e\u{301}mile", // letter + combining mark
            "\u{4f60}\u{597d}\u{4e16}\u{754c} hello",
            "emoji \u{1f600}\u{1f469}\u{200d}\u{1f4bb} tail",
            "\u{a0}nbsp \u{2000}space",
            "MiXeD CaSe HELLO Hello",
            "under_score kebab-case snake_case",
            "\"quoted\" (parens) [brackets] {braces}",
            "http://example.com/path?q=1&r=2",
            "tabs\tbetween\twords",
            "\u{2019}s curly quote",
            "a",
            " a",
            ".a",
            "'a",
            "\ra",
            "\na",
            "1a",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Deterministic pseudo-random strings over an alphabet that stresses every
    /// character class the patterns care about.
    fn random_texts(count: usize, max_len: usize) -> Vec<String> {
        const ALPHABET: &[char] = &[
            'a',
            'b',
            'Z',
            'q',
            '0',
            '7',
            ' ',
            ' ',
            '\t',
            '\n',
            '\r',
            '.',
            ',',
            '\'',
            '-',
            '(',
            '/',
            '"',
            '_',
            '9',
            ' ',
            '\u{b}',
            '\u{c}',
            '\0',
            '\u{e9}',
            '\u{4f60}',
            '\u{301}',
            '\u{1f600}',
            '\u{a0}',
            '\u{2019}',
            '\u{660}',
            '\u{2160}',
        ];
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = || {
            // xorshift64*
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d)
        };
        (0..count)
            .map(|_| {
                let len = (next() as usize) % (max_len + 1);
                (0..len)
                    .map(|_| ALPHABET[(next() as usize) % ALPHABET.len()])
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_fast_split_matches_fancy_regex() {
        let texts = tricky_texts();
        assert_same_pieces(R50K_PATTERN, &texts);
        assert_same_pieces(CL100K_PATTERN, &texts);

        let random = random_texts(3000, 24);
        assert_same_pieces(R50K_PATTERN, &random);
        assert_same_pieces(CL100K_PATTERN, &random);
    }

    #[test]
    fn test_unknown_pattern_has_no_fast_path() {
        // o200k's letter alternative absorbs a trailing contraction, so it must
        // not be treated as a known family.
        let o200k = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}|\s+";
        assert_eq!(PatternFamily::of(o200k), PatternFamily::Unknown);
        let regex = Regex::new(o200k).unwrap();
        let text = "it's a test";
        assert_eq!(
            fast_pieces(&regex, PatternFamily::Unknown, text),
            fancy_pieces(&regex, text)
        );
    }
}
