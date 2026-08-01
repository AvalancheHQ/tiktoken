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

// Splitting the text with the tokeniser pattern dominates every encode call:
// `fancy_regex` runs its backtracking VM once per piece, and a piece is only a
// handful of bytes. Real text keeps presenting the *same short contexts* over
// and over (" the", " and", ", ", " of", …), so the same VM run is repeated
// thousands of times for the same bytes.
//
// The cache below memoises that decision: it maps the bytes at a piece start to
// the length of the piece the regex produces there. On a hit the whole VM run
// disappears and the piece is taken with a single table lookup.
//
// Reusing a decision at another position is only sound when the piece really is
// a function of the bytes in the window, so an entry is only created when:
//   * the window is entirely ASCII — then every byte is a whole character, so a
//     piece ending inside the window has the context a one-character lookahead
//     would read inside the window too, and
//   * the piece ends at least two bytes before the end of the window.
// Two window sizes are kept, because pieces range from single punctuation marks
// to whole words: 8 bytes for pieces up to 6 bytes (the common case, and the
// most reusable key), and 16 bytes for the longer pieces that the short window
// cannot decide.
//
// Every stored entry comes from a real match against the haystack being split,
// and a pattern may only use the cache at all if it passes the locality checks
// in `split_cache_id_for`.
const SHORT_WINDOW: usize = 8;
const MAX_SHORT_PIECE: usize = SHORT_WINDOW - 2;
const LONG_WINDOW: usize = 16;
const MAX_LONG_PIECE: usize = LONG_WINDOW - 2;
const SPLIT_CACHE_LOG: u32 = 12;
const SPLIT_CACHE_SIZE: usize = 1 << SPLIT_CACHE_LOG;
/// High bit of every byte; zero iff every byte of the window is ASCII.
const NON_ASCII_MASK_64: u64 = 0x8080_8080_8080_8080;
const NON_ASCII_MASK_128: u128 = 0x8080_8080_8080_8080_8080_8080_8080_8080;

/// Direct-mapped table of split decisions, keyed on a window of input bytes.
struct SplitTable<K: Copy + PartialEq> {
    keys: Box<[K; SPLIT_CACHE_SIZE]>,
    /// Identity of the `CoreBPE` an entry belongs to, so that several
    /// tokenisers alive in the same process never read each other's entries.
    owners: Box<[u32; SPLIT_CACHE_SIZE]>,
    /// Piece length in bytes; 0 means "empty slot".
    lens: Box<[u8; SPLIT_CACHE_SIZE]>,
}

impl<K: Copy + PartialEq + Default> SplitTable<K> {
    fn new() -> Self {
        SplitTable {
            keys: Box::new([K::default(); SPLIT_CACHE_SIZE]),
            owners: Box::new([0; SPLIT_CACHE_SIZE]),
            lens: Box::new([0; SPLIT_CACHE_SIZE]),
        }
    }

    #[inline(always)]
    fn get(&self, key: K, slot: usize, id: u32) -> Option<usize> {
        let len = self.lens[slot];
        if len != 0 && self.keys[slot] == key && self.owners[slot] == id {
            Some(len as usize)
        } else {
            None
        }
    }

    #[inline(always)]
    fn put(&mut self, key: K, slot: usize, id: u32, len: usize) {
        self.keys[slot] = key;
        self.owners[slot] = id;
        self.lens[slot] = len as u8;
    }
}

/// Thread-local cache of split decisions.
///
/// It is thread-local (rather than shared behind a lock) on purpose: the
/// tokeniser is used from several Python threads at once and the performance
/// notes above are explicit that a shared, lock-protected cache is what made
/// the original implementation slow.
struct SplitCache {
    short: SplitTable<u64>,
    long: SplitTable<u128>,
}

impl SplitCache {
    fn new() -> Self {
        SplitCache {
            short: SplitTable::new(),
            long: SplitTable::new(),
        }
    }
}

#[inline(always)]
fn slot_of(hash: u64) -> usize {
    (hash.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - SPLIT_CACHE_LOG)) as usize
}

thread_local! {
    static SPLIT_CACHE: std::cell::RefCell<SplitCache> =
        std::cell::RefCell::new(SplitCache::new());
}

/// Index of the next byte that starts a character, mirroring what
/// `fancy_regex`'s match iterator does after an empty match.
#[inline]
fn next_char_boundary(bytes: &[u8], mut i: usize) -> usize {
    i += 1;
    while i < bytes.len() && (bytes[i] & 0xC0) == 0x80 {
        i += 1;
    }
    i
}

/// The 8-byte window at `pos`, if it exists and is all ASCII.
#[inline(always)]
fn short_window(bytes: &[u8], pos: usize) -> Option<u64> {
    let window = bytes.get(pos..pos + SHORT_WINDOW)?;
    let key = u64::from_le_bytes(window.try_into().unwrap());
    (key & NON_ASCII_MASK_64 == 0).then_some(key)
}

/// The 16-byte window at `pos`, if it exists and is all ASCII.
#[inline(always)]
fn long_window(bytes: &[u8], pos: usize) -> Option<u128> {
    let window = bytes.get(pos..pos + LONG_WINDOW)?;
    let key = u128::from_le_bytes(window.try_into().unwrap());
    (key & NON_ASCII_MASK_128 == 0).then_some(key)
}

static NEXT_SPLIT_CACHE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

/// Decides whether split decisions for `pattern` may be cached.
///
/// Two things have to hold for a cached decision to be reusable at another
/// position: the pattern must not look at anything *before* the piece start
/// (no lookbehind, no `^`/`\A`/`\b`, no `\G`, no back-references), and a short
/// match must be decided by the bytes inside the window. The first is a
/// structural property of the parsed pattern; the second is checked directly
/// against the compiled regex by `verify_split_locality`.
fn split_cache_id_for(pattern: &str, regex: &Regex) -> Option<u32> {
    let tree = fancy_regex::Expr::parse_tree(pattern).ok()?;
    if !has_only_local_constructs(&tree.expr) {
        return None;
    }
    if !verify_split_locality(regex) {
        return None;
    }
    Some(NEXT_SPLIT_CACHE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}

fn has_only_local_constructs(expr: &fancy_regex::Expr) -> bool {
    use fancy_regex::{Assertion, Expr, LookAround};
    match expr {
        Expr::Assertion(assertion) => matches!(
            assertion,
            // End-of-text/line assertions are fine: a match that relies on one
            // reaches the end of the haystack, and such a match is far longer
            // than the pieces we are allowed to cache.
            Assertion::EndText | Assertion::EndLine { .. }
        ),
        Expr::LookAround(child, kind) => {
            matches!(kind, LookAround::LookAhead | LookAround::LookAheadNeg)
                && has_only_local_constructs(child)
        }
        Expr::Concat(children) | Expr::Alt(children) => {
            children.iter().all(has_only_local_constructs)
        }
        Expr::Group(child) | Expr::AtomicGroup(child) => has_only_local_constructs(child),
        Expr::Repeat { child, .. } => has_only_local_constructs(child),
        Expr::Empty | Expr::Any { .. } | Expr::Literal { .. } | Expr::Delegate { .. } => true,
        // Anything that can depend on preceding text or on other groups
        // (`\G`, `\K`, back-references, conditionals, subroutines, …).
        _ => false,
    }
}

/// Checks on sample text that a cacheable match is fully determined by the
/// window at its start: the same window followed by different text must yield
/// the same piece, for both window sizes.
fn verify_split_locality(regex: &Regex) -> bool {
    const ALPHABET: [&str; 26] = [
        "a",
        "b",
        "z",
        "A",
        "Q",
        "0",
        "7",
        " ",
        "  ",
        "\t",
        "\n",
        "\r\n",
        ".",
        ",",
        "'",
        "!",
        "-",
        "_",
        "<",
        "|",
        ">",
        "é",
        "中",
        "🙂",
        "word",
        "loremipsum",
    ];
    const TAILS: [&str; 8] = ["", "a", " ", "\n", "0", ".", "  \n", "aaaaaaaa"];

    let mut rng: u64 = 0x243F_6A88_85A3_08D3;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    let mut text = String::with_capacity(64);
    for _ in 0..200 {
        text.clear();
        let parts = 6 + (next() % 8) as usize;
        for _ in 0..parts {
            text.push_str(ALPHABET[(next() % ALPHABET.len() as u64) as usize]);
        }
        let bytes = text.as_bytes();
        let mut pos = 0;
        while pos < bytes.len() {
            let mat = match regex.find_from_pos(&text, pos) {
                Ok(Some(mat)) => mat,
                _ => break,
            };
            if mat.start() == mat.end() {
                pos = next_char_boundary(bytes, mat.end());
                continue;
            }
            let len = mat.end() - mat.start();
            if mat.start() == pos {
                // Check exactly what caching relies on: for every window that
                // would be stored, the same window followed by different text
                // must produce the same piece.
                let windows = [
                    (len <= MAX_SHORT_PIECE && short_window(bytes, pos).is_some())
                        .then_some(SHORT_WINDOW),
                    (len <= MAX_LONG_PIECE && long_window(bytes, pos).is_some())
                        .then_some(LONG_WINDOW),
                ];
                for window_len in windows.into_iter().flatten() {
                    let window = &text[pos..pos + window_len];
                    for tail in TAILS {
                        let probe = format!("{window}{tail}");
                        match regex.find_from_pos(&probe, 0) {
                            Ok(Some(m)) if m.start() == 0 && m.end() == len => {}
                            _ => return false,
                        }
                    }
                }
            }
            pos = mat.end();
        }
    }
    true
}

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
    /// Identity used for this tokeniser's entries in the thread-local split
    /// cache, or `None` when the pattern is not eligible for caching.
    split_cache_id: Option<u32>,
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

    /// Yields the pieces the tokeniser pattern splits `text` into, exactly as
    /// `regex.find_iter(text)` would.
    ///
    /// When the pattern is eligible (see `split_cache_id_for`), short pieces
    /// are resolved from the thread-local split cache instead of running
    /// `fancy_regex`'s backtracking VM again for bytes that were already split
    /// before. Cache entries only ever come from a real match against this
    /// haystack, so a cold cache produces exactly the same pieces as `find_iter`.
    fn for_each_piece<F>(&self, text: &str, regex: &Regex, mut f: F) -> fancy_regex::Result<()>
    where
        F: FnMut(&str),
    {
        let Some(id) = self.split_cache_id else {
            for mat in regex.find_iter(text) {
                f(mat?.as_str());
            }
            return Ok(());
        };

        SPLIT_CACHE.with(|cell| {
            let mut cache = cell.borrow_mut();
            let bytes = text.as_bytes();
            let mut last_end = 0usize;
            let mut last_match: Option<usize> = None;

            while last_end <= bytes.len() {
                let pos = last_end;
                // Short window first: it decides most pieces and is the key
                // that generalises best across contexts.
                let short_key = short_window(bytes, pos);
                let short_slot = short_key.map(slot_of);
                if let (Some(key), Some(slot)) = (short_key, short_slot)
                    && let Some(len) = cache.short.get(key, slot, id)
                {
                    let end = pos + len;
                    f(&text[pos..end]);
                    last_match = Some(end);
                    last_end = end;
                    continue;
                }
                // Longer pieces (whole words, mostly) need a wider window.
                let long_key = long_window(bytes, pos);
                let long_slot = long_key.map(|key| slot_of((key ^ (key >> 64)) as u64));
                if let (Some(key), Some(slot)) = (long_key, long_slot)
                    && let Some(len) = cache.long.get(key, slot, id)
                {
                    let end = pos + len;
                    f(&text[pos..end]);
                    last_match = Some(end);
                    last_end = end;
                    continue;
                }

                let mat = match regex.find_from_pos(text, pos)? {
                    Some(mat) => mat,
                    None => break,
                };
                let (start, end) = (mat.start(), mat.end());
                if start == end {
                    // Mirror `fancy_regex::Matches`: after an empty match, move
                    // on and never yield an empty match right after a match.
                    last_end = next_char_boundary(bytes, end);
                    if Some(end) == last_match {
                        continue;
                    }
                } else {
                    last_end = end;
                    if start == pos {
                        let len = end - start;
                        if len <= MAX_SHORT_PIECE
                            && let (Some(key), Some(slot)) = (short_key, short_slot)
                        {
                            cache.short.put(key, slot, id, len);
                        } else if len <= MAX_LONG_PIECE
                            && let (Some(key), Some(slot)) = (long_key, long_slot)
                        {
                            cache.long.put(key, slot, id, len);
                        }
                    }
                }
                last_match = Some(end);
                f(&text[start..end]);
            }
            Ok(())
        })
    }

    pub fn encode_ordinary(&self, text: &str) -> Vec<Rank> {
        // This is the core of the encoding logic; the other functions in here
        // just make things complicated :-)
        let regex = self._get_tl_regex();
        let mut ret = vec![];
        self.for_each_piece(text, regex, |piece| {
            let piece = piece.as_bytes();
            match self.encoder.get(piece) {
                Some(token) => ret.push(*token),
                None => ret.extend(&byte_pair_encode(piece, &self.encoder)),
            }
        })
        .unwrap();
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
            let split = self.for_each_piece(&text[start..end], regex, |piece| {
                let piece = piece.as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    last_piece_token_len = 1;
                    ret.push(*token);
                    return;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                last_piece_token_len = tokens.len();
                ret.extend(&tokens);
            });
            if let Err(e) = split {
                return Err(EncodeError {
                    message: format!("Regex error while tokenizing: {e}"),
                });
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

        // Only patterns whose split decisions are local (see
        // `split_cache_id_for`) may use the thread-local split cache; anything
        // else keeps splitting with `find_iter` exactly as before.
        let split_cache_id = split_cache_id_for(pattern, &regex_tls[0]);

        Ok(Self {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            decoder_flat,
            regex_tls,
            special_regex_tls,
            sorted_token_bytes,
            split_cache_id,
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
    const O200K_PAT: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn core_bpe(pattern: &str) -> crate::CoreBPE {
        // A tiny vocabulary is enough: this only exercises the splitting.
        let mut encoder: HashMap<Vec<u8>, Rank> = HashMap::default();
        for b in 0u16..=255 {
            encoder.insert(vec![b as u8], b as Rank);
        }
        crate::CoreBPE::new_internal(encoder, HashMap::default(), pattern).unwrap()
    }

    fn pieces_with_cache(bpe: &crate::CoreBPE, text: &str) -> Vec<String> {
        let regex = bpe._get_tl_regex();
        let mut out = Vec::new();
        bpe.for_each_piece(text, regex, |piece| out.push(piece.to_string()))
            .unwrap();
        out
    }

    fn pieces_with_regex(bpe: &crate::CoreBPE, text: &str) -> Vec<String> {
        bpe._get_tl_regex()
            .find_iter(text)
            .map(|m| m.unwrap().as_str().to_string())
            .collect()
    }

    #[test]
    fn test_split_cache_is_enabled_for_shipped_patterns() {
        for pattern in [R50K_PAT, CL100K_PAT, O200K_PAT] {
            assert!(
                core_bpe(pattern).split_cache_id.is_some(),
                "expected split caching for {pattern}"
            );
        }
    }

    #[test]
    fn test_split_cache_rejects_non_local_patterns() {
        // `\G`, lookbehind and `^` all depend on more than the bytes at the
        // piece start, so they must keep the plain `find_iter` path.
        for pattern in [r"\Ga+|.", r"(?<=a)b|.", r"^a|."] {
            assert!(
                core_bpe(pattern).split_cache_id.is_none(),
                "expected no split caching for {pattern}"
            );
        }
    }

    #[test]
    fn test_cached_split_matches_find_iter() {
        let texts = [
            "",
            "a",
            "hello world",
            "The quick brown fox jumps over the lazy dog. The quick brown fox!",
            "   leading and    trailing   ",
            "line one\nline two\r\n\r\nline three\n\n\n",
            "punctuation!!! ...,,, ??? '''",
            "it's a dog's life, isn't it? I'd say it's fine",
            "héllo wörld é中文🙂 mixed 123 4567 89",
            "\t\ttabs\tand \n newlines \n\t ",
            "1234567890 12 345 6789 0",
            "a  b   c    d     e      f",
            "byte pair encoding compresses reversible representations repeatedly",
            "supercalifragilisticexpialidocious supercalifragilisticexpialidocious",
            "'s 's's ''s 't 'll 've 're 'd",
            "trailing whitespace at end of text   ",
        ];
        for pattern in [R50K_PAT, CL100K_PAT, O200K_PAT] {
            let bpe = core_bpe(pattern);
            for text in texts {
                // Run twice so the second pass reads the entries the first pass wrote.
                for _ in 0..2 {
                    assert_eq!(
                        pieces_with_cache(&bpe, text),
                        pieces_with_regex(&bpe, text),
                        "pieces diverged for {text:?} with pattern {pattern}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_cached_split_matches_find_iter_randomised() {
        const ALPHABET: [&str; 27] = [
            "a",
            "b",
            "Z",
            "0",
            "9",
            " ",
            "  ",
            "\t",
            "\n",
            "\r\n",
            ".",
            ",",
            "'",
            "!",
            "-",
            "<",
            "|",
            ">",
            "é",
            "中",
            "🙂",
            "\u{a0}",
            "encoding",
            "loremipsumdolor",
            " reversible",
            "1234567890123456789",
            "wörter",
        ];
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for pattern in [R50K_PAT, CL100K_PAT, O200K_PAT] {
            let bpe = core_bpe(pattern);
            for _ in 0..2000 {
                let mut text = String::new();
                let parts = (next() % 24) as usize;
                for _ in 0..parts {
                    text.push_str(ALPHABET[(next() % ALPHABET.len() as u64) as usize]);
                }
                assert_eq!(
                    pieces_with_cache(&bpe, &text),
                    pieces_with_regex(&bpe, &text),
                    "pieces diverged for {text:?} with pattern {pattern}"
                );
            }
        }
    }
}
