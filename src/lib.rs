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

// ============================================================================
// GPT-2 ("data gym") vocabulary files
// ============================================================================
//
// The GPT-2 vocabulary does not ship in the `.tiktoken` format. It is the
// original pair of files: `vocab.bpe`, a priority-ordered list of merges, and
// `encoder.json`, the same table spelled out as token -> rank. Both files write
// a token's bytes as a string of printable characters, one character per byte.
//
// Turning them into a mergeable-ranks table used to be a pure-Python loop that
// decoded every one of those characters through a dict lookup — twice per merge
// plus once per `encoder.json` key, ~150k calls for GPT-2. The primitives below
// do that work with a flat table instead, and cross-check `encoder.json` in a
// single streaming pass so the loader no longer has to build a throwaway
// 50k-entry dict just to compare it away.

/// Error raised while building the mergeable ranks from the GPT-2 vocabulary
/// files: a malformed file, or a table that disagrees with `encoder.json`.
#[derive(Debug, Clone)]
pub struct DataGymError {
    pub message: String,
    /// Set when the two files parse fine but describe different tables. The
    /// Python loader reported that case with an `assert`, so the binding keeps
    /// raising `AssertionError` for it.
    pub is_mismatch: bool,
}

impl DataGymError {
    fn malformed(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            is_mismatch: false,
        }
    }

    fn mismatch(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            is_mismatch: true,
        }
    }
}

impl std::fmt::Display for DataGymError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for DataGymError {}

/// Maps the printable characters the GPT-2 vocabulary files use back to the
/// bytes they stand for.
///
/// The mapping itself is built by `tiktoken/load.py` (it owns the
/// `str.isprintable()` rules) and handed over as `char -> byte` pairs. Every
/// character it uses is below [`Self::TABLE_LEN`], so a flat table resolves one
/// with a single indexed load instead of a hash lookup.
pub struct DataGymByteTable {
    table: Box<[u16; Self::TABLE_LEN]>,
}

impl DataGymByteTable {
    const TABLE_LEN: usize = 512;
    const UNMAPPED: u16 = u16::MAX;

    pub fn new(mapping: impl IntoIterator<Item = (char, u8)>) -> Result<Self, DataGymError> {
        let mut table = Box::new([Self::UNMAPPED; Self::TABLE_LEN]);
        for (ch, byte) in mapping {
            let slot = usize::try_from(u32::from(ch))
                .ok()
                .filter(|&slot| slot < Self::TABLE_LEN)
                .ok_or_else(|| {
                    DataGymError::malformed(format!(
                        "data gym character {ch:?} is outside the supported range"
                    ))
                })?;
            table[slot] = u16::from(byte);
        }
        Ok(Self { table })
    }

    /// Appends the byte `ch` stands for to `out`.
    #[inline]
    fn push_byte_of(&self, ch: char, out: &mut Vec<u8>) -> Result<(), DataGymError> {
        let slot = u32::from(ch) as usize;
        match self.table.get(slot) {
            Some(&byte) if byte != Self::UNMAPPED => {
                out.push(byte as u8);
                Ok(())
            }
            _ => Err(DataGymError::malformed(format!(
                "{ch:?} is not a data gym character"
            ))),
        }
    }

    /// Appends the bytes `s` stands for to `out`.
    pub fn push_decoded(&self, s: &str, out: &mut Vec<u8>) -> Result<(), DataGymError> {
        out.reserve(s.len());
        for ch in s.chars() {
            self.push_byte_of(ch, out)?;
        }
        Ok(())
    }
}

/// The merge pairs of a `vocab.bpe` file, in priority order.
///
/// Mirrors `contents.split("\n")[1:-1]` followed by `merge_str.split()`: the
/// first line is the version header and the piece after the final newline is
/// dropped.
pub fn data_gym_merge_pairs(contents: &str) -> DataGymMergePairs<'_> {
    let mut lines = contents.split('\n');
    lines.next(); // version header
    DataGymMergePairs {
        lines: lines.peekable(),
    }
}

pub struct DataGymMergePairs<'a> {
    lines: std::iter::Peekable<std::str::Split<'a, char>>,
}

impl<'a> Iterator for DataGymMergePairs<'a> {
    type Item = Result<(&'a str, &'a str), DataGymError>;

    fn next(&mut self) -> Option<Self::Item> {
        let line = self.lines.next()?;
        // The `[..-1]` half of the Python slice: whatever follows the last
        // newline is not a merge.
        self.lines.peek()?;
        let mut halves = line.split_whitespace();
        Some(match (halves.next(), halves.next(), halves.next()) {
            (Some(first), Some(second), None) => Ok((first, second)),
            _ => Err(DataGymError::malformed(format!(
                "expected two merge halves in vocab.bpe line {line:?}"
            ))),
        })
    }
}

/// Streams a flat `{"token": rank, ...}` JSON object — the shape of the GPT-2
/// `encoder.json` — decoding every key's characters straight into token bytes
/// and handing each entry to `visit`.
///
/// Deliberately strict: only that exact shape is accepted, so anything else is
/// reported as malformed rather than silently reinterpreted. Keys are decoded
/// into a reused buffer, so no per-token allocation is needed to check the file.
pub fn for_each_encoder_json_entry(
    contents: &[u8],
    table: &DataGymByteTable,
    mut visit: impl FnMut(&[u8], Rank) -> Result<(), DataGymError>,
) -> Result<(), DataGymError> {
    let text = std::str::from_utf8(contents)
        .map_err(|e| DataGymError::malformed(format!("encoder.json is not valid UTF-8: {e}")))?;
    let mut scanner = JsonScanner { text, pos: 0 };
    let mut token = Vec::new();

    scanner.skip_whitespace();
    scanner.expect(b'{')?;
    scanner.skip_whitespace();
    if scanner.peek() == Some(b'}') {
        scanner.pos += 1;
    } else {
        loop {
            scanner.skip_whitespace();
            token.clear();
            scanner.read_string_as_bytes(table, &mut token)?;
            scanner.skip_whitespace();
            scanner.expect(b':')?;
            scanner.skip_whitespace();
            let rank = scanner.read_rank()?;
            visit(&token, rank)?;
            scanner.skip_whitespace();
            match scanner.bump() {
                Some(b',') => continue,
                Some(b'}') => break,
                _ => return Err(scanner.error("expected ',' or '}'")),
            }
        }
    }
    scanner.skip_whitespace();
    if scanner.pos != scanner.text.len() {
        return Err(scanner.error("expected end of encoder.json"));
    }
    Ok(())
}

struct JsonScanner<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> JsonScanner<'a> {
    #[inline]
    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }

    #[inline]
    fn bump(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.pos += 1;
        Some(byte)
    }

    #[inline]
    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), DataGymError> {
        if self.bump() == Some(byte) {
            Ok(())
        } else {
            self.pos = self.pos.saturating_sub(1);
            Err(self.error(&format!("expected {:?}", byte as char)))
        }
    }

    fn error(&self, what: &str) -> DataGymError {
        DataGymError::malformed(format!(
            "malformed encoder.json: {what} at byte {}",
            self.pos
        ))
    }

    /// Reads a JSON string, mapping each of its characters through `table`.
    fn read_string_as_bytes(
        &mut self,
        table: &DataGymByteTable,
        out: &mut Vec<u8>,
    ) -> Result<(), DataGymError> {
        self.expect(b'"')?;
        loop {
            let byte = self
                .peek()
                .ok_or_else(|| self.error("unterminated string"))?;
            match byte {
                b'"' => {
                    self.pos += 1;
                    return Ok(());
                }
                b'\\' => {
                    self.pos += 1;
                    let ch = self.read_escape()?;
                    table.push_byte_of(ch, out)?;
                }
                0x00..=0x1f => return Err(self.error("unescaped control character")),
                _ => {
                    // `self.text` is valid UTF-8 and `pos` sits on a character
                    // boundary, so a character is always there to read.
                    let ch = self.text[self.pos..].chars().next().unwrap();
                    self.pos += ch.len_utf8();
                    table.push_byte_of(ch, out)?;
                }
            }
        }
    }

    fn read_escape(&mut self) -> Result<char, DataGymError> {
        let byte = self
            .bump()
            .ok_or_else(|| self.error("unterminated escape"))?;
        Ok(match byte {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => {
                let unit = self.read_hex4()?;
                match unit {
                    0xd800..=0xdbff => {
                        if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
                            return Err(self.error("unpaired surrogate escape"));
                        }
                        let low = self.read_hex4()?;
                        if !(0xdc00..=0xdfff).contains(&low) {
                            return Err(self.error("unpaired surrogate escape"));
                        }
                        let code = 0x1_0000
                            + ((u32::from(unit) - 0xd800) << 10)
                            + (u32::from(low) - 0xdc00);
                        char::from_u32(code).ok_or_else(|| self.error("invalid surrogate pair"))?
                    }
                    0xdc00..=0xdfff => return Err(self.error("unpaired surrogate escape")),
                    _ => char::from_u32(u32::from(unit))
                        .ok_or_else(|| self.error("invalid escape"))?,
                }
            }
            _ => return Err(self.error("unknown escape")),
        })
    }

    fn read_hex4(&mut self) -> Result<u16, DataGymError> {
        let mut unit = 0u16;
        for _ in 0..4 {
            let digit = self
                .bump()
                .and_then(|byte| (byte as char).to_digit(16))
                .ok_or_else(|| self.error("expected four hex digits"))?;
            unit = unit * 16 + digit as u16;
        }
        Ok(unit)
    }

    fn read_rank(&mut self) -> Result<Rank, DataGymError> {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(self.error("expected a rank"));
        }
        self.text[start..self.pos]
            .parse()
            .map_err(|_| self.error("rank out of range"))
    }
}

/// Builds the mergeable-ranks table of the GPT-2 vocabulary.
///
/// `single_bytes` are the one-byte tokens in rank order and `vocab_bpe` is the
/// contents of `vocab.bpe`; together they define every token and its rank.
/// `encoder_json` is then streamed to cross-check the result, which is what the
/// Python loader's `assert bpe_ranks == encoder_json_loaded` did — tiktoken
/// relies on ranks being ordered by merge priority, so the check is load
/// bearing. With `clobber_one_byte_tokens` the one-byte ranks are taken from
/// `encoder.json` instead of being checked against it.
///
/// Returns the tokens in rank order together with their ranks (they only differ
/// from `0..n` when one-byte tokens are clobbered).
pub fn data_gym_mergeable_ranks(
    table: &DataGymByteTable,
    single_bytes: &[u8],
    vocab_bpe: &str,
    encoder_json: &[u8],
    clobber_one_byte_tokens: bool,
) -> Result<Vec<(Vec<u8>, Rank)>, DataGymError> {
    // Tokens in insertion order, which is also rank order: the one-byte tokens
    // first, then one per merge.
    let mut tokens: Vec<Vec<u8>> = single_bytes.iter().map(|&byte| vec![byte]).collect();
    let mut merged = Vec::new();
    for pair in data_gym_merge_pairs(vocab_bpe) {
        let (first, second) = pair?;
        merged.clear();
        table.push_decoded(first, &mut merged)?;
        table.push_decoded(second, &mut merged)?;
        tokens.push(std::mem::take(&mut merged));
    }
    let mut ranks: Vec<Rank> = (0..tokens.len())
        .map(|rank| {
            Rank::try_from(rank)
                .map_err(|_| DataGymError::malformed("too many tokens for a 32-bit rank"))
        })
        .collect::<Result<_, _>>()?;

    // A later merge that produces an already-known token wins, exactly like
    // re-assigning the key in the Python dict.
    let index: HashMap<&[u8], usize> = tokens
        .iter()
        .enumerate()
        .map(|(i, token)| (token.as_slice(), i))
        .collect();

    // `encoder.json` also lists the special tokens, which are not mergeable.
    const NOT_MERGEABLE: [&[u8]; 2] = [b"<|endoftext|>", b"<|startoftext|>"];
    let mut checked = 0usize;
    for_each_encoder_json_entry(encoder_json, table, |token, rank| {
        if NOT_MERGEABLE.contains(&token) {
            return Ok(());
        }
        checked += 1;
        let &i = index.get(token).ok_or_else(|| {
            DataGymError::mismatch(format!(
                "encoder.json has token {token:?} (rank {rank}), which vocab.bpe does not produce"
            ))
        })?;
        if clobber_one_byte_tokens && token.len() == 1 {
            ranks[i] = rank;
        } else if ranks[i] != rank {
            return Err(DataGymError::mismatch(format!(
                "encoder.json ranks token {token:?} {rank}, vocab.bpe ranks it {}",
                ranks[i]
            )));
        }
        Ok(())
    })?;
    if checked != index.len() {
        return Err(DataGymError::mismatch(format!(
            "vocab.bpe produces {} tokens, encoder.json lists {checked}",
            index.len()
        )));
    }
    drop(index);

    Ok(tokens.into_iter().zip(ranks).collect())
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

    /// The subset of the GPT-2 character mapping the tests below need: printable
    /// ASCII stands for itself, and `Ġ`/`Ĉ` stand for the space and 0x08.
    fn data_gym_table() -> crate::DataGymByteTable {
        let mut mapping: Vec<(char, u8)> = (b'!'..=b'~').map(|byte| (byte as char, byte)).collect();
        mapping.push(('Ġ', b' '));
        mapping.push(('Ĉ', 0x08));
        crate::DataGymByteTable::new(mapping).unwrap()
    }

    #[test]
    fn test_data_gym_merge_pairs() {
        // The version header and whatever follows the last newline are dropped,
        // just like `contents.split("\n")[1:-1]`.
        let pairs: Vec<_> = crate::data_gym_merge_pairs("#version: 0.2\nĠ t\nh e\n")
            .map(Result::unwrap)
            .collect();
        assert_eq!(pairs, vec![("Ġ", "t"), ("h", "e")]);
        assert_eq!(crate::data_gym_merge_pairs("").count(), 0);
        assert_eq!(crate::data_gym_merge_pairs("#version: 0.2\n").count(), 0);
        // A line without exactly two halves is malformed, as unpacking it was in
        // Python.
        assert!(
            crate::data_gym_merge_pairs("#version: 0.2\nlonely\n")
                .next()
                .unwrap()
                .is_err()
        );
    }

    #[test]
    fn test_encoder_json_entries() {
        let table = data_gym_table();
        let mut entries = Vec::new();
        crate::for_each_encoder_json_entry(
            br#" {"!": 0, "\"": 1, "\u0120t": 2, "\\": 3, "\u0108": 4} "#,
            &table,
            |token, rank| {
                entries.push((token.to_vec(), rank));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            entries,
            vec![
                (b"!".to_vec(), 0),
                (b"\"".to_vec(), 1),
                (b" t".to_vec(), 2),
                (b"\\".to_vec(), 3),
                (vec![0x08], 4),
            ]
        );

        // An empty object is valid; anything that is not this exact shape is not.
        crate::for_each_encoder_json_entry(b"{}", &table, |_, _| Ok(())).unwrap();
        for malformed in [
            &b"{"[..],
            b"{\"!\" 0}",
            b"{\"!\": }",
            b"{\"!\": -1}",
            b"{\"!\": 0,}",
            b"{\"!\": 0} trailing",
            b"{\"\\q\": 0}",
            b"{\"\\ud800\": 0}",
        ] {
            assert!(
                crate::for_each_encoder_json_entry(malformed, &table, |_, _| Ok(())).is_err(),
                "{malformed:?} should be rejected"
            );
        }
    }

    #[test]
    fn test_data_gym_mergeable_ranks() {
        let table = data_gym_table();
        let single_bytes = [b'!', b' '];
        let vocab_bpe = "#version: 0.2\n! Ġ\n!Ġ !\n";
        // Ranks: "!" 0, " " 1, "! " 2, "! !" 3.
        let encoder_json = br#"{"!": 0, "\u0120": 1, "!\u0120": 2, "!\u0120!": 3}"#;
        let ranks =
            crate::data_gym_mergeable_ranks(&table, &single_bytes, vocab_bpe, encoder_json, false)
                .unwrap();
        assert_eq!(
            ranks,
            vec![
                (b"!".to_vec(), 0),
                (b" ".to_vec(), 1),
                (b"! ".to_vec(), 2),
                (b"! !".to_vec(), 3),
            ]
        );

        // Special tokens in the encoder json are not mergeable and are ignored.
        let with_special =
            br#"{"!": 0, "\u0120": 1, "!\u0120": 2, "!\u0120!": 3, "<|endoftext|>": 9}"#;
        assert!(
            crate::data_gym_mergeable_ranks(&table, &single_bytes, vocab_bpe, with_special, false)
                .is_ok()
        );

        // A disagreement between the two files is reported as a mismatch.
        let disagrees = br#"{"!": 0, "\u0120": 1, "!\u0120": 3, "!\u0120!": 2}"#;
        let err =
            crate::data_gym_mergeable_ranks(&table, &single_bytes, vocab_bpe, disagrees, false)
                .unwrap_err();
        assert!(err.is_mismatch);
        // As is a missing entry.
        let incomplete = br#"{"!": 0, "\u0120": 1, "!\u0120": 2}"#;
        assert!(
            crate::data_gym_mergeable_ranks(&table, &single_bytes, vocab_bpe, incomplete, false)
                .unwrap_err()
                .is_mismatch
        );

        // Clobbering takes the one-byte ranks from the encoder json instead.
        let clobbered = br#"{"!": 7, "\u0120": 1, "!\u0120": 2, "!\u0120!": 3}"#;
        let ranks =
            crate::data_gym_mergeable_ranks(&table, &single_bytes, vocab_bpe, clobbered, true)
                .unwrap();
        assert_eq!(ranks[0], (b"!".to_vec(), 7));
    }
}
