"""Representative tiktoken workload used to train a PGO profile.

Run by ``scripts/pgo_build.sh`` against an instrumented build of the Rust core.
The corpus is deliberately *not* the benchmark corpus: it covers the input
shapes real callers hit (prose, source code, JSON, markup, non-ASCII text,
whitespace runs, numbers, contractions) across both vocabulary generations, and
it exercises the encode, decode and batch entry points so the profile is not
skewed towards a single code path.
"""

from __future__ import annotations

import tiktoken

PROSE = (
    "In the beginning the Universe was created. This has made a lot of people "
    "very angry and been widely regarded as a bad move. Meanwhile, the ships "
    "hung in the sky in much the same way that bricks don't.\n\n"
)
CODE = (
    "fn main() {\n    let mut total = 0usize;\n    for (i, item) in items.iter().enumerate() {\n"
    '        total += item.len() * i;  // accumulate\n    }\n    println!("{total:?}");\n}\n'
)
JSON = '{"id": 12345, "name": "widget", "tags": ["a", "b", "c"], "price": 9.99, "ok": true}\n'
MARKUP = "<div class=\"row\"><span id='x'>Hello &amp; goodbye</span></div>\n"
UNICODE = "Größe: 12 µm — naïve café; 東京は晴れです。 Привет мир! 🚀🚀 3.14159\n"
SPACES = "word    \n\n\t  tabbed     spaced\n   \n"
NUMBERS = "1 22 333 4444 55555 2024-01-31 3.14159 1e10 0xFF 100%\n"
CONTRACTIONS = "It's Bob's, they're, we've, I'd, don't, can't, o'clock, 'quoted'\n"

DOCS = [PROSE, CODE, JSON, MARKUP, UNICODE, SPACES, NUMBERS, CONTRACTIONS]


def main() -> None:
    for name in ("cl100k_base", "gpt2"):
        enc = tiktoken.get_encoding(name)
        for scale in (1, 8, 64):
            for doc in DOCS:
                text = doc * scale
                tokens = enc.encode_ordinary(text)
                enc.decode(tokens)
                enc.encode(text)
                enc.decode_bytes(tokens)
        enc.encode_ordinary_batch(
            [PROSE * 16, CODE * 16, JSON * 16, UNICODE * 16] * 8, num_threads=4
        )
        enc.encode_batch([PROSE * 8, CODE * 8] * 8, num_threads=4)
        enc.encode("hello <|endoftext|> world", allowed_special="all")
        enc.encode_with_unstable(PROSE[:200])


if __name__ == "__main__":
    main()
