"""CodSpeed performance benchmarks for tiktoken.

These benchmarks exercise the hot paths of the public tokeniser API
(encoding, decoding, and batch encoding) across a couple of representative
encodings. They are written with the ``pytest-codspeed`` ``benchmark`` fixture
so they can be measured with CodSpeed's instrumentation, walltime and memory
instruments.

The encodings are loaded once at module import time (the vocabulary files are
downloaded and cached by tiktoken), so the download cost is not part of any
measurement.
"""

from __future__ import annotations

import functools

import pytest

import tiktoken

# Encodings covering the two main BPE regex/vocabulary generations used by
# OpenAI models: the GPT-2/GPT-3 family (gpt2) and the GPT-4 family (cl100k_base).
ENCODING_NAMES = ["gpt2", "cl100k_base"]


@functools.lru_cache(maxsize=None)
def _get_encoding(name: str) -> tiktoken.Encoding:
    enc = tiktoken.get_encoding(name)
    # Warm up so the very first call's lazy initialisation is not measured.
    enc.encode_ordinary("warmup")
    return enc


# A paragraph of natural language, repeated to build inputs of varying size.
_PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog. "
    "Byte pair encoding (BPE) is a way of converting text into tokens. "
    "It is reversible and lossless, works on arbitrary text, and it "
    "compresses the input so the token sequence is shorter than the bytes. "
    "Language models see a sequence of numbers known as tokens, not raw text. "
)


def _make_text(target_len: int) -> str:
    text = _PARAGRAPH * (target_len // len(_PARAGRAPH) + 1)
    return text[:target_len]


# Small (~1 line), medium (~2 KB) and large (~64 KB) documents.
TEXT_SIZES = {
    "small": _make_text(64),
    "medium": _make_text(2_048),
    "large": _make_text(65_536),
}


@pytest.mark.parametrize("encoding_name", ENCODING_NAMES)
@pytest.mark.parametrize("size", list(TEXT_SIZES))
def test_encode_ordinary(benchmark, encoding_name: str, size: str) -> None:
    enc = _get_encoding(encoding_name)
    text = TEXT_SIZES[size]
    tokens = benchmark(enc.encode_ordinary, text)
    assert tokens


@pytest.mark.parametrize("encoding_name", ENCODING_NAMES)
@pytest.mark.parametrize("size", list(TEXT_SIZES))
def test_encode(benchmark, encoding_name: str, size: str) -> None:
    enc = _get_encoding(encoding_name)
    text = TEXT_SIZES[size]
    tokens = benchmark(enc.encode, text)
    assert tokens


@pytest.mark.parametrize("encoding_name", ENCODING_NAMES)
@pytest.mark.parametrize("size", list(TEXT_SIZES))
def test_decode(benchmark, encoding_name: str, size: str) -> None:
    enc = _get_encoding(encoding_name)
    tokens = enc.encode_ordinary(TEXT_SIZES[size])
    text = benchmark(enc.decode, tokens)
    assert text


@pytest.mark.parametrize("encoding_name", ENCODING_NAMES)
def test_encode_ordinary_batch(benchmark, encoding_name: str) -> None:
    enc = _get_encoding(encoding_name)
    documents = [TEXT_SIZES["medium"]] * 64
    result = benchmark(functools.partial(enc.encode_ordinary_batch, num_threads=4), documents)
    assert len(result) == len(documents)


# Prompts that stop in the middle of a word, i.e. the case
# ``encode_with_unstable`` exists for: the last token is unstable because more
# text could still merge into it, so every vocabulary token that could continue
# it has to be considered.
UNSTABLE_TEXTS = [
    "Language models see a sequence of numbers known as toke",
    "The quick brown fox jumps over the lazy do",
    "def compute_average(values):\n    return sum(values) / le",
]


@pytest.mark.parametrize("encoding_name", ENCODING_NAMES)
def test_encode_with_unstable(benchmark, encoding_name: str) -> None:
    enc = _get_encoding(encoding_name)

    def encode_all() -> list:
        return [enc.encode_with_unstable(text) for text in UNSTABLE_TEXTS]

    results = benchmark(encode_all)
    assert all(completions for _, completions in results)


@pytest.mark.parametrize("encoding_name", ENCODING_NAMES)
def test_construct_encoding(benchmark, encoding_name: str) -> None:
    """Building a tokeniser from an already-loaded vocabulary.

    This is what every process pays on its first ``tiktoken.get_encoding()``
    call: the vocabulary files are downloaded and parsed once and cached on
    disk, but the tokeniser itself (the encoder/decoder tables and the split
    regexes) is rebuilt in every process. The download/parse is excluded here
    so the measurement only covers that work.
    """
    template = _get_encoding(encoding_name)
    mergeable_ranks = template._mergeable_ranks
    special_tokens = template._special_tokens
    pat_str = template._pat_str

    enc = benchmark(
        functools.partial(
            tiktoken.Encoding,
            encoding_name,
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
    )
    assert enc.max_token_value == template.max_token_value
