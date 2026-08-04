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


# The same content in Chinese. Text without spaces makes the tokeniser split
# emit long pieces (tens of bytes instead of a handful), so the BPE merges,
# rather than the split, dominate encoding.
_CJK_PARAGRAPH = (
    "字节对编码是一种把文本转换成词元的方法。"
    "它可以逆向还原而不丢失信息，适用于任意文本，"
    "并且会压缩输入，使词元序列比原始字节更短。"
    "语言模型看到的是一串数字，也就是词元，而不是原始文本。"
)


def _make_text(target_len: int, paragraph: str = _PARAGRAPH) -> str:
    text = paragraph * (target_len // len(paragraph) + 1)
    return text[:target_len]


# Small (~1 line), medium (~2 KB) and large (~64 KB) documents, plus a
# non-ASCII (~2 KB) document.
TEXT_SIZES = {
    "small": _make_text(64),
    "medium": _make_text(2_048),
    "large": _make_text(65_536),
    "cjk": _make_text(2_048, _CJK_PARAGRAPH),
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
