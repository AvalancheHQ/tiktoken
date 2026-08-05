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


def _make_text_from(paragraph: str, target_len: int) -> str:
    text = paragraph * (target_len // len(paragraph) + 1)
    return text[:target_len]


def _make_text(target_len: int) -> str:
    return _make_text_from(_PARAGRAPH, target_len)


# Small (~1 line), medium (~2 KB) and large (~64 KB) documents.
TEXT_SIZES = {
    "small": _make_text(64),
    "medium": _make_text(2_048),
    "large": _make_text(65_536),
}


# Text in scripts that are written without spaces between words (Japanese,
# Chinese, Thai). The split pattern keeps a whole run of letters together, so
# these inputs produce long multi-byte pieces that are not vocabulary entries
# and therefore go through the byte-pair merges, unlike English prose where
# almost every piece is a single token.
_CJK_PARAGRAPH = (
    "私はプログラミングが好きです。トークン化とはテキストを数値の列に変換する処理のことです。"
    "言語モデルは文字ではなくトークンの並びを見ています。日本語や中国語には単語の区切りがありません。"
    "语言模型看到的是词元序列而不是原始文本。字节对编码是一种把文本转换成词元的方法。"
    "โมเดลภาษาเห็นลำดับของโทเคนไม่ใช่ข้อความดิบ"
)

CJK_TEXT = _make_text_from(_CJK_PARAGRAPH, 4_096)


@pytest.mark.parametrize("encoding_name", ENCODING_NAMES)
def test_encode_ordinary_cjk(benchmark, encoding_name: str) -> None:
    enc = _get_encoding(encoding_name)
    tokens = benchmark(enc.encode_ordinary, CJK_TEXT)
    assert tokens


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
