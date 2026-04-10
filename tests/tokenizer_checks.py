from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer

from tokcleanse import TokenizerContents

_BOOK_SAMPLE_LENGTH = 12000
_TEXT_BUCKET_MAX_LENGTHS = (64, 512)


def build_tokenizer(
    contents: TokenizerContents,
    merges: tuple[tuple[str, str], ...] | list[tuple[str, str]],
) -> Tokenizer:
    tokenizer_config = json.loads((contents.resolved_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer_config["model"]["merges"] = [list(merge) for merge in merges]
    return Tokenizer.from_str(json.dumps(tokenizer_config, ensure_ascii=False))


def load_local_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    return Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))


def assert_tokenizers_equivalent(
    *,
    original: Tokenizer,
    reordered: Tokenizer,
    texts: tuple[str, ...],
) -> None:
    for bucket in bucket_texts_by_length(texts):
        original_encodings = original.encode_batch(list(bucket))
        reordered_encodings = reordered.encode_batch(list(bucket))
        assert [encoding.ids for encoding in reordered_encodings] == [
            encoding.ids for encoding in original_encodings
        ]


def equivalence_texts(
    *,
    adversarial_strings: tuple[str, ...],
    fuzz_strings: tuple[str, ...],
    gutenberg_books: dict[str, str],
) -> tuple[str, ...]:
    return (
        *adversarial_strings,
        *fuzz_strings,
        *book_samples(gutenberg_books),
    )


def book_samples(gutenberg_books: dict[str, str]) -> tuple[str, ...]:
    samples: list[str] = []
    for key in sorted(gutenberg_books):
        text = gutenberg_books[key]
        if len(text) <= _BOOK_SAMPLE_LENGTH:
            samples.append(text)
            continue
        start = text[:_BOOK_SAMPLE_LENGTH]
        middle_index = max(0, (len(text) - _BOOK_SAMPLE_LENGTH) // 2)
        middle = text[middle_index : middle_index + _BOOK_SAMPLE_LENGTH]
        end = text[-_BOOK_SAMPLE_LENGTH:]
        samples.extend((start, middle, end))
    return tuple(samples)


def bucket_texts_by_length(texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    buckets: list[list[str]] = [[] for _ in range(len(_TEXT_BUCKET_MAX_LENGTHS) + 1)]
    for text in texts:
        bucket_index = _bucket_index_for_length(len(text))
        buckets[bucket_index].append(text)
    return tuple(tuple(bucket) for bucket in buckets if bucket)


def _bucket_index_for_length(length: int) -> int:
    for index, max_length in enumerate(_TEXT_BUCKET_MAX_LENGTHS):
        if length <= max_length:
            return index
    return len(_TEXT_BUCKET_MAX_LENGTHS)
