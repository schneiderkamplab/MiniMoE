from __future__ import annotations

from typing import cast

import pytest

from tests.tokenizer_checks import assert_tokenizers_equivalent, build_tokenizer, equivalence_texts
from tokcleanse import MergeRule, TokenizerContents, clean, topological_sort_rules

_RANDOM_SEEDS = (0, 1, 7, 42, 1729)


def test_clean_literal_matches_original_merges(tokenizer_contents: TokenizerContents) -> None:
    assert clean(tokenizer_contents, mode="literal") == list(tokenizer_contents.original_merges)


@pytest.mark.parametrize("seed", _RANDOM_SEEDS)
def test_random_topological_sorts_match_original_tokenization(
    tokenizer_contents: TokenizerContents,
    adversarial_strings: tuple[str, ...],
    fuzz_strings: tuple[str, ...],
    gutenberg_books: dict[str, str],
    seed: int,
) -> None:
    original = _build_tokenizer(tokenizer_contents, tokenizer_contents.original_merges)
    reordered_merges = cast(list[tuple[str, str]], clean(tokenizer_contents, mode="literal", seed=seed))
    reordered = _build_tokenizer(tokenizer_contents, reordered_merges)

    assert_tokenizers_equivalent(
        original=original,
        reordered=reordered,
        texts=equivalence_texts(
            adversarial_strings=adversarial_strings,
            fuzz_strings=fuzz_strings,
            gutenberg_books=gutenberg_books,
        ),
    )


@pytest.mark.parametrize(
    "order_name",
    [
        "good",
        "length_then_lexicographic",
        "lexicographic_then_length",
        "descending_length_then_lexicographic",
        "operands_then_output",
        "whitespace_priority_then_length",
    ],
)
def test_ordered_topological_sorts_match_original_tokenization(
    tokenizer_contents: TokenizerContents,
    adversarial_strings: tuple[str, ...],
    fuzz_strings: tuple[str, ...],
    gutenberg_books: dict[str, str],
    order_name: str,
) -> None:
    original = _build_tokenizer(tokenizer_contents, tokenizer_contents.original_merges)
    reordered_merges = cast(
        list[tuple[str, str]],
        clean(tokenizer_contents, mode="literal", order_name=order_name),
    )
    reordered = _build_tokenizer(tokenizer_contents, reordered_merges)

    assert _is_topological_sort(
        tokenizer_contents,
        topological_sort_rules(tokenizer_contents, order_name=order_name),
    )
    assert_tokenizers_equivalent(
        original=original,
        reordered=reordered,
        texts=equivalence_texts(
            adversarial_strings=adversarial_strings,
            fuzz_strings=fuzz_strings,
            gutenberg_books=gutenberg_books,
        ),
    )


def _build_tokenizer(
    contents: TokenizerContents,
    merges: tuple[tuple[str, str], ...] | list[tuple[str, str]],
):
    return build_tokenizer(contents, merges)


def _is_topological_sort(contents: TokenizerContents, rules: list[MergeRule]) -> bool:
    seen: set[int] = set()
    for rule in rules:
        if any(predecessor not in seen for predecessor in contents.graph.predecessors[rule.index]):
            return False
        seen.add(rule.index)
    return True
