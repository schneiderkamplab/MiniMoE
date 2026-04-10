from __future__ import annotations

from pathlib import Path

from tests.tokenizer_checks import (
    assert_tokenizers_equivalent,
    equivalence_texts,
    load_local_tokenizer,
)
from tokcleanse import TokenizerContents


def test_odin_id_matches_original_tokenizer(
    model_dir: Path,
    odin_id_dir: Path,
    tokenizer_contents: TokenizerContents,
    adversarial_strings: tuple[str, ...],
    fuzz_strings: tuple[str, ...],
    gutenberg_books: dict[str, str],
) -> None:
    original = load_local_tokenizer(model_dir)
    odin_id = load_local_tokenizer(odin_id_dir)

    assert odin_id_dir.name == "odin-id"
    assert odin_id_dir != tokenizer_contents.resolved_dir
    assert_tokenizers_equivalent(
        original=original,
        reordered=odin_id,
        texts=equivalence_texts(
            adversarial_strings=adversarial_strings,
            fuzz_strings=fuzz_strings,
            gutenberg_books=gutenberg_books,
        ),
    )
