from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokcleanse.compare import (
    ComparisonSummary,
    DEFAULT_COMPARE_PROMPT,
    PromptComparison,
    load_prompts,
    serialize_comparisons,
    summarize_comparisons,
)


def test_load_prompts_combines_flags_and_jsonl(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        "\n".join(
            [
                json.dumps({"text": "Prompt from file 1"}),
                json.dumps({"text": "Prompt from file 2"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = load_prompts(prompt_texts=["Inline prompt"], prompt_file=prompt_file)

    assert prompts == ("Prompt from file 1", "Prompt from file 2")


def test_load_prompts_defaults_when_no_input_is_provided() -> None:
    prompts = load_prompts(prompt_texts=[], prompt_file=None)

    assert prompts == (DEFAULT_COMPARE_PROMPT,)


def test_summarize_comparisons_includes_close_match_statistics() -> None:
    summary = summarize_comparisons(
        [
            PromptComparison(
                prompt="A",
                answer_a="x",
                answer_b="x",
                match=True,
                normalized_match=True,
                semantic_similarity=1.0,
                semantic_match=True,
                judge_match=None,
                close_match=True,
            ),
            PromptComparison(
                prompt="B",
                answer_a="same meaning",
                answer_b="Same   meaning",
                match=False,
                normalized_match=True,
                semantic_similarity=0.98,
                semantic_match=True,
                judge_match=None,
                close_match=True,
            ),
            PromptComparison(
                prompt="C",
                answer_a="left",
                answer_b="right",
                match=False,
                normalized_match=False,
                semantic_similarity=0.31,
                semantic_match=False,
                judge_match=True,
                close_match=True,
            ),
        ]
    )

    assert summary == ComparisonSummary(
        total=3,
        matches=1,
        normalized_matches=2,
        semantic_matches=2,
        close_matches=3,
        judge_matches=1,
        judged_prompts=1,
        mean_semantic_similarity=(1.0 + 0.98 + 0.31) / 3,
    )


def test_serialize_comparisons_includes_summary() -> None:
    lines = serialize_comparisons(
        [
            PromptComparison(
                prompt="A",
                answer_a="x",
                answer_b="x",
                match=True,
                normalized_match=True,
                semantic_similarity=1.0,
                semantic_match=True,
                judge_match=None,
                close_match=True,
            ),
            PromptComparison(
                prompt="B",
                answer_a="x",
                answer_b="y",
                match=False,
                normalized_match=False,
                semantic_similarity=0.2,
                semantic_match=False,
                judge_match=False,
                close_match=False,
            ),
        ],
    )

    assert json.loads(lines[0]) == {
        "prompt": "A",
        "answer_a": "x",
        "answer_b": "x",
        "match": True,
        "normalized_match": True,
        "semantic_similarity": 1.0,
        "semantic_match": True,
        "judge_match": None,
        "close_match": True,
    }
    assert json.loads(lines[-1]) == {
        "summary": {
            "total": 2,
            "matches": 1,
            "normalized_matches": 1,
            "semantic_matches": 1,
            "close_matches": 1,
            "judge_matches": 0,
            "judged_prompts": 1,
            "mean_semantic_similarity": 0.6,
        }
    }
