from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tokcleanse.cli import app
from tokcleanse.compare import PromptComparison

_RUNNER = CliRunner()


def test_help_lists_available_orders() -> None:
    result = _RUNNER.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "download" in result.stdout
    assert "sanitize" in result.stdout
    assert "compare" in result.stdout

    sanitize_help = _RUNNER.invoke(app, ["sanitize", "--help"])
    assert sanitize_help.exit_code == 0
    assert "good (default)" in sanitize_help.stdout
    assert "original:" in sanitize_help.stdout


def test_sanitize_command_writes_reordered_directory(
    sample_tokenizer_dir: Path,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "cli-saved-tokenizer"

    result = _RUNNER.invoke(
        app,
        [
            "sanitize",
            str(sample_tokenizer_dir),
            str(destination),
            "--order",
            "good",
        ],
    )

    assert result.exit_code == 0
    assert destination.resolve().as_posix() in result.stdout

    tokenizer_data = json.loads((destination / "tokenizer.json").read_text(encoding="utf-8"))
    assert tokenizer_data["model"]["merges"][:2] == [["a", "b"], ["x", "y"]]


def test_sanitize_command_with_reassign_writes_mapping(
    sample_tokenizer_dir: Path,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "cli-reassigned-tokenizer"

    result = _RUNNER.invoke(
        app,
        [
            "sanitize",
            str(sample_tokenizer_dir),
            str(destination),
            "--order",
            "good",
            "--reassign",
        ],
    )

    assert result.exit_code == 0
    tokenizer_data = json.loads((destination / "tokenizer.json").read_text(encoding="utf-8"))
    assert tokenizer_data["model"]["vocab"]["<pad>"] == 0
    assert tokenizer_data["model"]["vocab"]["ab"] == 7
    assert tokenizer_data["model"]["vocab"]["xy"] == 8

    mapping_data = json.loads((destination / "tokenizer_mapping.json").read_text(encoding="utf-8"))
    assert mapping_data["8"] == 7
    assert mapping_data["7"] == 8


def test_sanitize_command_rejects_special_token_map_without_reassign(
    tmp_path: Path,
) -> None:
    special_token_map_path = tmp_path / "specials.json"
    special_token_map_path.write_text(json.dumps({"<pad>": None}) + "\n", encoding="utf-8")

    result = _RUNNER.invoke(
        app,
        [
            "sanitize",
            "source",
            str(tmp_path / "destination"),
            "--special-token-map-file",
            str(special_token_map_path),
        ],
    )

    assert result.exit_code != 0
    assert "--special-token-map-file requires --reassign" in result.stderr


def test_download_command_downloads_into_models_repo_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_path = (tmp_path / "models" / "google" / "gemma-4-E2B-it").resolve()

    def fake_download_model_snapshot(
        repo_id: str,
        *,
        models_dir: Path,
        force_download: bool,
    ) -> Path:
        assert repo_id == "google/gemma-4-E2B-it"
        assert models_dir == tmp_path / "models"
        assert force_download is True
        return expected_path

    monkeypatch.setattr("tokcleanse.cli.download_model_snapshot", fake_download_model_snapshot)

    result = _RUNNER.invoke(
        app,
        [
            "download",
            "google/gemma-4-E2B-it",
            "--models-dir",
            str(tmp_path / "models"),
            "--force-download",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == expected_path.as_posix()


def test_compare_command_outputs_answers_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(json.dumps({"text": "file prompt"}) + "\n", encoding="utf-8")

    def fake_compare_models(
        model_a: str,
        model_b: str,
        *,
        prompts: list[str] | tuple[str, ...],
        completion: bool,
        max_new_tokens: int,
        batch_size: int,
        judge_batch_size: int | None,
        device: str,
        dtype: str,
        semantic_model_name: str,
        semantic_threshold: float,
        llm_judge: str | None,
        judge_all: bool,
    ) -> tuple[PromptComparison, ...]:
        assert model_a == "model-a"
        assert model_b == "model-b"
        assert prompts == ("file prompt",)
        assert completion is True
        assert max_new_tokens == 16
        assert batch_size == 4
        assert judge_batch_size == 2
        assert device == "cpu"
        assert dtype == "float32"
        assert semantic_model_name == "semantic-model"
        assert semantic_threshold == 0.9
        assert llm_judge == "judge-model"
        assert judge_all is True
        return (
            PromptComparison(
                prompt="file prompt",
                answer_a="answer a",
                answer_b="answer b",
                match=False,
                normalized_match=False,
                semantic_similarity=0.88,
                semantic_match=False,
                judge_match=True,
                close_match=True,
            ),
        )

    monkeypatch.setattr("tokcleanse.cli.compare_models", fake_compare_models)
    monkeypatch.setattr("tokcleanse.cli.resolve_compare_device", lambda device: "cpu")

    result = _RUNNER.invoke(
        app,
        [
            "compare",
            "model-a",
            "model-b",
            "--prompt",
            "inline prompt",
            "--prompt-file",
            str(prompt_file),
            "--completion",
            "--max-new-tokens",
            "16",
            "--batch-size",
            "4",
            "--judge-batch-size",
            "2",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--semantic-model",
            "semantic-model",
            "--semantic-threshold",
            "0.9",
            "--llm-judge",
            "judge-model",
            "--judge-all",
        ],
    )

    assert result.exit_code == 0
    assert "Using device: cpu" in result.stderr
    output_lines = result.stdout.strip().splitlines()
    assert json.loads(output_lines[0]) == {
        "prompt": "file prompt",
        "answer_a": "answer a",
        "answer_b": "answer b",
        "match": False,
        "normalized_match": False,
        "semantic_similarity": 0.88,
        "semantic_match": False,
        "judge_match": True,
        "close_match": True,
    }
    assert json.loads(output_lines[1]) == {
        "summary": {
            "total": 1,
            "matches": 0,
            "normalized_matches": 0,
            "semantic_matches": 0,
            "close_matches": 1,
            "judge_matches": 1,
            "judged_prompts": 1,
            "mean_semantic_similarity": 0.88,
        }
    }


def test_compare_command_output_file_writes_prompt_rows_and_prints_only_summary_for_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_file = tmp_path / "compare.jsonl"

    def fake_compare_models(
        model_a: str,
        model_b: str,
        *,
        prompts: list[str] | tuple[str, ...],
        completion: bool,
        max_new_tokens: int,
        batch_size: int,
        judge_batch_size: int | None,
        device: str,
        dtype: str,
        semantic_model_name: str,
        semantic_threshold: float,
        llm_judge: str | None,
        judge_all: bool,
    ) -> tuple[PromptComparison, ...]:
        assert dtype == "auto"
        return (
            PromptComparison(
                prompt="Who are you?",
                answer_a="A",
                answer_b="A",
                match=True,
                normalized_match=True,
                semantic_similarity=1.0,
                semantic_match=True,
                judge_match=None,
                close_match=True,
            ),
        )

    monkeypatch.setattr("tokcleanse.cli.compare_models", fake_compare_models)
    monkeypatch.setattr("tokcleanse.cli.resolve_compare_device", lambda device: "cpu")

    result = _RUNNER.invoke(
        app,
        ["compare", "model-a", "model-b", "--output-file", str(output_file)],
    )

    assert result.exit_code == 0
    output_lines = result.stdout.strip().splitlines()
    assert len(output_lines) == 1
    assert json.loads(output_lines[0]) == {
        "summary": {
            "total": 1,
            "matches": 1,
            "normalized_matches": 1,
            "semantic_matches": 1,
            "close_matches": 1,
            "judge_matches": 0,
            "judged_prompts": 0,
            "mean_semantic_similarity": 1.0,
        }
    }
    assert output_file.read_text(encoding="utf-8").splitlines() == [
        json.dumps(
            {
                "prompt": "Who are you?",
                "answer_a": "A",
                "answer_b": "A",
                "match": True,
                "normalized_match": True,
                "semantic_similarity": 1.0,
                "semantic_match": True,
                "judge_match": None,
                "close_match": True,
            }
        )
    ]


def test_compare_command_output_file_prints_mismatch_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_file = tmp_path / "compare.jsonl"

    def fake_compare_models(
        model_a: str,
        model_b: str,
        *,
        prompts: list[str] | tuple[str, ...],
        completion: bool,
        max_new_tokens: int,
        batch_size: int,
        judge_batch_size: int | None,
        device: str,
        dtype: str,
        semantic_model_name: str,
        semantic_threshold: float,
        llm_judge: str | None,
        judge_all: bool,
    ) -> tuple[PromptComparison, ...]:
        assert dtype == "auto"
        return (
            PromptComparison(
                prompt="Who are you?",
                answer_a="A",
                answer_b="B",
                match=False,
                normalized_match=False,
                semantic_similarity=0.2,
                semantic_match=False,
                judge_match=False,
                close_match=False,
            ),
        )

    monkeypatch.setattr("tokcleanse.cli.compare_models", fake_compare_models)
    monkeypatch.setattr("tokcleanse.cli.resolve_compare_device", lambda device: "cpu")

    result = _RUNNER.invoke(
        app,
        ["compare", "model-a", "model-b", "--output-file", str(output_file)],
    )

    assert result.exit_code == 0
    output_lines = result.stdout.strip().splitlines()
    assert len(output_lines) == 2
    assert json.loads(output_lines[0]) == {
        "prompt": "Who are you?",
        "answer_a": "A",
        "answer_b": "B",
        "match": False,
        "normalized_match": False,
        "semantic_similarity": 0.2,
        "semantic_match": False,
        "judge_match": False,
        "close_match": False,
    }
    assert json.loads(output_lines[1]) == {
        "summary": {
            "total": 1,
            "matches": 0,
            "normalized_matches": 0,
            "semantic_matches": 0,
            "close_matches": 0,
            "judge_matches": 0,
            "judged_prompts": 1,
            "mean_semantic_similarity": 0.2,
        }
    }
    assert output_file.read_text(encoding="utf-8").splitlines() == [
        json.dumps(
            {
                "prompt": "Who are you?",
                "answer_a": "A",
                "answer_b": "B",
                "match": False,
                "normalized_match": False,
                "semantic_similarity": 0.2,
                "semantic_match": False,
                "judge_match": False,
                "close_match": False,
            }
        )
    ]


def test_compare_command_prompt_file_ignores_inline_prompts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(json.dumps({"text": "file prompt"}) + "\n", encoding="utf-8")

    def fake_compare_models(
        model_a: str,
        model_b: str,
        *,
        prompts: list[str] | tuple[str, ...],
        completion: bool,
        max_new_tokens: int,
        batch_size: int,
        judge_batch_size: int | None,
        device: str,
        dtype: str,
        semantic_model_name: str,
        semantic_threshold: float,
        llm_judge: str | None,
        judge_all: bool,
    ) -> tuple[PromptComparison, ...]:
        assert prompts == ("file prompt",)
        assert batch_size == 8
        assert judge_batch_size is None
        assert dtype == "auto"
        assert judge_all is False
        return (
            PromptComparison(
                prompt="file prompt",
                answer_a="A",
                answer_b="A",
                match=True,
                normalized_match=True,
                semantic_similarity=1.0,
                semantic_match=True,
                judge_match=None,
                close_match=True,
            ),
        )

    monkeypatch.setattr("tokcleanse.cli.compare_models", fake_compare_models)
    monkeypatch.setattr("tokcleanse.cli.resolve_compare_device", lambda device: "cpu")

    result = _RUNNER.invoke(
        app,
        [
            "compare",
            "model-a",
            "model-b",
            "--prompt",
            "inline prompt",
            "--prompt-file",
            str(prompt_file),
        ],
    )

    assert result.exit_code == 0
    assert "Using device: cpu" in result.stderr


def test_compare_command_uses_default_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_compare_models(
        model_a: str,
        model_b: str,
        *,
        prompts: list[str] | tuple[str, ...],
        completion: bool,
        max_new_tokens: int,
        batch_size: int,
        judge_batch_size: int | None,
        device: str,
        dtype: str,
        semantic_model_name: str,
        semantic_threshold: float,
        llm_judge: str | None,
        judge_all: bool,
    ) -> tuple[PromptComparison, ...]:
        assert model_a == "model-a"
        assert model_b == "model-b"
        assert prompts == ("Who are you?",)
        assert completion is False
        assert max_new_tokens == 128
        assert batch_size == 8
        assert judge_batch_size is None
        assert device == "cpu"
        assert dtype == "auto"
        assert semantic_model_name
        assert semantic_threshold > 0.0
        assert llm_judge is None
        assert judge_all is False
        return (
            PromptComparison(
                prompt="Who are you?",
                answer_a="A",
                answer_b="B",
                match=False,
                normalized_match=False,
                semantic_similarity=0.2,
                semantic_match=False,
                judge_match=None,
                close_match=False,
            ),
        )

    monkeypatch.setattr("tokcleanse.cli.compare_models", fake_compare_models)
    monkeypatch.setattr("tokcleanse.cli.resolve_compare_device", lambda device: "cpu")

    result = _RUNNER.invoke(app, ["compare", "model-a", "model-b"])

    assert result.exit_code == 0
    assert "Using device: cpu" in result.stderr


def test_compare_command_rejects_judge_all_without_llm_judge() -> None:
    result = _RUNNER.invoke(app, ["compare", "model-a", "model-b", "--judge-all"])

    assert result.exit_code != 0
    assert "--judge-all requires --llm-judge" in result.stderr


def test_compare_command_rejects_invalid_batch_size() -> None:
    result = _RUNNER.invoke(app, ["compare", "model-a", "model-b", "--batch-size", "0"])

    assert result.exit_code != 0
    assert "--batch-size must be at least 1" in result.stderr


def test_compare_command_rejects_invalid_judge_batch_size() -> None:
    result = _RUNNER.invoke(app, ["compare", "model-a", "model-b", "--judge-batch-size", "0"])

    assert result.exit_code != 0
    assert "--judge-batch-size must be at least 1" in result.stderr
