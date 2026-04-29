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
    assert "train" in result.stdout
    assert "upcycle" in result.stdout

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
    assert "Sanitize run: order=good, reassign=yes, strip_multimodal=no" in result.stderr
    assert "Artifacts: mapping=yes, token_groups=yes, added_initializers=no" in result.stderr
    tokenizer_data = json.loads((destination / "tokenizer.json").read_text(encoding="utf-8"))
    assert tokenizer_data["model"]["vocab"]["<pad>"] == 0
    assert tokenizer_data["model"]["vocab"]["ab"] == 7
    assert tokenizer_data["model"]["vocab"]["xy"] == 8
    assert tokenizer_data["model"]["vocab"]["a"] == 1
    assert tokenizer_data["model"]["vocab"]["x"] == 4

    mapping_data = json.loads((destination / "tokenizer_mapping.json").read_text(encoding="utf-8"))
    assert mapping_data["8"] == 7
    assert mapping_data["7"] == 8
    assert mapping_data["6"] == 6
    assert mapping_data["0"] == 0
    assert mapping_data["3"] == 1
    assert mapping_data["1"] == 4


def test_sanitize_command_verbose_prints_detail_sets(
    sample_tokenizer_dir: Path,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "cli-verbose-tokenizer"
    token_map_path = tmp_path / "tokens.json"
    special_token_map_path = tmp_path / "specials.json"
    token_map_path.write_text(
        json.dumps({"delete": ["ab"], "add": ["a", "ayz"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    special_token_map_path.write_text(
        json.dumps({"keep": ["<pad>"], "rename": {}}, indent=2) + "\n",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        [
            "sanitize",
            str(sample_tokenizer_dir),
            str(destination),
            "--reassign",
            "--token-map",
            str(token_map_path),
            "--special-token-map",
            str(special_token_map_path),
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert "warning" not in result.stderr.lower()
    assert "Sanitize run: order=good, reassign=yes, strip_multimodal=no" in result.stderr
    assert "Tokenizer: vocab_size=11, merge_count=4" in result.stderr
    assert "Artifacts: mapping=yes, token_groups=yes, added_initializers=yes" in result.stderr
    assert "Special tokens: kept=1, renamed=0, dropped=0" in result.stderr
    assert "Tokens: delete_requested=1, rename_requested=0, deleted_total=2, add_requested=2, existing_ignored=1, requested_added=1, intermediate_added=1" in result.stderr
    assert "Merges: added=2, deleted=2" in result.stderr
    assert 'Requested token deletes: ["ab"]' in result.stderr
    assert 'Requested token renames: {}' in result.stderr
    assert 'Deleted tokens: ["ab", "abc"]' in result.stderr
    assert 'Requested token adds: ["a", "ayz"]' in result.stderr
    assert 'Existing add tokens ignored: ["a"]' in result.stderr
    assert 'Requested added tokens: ["ayz"]' in result.stderr
    assert 'Intermediate added tokens: ["yz"]' in result.stderr
    assert 'Added merges: [["y", "z"], ["a", "yz"]]' in result.stderr
    assert 'Kept special tokens: ["<pad>"]' in result.stderr


def test_sanitize_command_rejects_special_token_map_without_reassign(
    tmp_path: Path,
) -> None:
    special_token_map_path = tmp_path / "specials.json"
    special_token_map_path.write_text(
        json.dumps({"keep": ["<pad>"], "rename": {}}, indent=2) + "\n",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        [
            "sanitize",
            "source",
            str(tmp_path / "destination"),
            "--special-token-map",
            str(special_token_map_path),
        ],
    )

    assert result.exit_code != 0
    assert "--special-token-map requires --reassign" in result.stderr


def test_sanitize_command_rejects_token_map_without_reassign(tmp_path: Path) -> None:
    token_map_path = tmp_path / "tokens.json"
    token_map_path.write_text(
        json.dumps({"delete": ["ab"], "add": ["newtok"]}, indent=2) + "\n",
        encoding="utf-8",
    )

    result = _RUNNER.invoke(
        app,
        [
            "sanitize",
            "source",
            str(tmp_path / "destination"),
            "--token-map",
            str(token_map_path),
        ],
    )

    assert result.exit_code != 0
    assert "--token-map requires --reassign" in result.stderr


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


def test_upcycle_command_invokes_model_upcycling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_path = (tmp_path / "odin-danish-moe").resolve()
    expected_path.mkdir()
    (expected_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_text",
                "num_experts": 2,
                "top_k_experts": 1,
                "expert_intermediate_size": 6144,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (expected_path / "model.safetensors").write_text("", encoding="utf-8")

    def fake_upcycle_gemma4_model(
        source: str,
        destination: Path,
        *,
        models_dir: Path,
        overwrite: bool,
        num_experts: int,
        top_k_experts: int,
        expert_init: str,
    ) -> Path:
        assert source == "models/odin-danish"
        assert destination == tmp_path / "odin-danish-moe"
        assert models_dir == tmp_path / "models"
        assert overwrite is True
        assert num_experts == 2
        assert top_k_experts == 1
        assert expert_init == "zero"
        return expected_path

    monkeypatch.setattr("tokcleanse.cli.upcycle_gemma4_model", fake_upcycle_gemma4_model)

    result = _RUNNER.invoke(
        app,
        [
            "upcycle",
            "models/odin-danish",
            str(tmp_path / "odin-danish-moe"),
            "--models-dir",
            str(tmp_path / "models"),
            "--overwrite",
            "--num-experts",
            "2",
            "--top-k-experts",
            "1",
            "--expert-init",
            "zero",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == expected_path.as_posix()
    assert "MoE config: experts=2, top_k=1, expert_intermediate_size=6144, init=zero" in result.stderr


def test_upcycle_command_verbose_prints_artifact_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_path = (tmp_path / "odin-danish-moe").resolve()
    expected_path.mkdir()
    (expected_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_text",
                "num_experts": 1,
                "top_k_experts": 1,
                "expert_intermediate_size": 12288,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (expected_path / "model.safetensors").write_text("", encoding="utf-8")
    (expected_path / "model_upcycle_plan.yaml").write_text("plan\n", encoding="utf-8")

    monkeypatch.setattr(
        "tokcleanse.cli.upcycle_gemma4_model",
        lambda *args, **kwargs: expected_path,
    )

    result = _RUNNER.invoke(
        app,
        [
            "upcycle",
            "models/odin-danish",
            str(expected_path),
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert "Upcycle source: models/odin-danish" in result.stderr
    assert "Artifacts: plan=model_upcycle_plan.yaml, checkpoint=model.safetensors" in result.stderr


def test_train_command_invokes_distillation_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_path = (tmp_path / "odin-moe-trained").resolve()

    def fake_train_distilled_model(
        student_model: str,
        teacher_model: str,
        output_dir: Path,
        *,
        ind_files: tuple[Path, ...],
        ood_files: tuple[Path, ...],
        eval_files: tuple[Path, ...],
        device: str,
        dtype: str,
        steps: int,
        batch_size: int,
        eval_batch_size: int | None,
        weight_diff_every: int,
        gradient_accumulation_steps: int,
        gradient_checkpointing: bool,
        max_length: int,
        learning_rate: float,
        weight_decay: float,
        checkpoint_every: int,
        checkpoint_dir: Path | None,
        eval_every_steps: int,
        log_every: int,
        print_every: int,
        ind_batches_per_cycle: int,
        ood_batches_per_cycle: int,
        ind_lm_weight: float,
        ind_distill_weight: float,
        ind_route_weight: float,
        ind_route_logit_bias: float,
        ood_lm_weight: float,
        ood_distill_weight: float,
        ood_route_weight: float,
        ood_route_logit_bias: float,
        route_logit_bias_anneal_steps: int,
        router_learning_rate_multiplier: float,
        seed: int,
        overwrite: bool,
        **kwargs: object,
    ) -> Path:
        del kwargs
        assert student_model == "models/odin-moe"
        assert teacher_model == "models/odin-danish"
        assert output_dir == tmp_path / "odin-moe-trained"
        assert ind_files == (tmp_path / "ind.jsonl",)
        assert ood_files == (tmp_path / "ood.jsonl",)
        assert eval_files == (tmp_path / "eval.jsonl.gz",)
        assert device == "cpu"
        assert dtype == "float32"
        assert steps == 25
        assert batch_size == 2
        assert eval_batch_size == 3
        assert weight_diff_every == 25
        assert gradient_accumulation_steps == 4
        assert gradient_checkpointing is True
        assert max_length == 512
        assert learning_rate == 2e-4
        assert weight_decay == 0.1
        assert checkpoint_every == 50
        assert checkpoint_dir == tmp_path / "checkpoints"
        assert eval_every_steps == 1000
        assert log_every == 2
        assert print_every == 5
        assert ind_batches_per_cycle == 1
        assert ood_batches_per_cycle == 2
        assert ind_lm_weight == 0.05
        assert ind_distill_weight == 1.0
        assert ind_route_weight == 0.0
        assert ind_route_logit_bias == 0.0
        assert ood_lm_weight == 1.0
        assert ood_distill_weight == 0.3
        assert ood_route_weight == 0.0
        assert ood_route_logit_bias == 0.0
        assert route_logit_bias_anneal_steps == 0
        assert router_learning_rate_multiplier == 1.0
        assert seed == 7
        assert overwrite is True
        return expected_path

    monkeypatch.setattr("tokcleanse.cli.train_distilled_model", fake_train_distilled_model)
    monkeypatch.setattr("tokcleanse.cli.resolve_compare_device", lambda device: "cpu")

    result = _RUNNER.invoke(
        app,
        [
            "train",
            "models/odin-moe",
            "models/odin-danish",
            str(tmp_path / "odin-moe-trained"),
            "--ind-file",
            str(tmp_path / "ind.jsonl"),
            "--ood-file",
            str(tmp_path / "ood.jsonl"),
            "--eval-file",
            str(tmp_path / "eval.jsonl.gz"),
            "--steps",
            "25",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "3",
            "--gradient-accumulation",
            "4",
            "--max-length",
            "512",
            "--learning-rate",
            "2e-4",
            "--weight-decay",
            "0.1",
            "--checkpoint-every",
            "50",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--eval-every",
            "1000",
            "--log-every",
            "2",
            "--print-every",
            "5",
            "--ind-batches-per-cycle",
            "1",
            "--ood-batches-per-cycle",
            "2",
            "--ind-lm-weight",
            "0.05",
            "--ind-distill-weight",
            "1.0",
            "--ood-lm-weight",
            "1.0",
            "--ood-distill-weight",
            "0.3",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--seed",
            "7",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    assert "Using device: cpu" in result.stderr
    assert result.stdout.strip() == expected_path.as_posix()


def test_train_command_dry_run_prints_report_and_skips_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tokcleanse.cli.resolve_compare_device", lambda device: "cpu")
    monkeypatch.setattr(
        "tokcleanse.cli.build_training_dry_run_report",
        lambda *args, **kwargs: type(
            "_Report",
            (),
            {
                "resolved_device": "cpu",
                "dtype": "float32",
                "requested_steps": -1,
                "resolved_steps": 123,
                "resolved_epochs": 1,
                "resume_from": None,
                "resume_step": None,
                "batch_size": 1,
                "eval_max_batches": 32,
                "weight_diff_every": 25,
                "pad_to_max_length": None,
                "torch_compile": False,
                "distill_ind": True,
                "distill_ood": True,
                "distill_original_tokens_only": False,
                "distill_every": 1,
                "lr_warmup_steps": 0,
                "eval_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "gradient_checkpointing": True,
                "learning_rate": 5e-5,
                "weight_decay": 0.0,
                "distill_kl_vocab_chunk_size": 0,
                "ind_route_weight": 0.0,
                "ind_route_logit_bias": 0.0,
                "ood_route_weight": 0.0,
                "ood_route_logit_bias": 0.0,
                "route_logit_bias_anneal_steps": 0,
                "router_learning_rate_multiplier": 1.0,
                "checkpoint_every": 0,
                "checkpoint_dir": None,
                "masked_row_parameter_names": ("model.embed_tokens.weight", "lm_head.weight"),
                "added_token_ids": (1, 3),
                "requested_added_token_ids": (3,),
                "intermediate_added_token_ids": (1,),
                "optimizer_groups": (
                    {"weight_decay": 0.1, "parameter_names": ["model.layers.0.router.proj.weight"]},
                    {"weight_decay": 0.0, "parameter_names": ["model.embed_tokens.weight", "lm_head.weight"]},
                ),
                "trainable_parameter_names": (
                    "model.layers.0.router.proj.weight",
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                ),
            },
        )(),
    )
    monkeypatch.setattr(
        "tokcleanse.cli.train_distilled_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("train_distilled_model should not run")),
    )

    result = _RUNNER.invoke(
        app,
        [
            "train",
            "models/odin-moe",
            "models/odin-danish",
            str(tmp_path / "odin-moe-trained"),
            "--ind-file",
            str(tmp_path / "ind.jsonl"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Using device: cpu" in result.stderr
    dry_run_payload = json.loads(result.stdout)
    assert dry_run_payload["dry_run"]["resolved_steps"] == 123
    assert dry_run_payload["dry_run"]["added_token_ids"] == [1, 3]
    assert dry_run_payload["dry_run"]["gradient_accumulation"] == 8
    assert dry_run_payload["dry_run"]["gradient_checkpointing"] is True
    assert dry_run_payload["dry_run"]["checkpoint_every"] == 0
    assert dry_run_payload["dry_run"]["checkpoint_dir"] is None


def test_train_command_defaults_checkpoint_every_to_eval_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_path = tmp_path / "odin-moe-trained"

    def fake_train_distilled_model(
        student_model: str,
        teacher_model: str,
        output_dir: Path,
        *,
        ind_files: tuple[Path, ...],
        ood_files: tuple[Path, ...],
        eval_files: tuple[Path, ...],
        device: str,
        dtype: str,
        steps: int,
        batch_size: int,
        eval_batch_size: int | None,
        weight_diff_every: int,
        gradient_accumulation_steps: int,
        gradient_checkpointing: bool,
        max_length: int,
        learning_rate: float,
        weight_decay: float,
        checkpoint_every: int,
        checkpoint_dir: Path | None,
        eval_every_steps: int,
        log_every: int,
        print_every: int,
        ind_batches_per_cycle: int,
        ood_batches_per_cycle: int,
        ind_lm_weight: float,
        ind_distill_weight: float,
        ind_route_weight: float,
        ind_route_logit_bias: float,
        ood_lm_weight: float,
        ood_distill_weight: float,
        ood_route_weight: float,
        ood_route_logit_bias: float,
        route_logit_bias_anneal_steps: int,
        router_learning_rate_multiplier: float,
        seed: int,
        overwrite: bool,
        **kwargs: object,
    ) -> Path:
        del kwargs
        assert checkpoint_every == 123
        assert checkpoint_dir == Path("checkpoints")
        assert eval_every_steps == 123
        assert gradient_checkpointing is True
        return expected_path

    monkeypatch.setattr("tokcleanse.cli.train_distilled_model", fake_train_distilled_model)
    monkeypatch.setattr("tokcleanse.cli.resolve_compare_device", lambda device: "cpu")

    result = _RUNNER.invoke(
        app,
        [
            "train",
            "models/odin-moe",
            "models/odin-danish",
            str(expected_path),
            "--ind-file",
            str(tmp_path / "ind.jsonl"),
            "--eval-every",
            "123",
        ],
    )

    assert result.exit_code == 0
    assert "Using device: cpu" in result.stderr
    assert result.stdout.strip() == expected_path.as_posix()


def test_logit_diff_command_invokes_backend_and_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tokcleanse.cli.resolve_compare_device", lambda device: "cpu")
    monkeypatch.setattr(
        "tokcleanse.cli.compare_model_logits",
        lambda *args, **kwargs: ("sentinel",),
    )
    monkeypatch.setattr(
        "tokcleanse.cli.serialize_prompt_logit_diffs",
        lambda diffs: ('{"prompt":"hello","max_abs_diff":0.0}',),
    )
    monkeypatch.setattr(
        "tokcleanse.cli.serialize_logit_diff_summary",
        lambda diffs: '{"summary":{"total":1,"allclose_prompts":1}}',
    )

    result = _RUNNER.invoke(
        app,
        [
            "logit-diff",
            "models/odin-danish",
            "models/odin-moe",
            "--prompt",
            "hello",
        ],
    )

    assert result.exit_code == 0
    assert "Using device: cpu" in result.stderr
    output_lines = result.stdout.strip().splitlines()
    assert output_lines == [
        '{"prompt":"hello","max_abs_diff":0.0}',
        '{"summary":{"total":1,"allclose_prompts":1}}',
    ]
