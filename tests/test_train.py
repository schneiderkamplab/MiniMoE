from __future__ import annotations

import gzip
import json
import io
import random
from pathlib import Path
from contextlib import redirect_stdout

import pytest
import torch

from tokcleanse.train import (
    _CorpusStream,
    _LanguageScheduler,
    _RouteBiasAnnealer,
    _build_training_configuration_state,
    _build_learning_rate_scheduler,
    _capture_training_random_generator_state,
    _capture_mps_memory_snapshot,
    _compute_distillation_loss,
    _compute_expert_pair_weight_diff_metrics,
    _compute_language_model_loss_components,
    _count_corpus_examples,
    _evaluate_language_model_loss,
    _load_corpus_examples,
    _masked_row_parameter_ids,
    _print_training_metrics,
    _release_mps_working_set,
    _restore_training_random_generator_state,
    _resolve_torch_compile,
    _resolve_training_steps,
    _should_run_distillation,
    _tokenize_batch,
    _validate_resume_training_configuration,
    TokenGroupMetadata,
    TrainingParameterSelection,
    build_training_dry_run_report,
    load_token_group_metadata,
    prepare_student_for_distillation,
)


class _DummyRouter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 2, bias=False)
        self.scale = torch.nn.Parameter(torch.ones(4))
        self.per_expert_scale = torch.nn.Parameter(torch.ones(2))


class _DummyExperts(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.ones((2, 8, 4)))
        self.down_proj = torch.nn.Parameter(torch.ones((2, 4, 4)))


class _DummyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = _DummyRouter()
        self.experts = _DummyExperts()
        self.self_attn = torch.nn.Linear(4, 4, bias=False)


class _DummyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(6, 4)
        self.layers = torch.nn.ModuleList([_DummyLayer()])


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _DummyBackbone()
        self.lm_head = torch.nn.Linear(4, 6, bias=False)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head


class _DummyAutoModelFactory:
    def __init__(self, model: _DummyModel) -> None:
        self._model = model

    def from_pretrained(self, *args: object, **kwargs: object) -> _DummyModel:
        return self._model


class _FakeMpsModule:
    def __init__(self) -> None:
        self.empty_cache_calls = 0
        self.synchronize_calls = 0
        self.current_allocated_value = 3 * 1024 ** 3
        self.driver_allocated_value = 5 * 1024 ** 3
        self.recommended_max_value = 8 * 1024 ** 3

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def current_allocated_memory(self) -> int:
        return self.current_allocated_value

    def driver_allocated_memory(self) -> int:
        return self.driver_allocated_value

    def recommended_max_memory(self) -> int:
        return self.recommended_max_value


class _FakeTorchModule:
    def __init__(self) -> None:
        self.mps = _FakeMpsModule()


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, object]:
        self.calls.append({"texts": texts, **kwargs})
        return {"texts": texts, **kwargs}


class _EvalModel:
    def __init__(self) -> None:
        self.training = True
        self.forward_calls = 0

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


def test_load_token_group_metadata_reads_saved_groups(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "tokenizer_token_groups.json").write_text(
        json.dumps(
            {
                "original_token_ids": [0, 1, 2],
                "added_token_ids": [3, 4],
                "requested_added_token_ids": [4],
                "intermediate_added_token_ids": [3],
                "special_token_ids": [0],
                "added_tokens": ["yz", "ayz"],
                "requested_added_tokens": ["ayz"],
                "intermediate_added_tokens": ["yz"],
                "special_tokens": ["<pad>"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = load_token_group_metadata(model_path)

    assert metadata.original_token_ids == (0, 1, 2)
    assert metadata.added_token_ids == (3, 4)
    assert metadata.requested_added_token_ids == (4,)
    assert metadata.intermediate_added_token_ids == (3,)
    assert metadata.special_token_ids == (0,)
    assert metadata.added_tokens == ("yz", "ayz")


def test_load_token_group_metadata_defaults_when_file_is_missing(tmp_path: Path) -> None:
    metadata = load_token_group_metadata(tmp_path / "missing-model")

    assert metadata.original_token_ids == ()
    assert metadata.added_token_ids == ()
    assert metadata.special_tokens == ()


def test_prepare_student_for_distillation_freezes_shared_weights_and_masks_added_rows(
    tmp_path: Path,
) -> None:
    model = _DummyModel()
    metadata_dir = tmp_path / "model"
    metadata_dir.mkdir()
    (metadata_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (metadata_dir / "tokenizer_token_groups.json").write_text(
        json.dumps(
            {
                "original_token_ids": [0, 2, 4, 5],
                "added_token_ids": [1, 3],
                "requested_added_token_ids": [3],
                "intermediate_added_token_ids": [1],
                "special_token_ids": [0],
                "added_tokens": ["yz", "ayz"],
                "requested_added_tokens": ["ayz"],
                "intermediate_added_tokens": ["yz"],
                "special_tokens": ["<pad>"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    prepare_student_for_distillation(
        model,
        token_groups=load_token_group_metadata(metadata_dir),
        torch_module=torch,
    )

    named_parameters = dict(model.named_parameters())
    assert named_parameters["model.layers.0.experts.gate_up_proj"].requires_grad
    assert named_parameters["model.layers.0.experts.down_proj"].requires_grad
    assert named_parameters["model.layers.0.router.proj.weight"].requires_grad
    assert named_parameters["model.layers.0.router.scale"].requires_grad
    assert named_parameters["model.layers.0.router.per_expert_scale"].requires_grad
    assert not named_parameters["model.layers.0.self_attn.weight"].requires_grad
    assert named_parameters["model.embed_tokens.weight"].requires_grad
    assert named_parameters["lm_head.weight"].requires_grad

    loss = (
        model.model.embed_tokens.weight.sum()
        + model.lm_head.weight.sum()
        + model.model.layers[0].experts.gate_up_proj.sum()
        + model.model.layers[0].experts.down_proj.sum()
    )
    loss.backward()

    embedding_grad_rows = model.model.embed_tokens.weight.grad.abs().sum(dim=1).tolist()
    lm_head_grad_rows = model.lm_head.weight.grad.abs().sum(dim=1).tolist()
    assert embedding_grad_rows == [0.0, 4.0, 0.0, 4.0, 0.0, 0.0]
    assert lm_head_grad_rows == [0.0, 4.0, 0.0, 4.0, 0.0, 0.0]
    assert torch.count_nonzero(model.model.layers[0].experts.gate_up_proj.grad[0]) == 0
    assert torch.count_nonzero(model.model.layers[0].experts.down_proj.grad[0]) == 0
    assert torch.count_nonzero(model.model.layers[0].experts.gate_up_proj.grad[1]) > 0
    assert torch.count_nonzero(model.model.layers[0].experts.down_proj.grad[1]) > 0


def test_prepare_student_for_distillation_honors_parameter_selection() -> None:
    model = _DummyModel()
    token_groups = TokenGroupMetadata(
        original_token_ids=(0, 2, 4, 5),
        added_token_ids=(1, 3),
        requested_added_token_ids=(3,),
        intermediate_added_token_ids=(1,),
        special_token_ids=(0,),
        added_tokens=("yz", "ayz"),
        requested_added_tokens=("ayz",),
        intermediate_added_tokens=("yz",),
        special_tokens=("<pad>",),
    )

    prepare_student_for_distillation(
        model,
        token_groups=token_groups,
        torch_module=torch,
        parameter_selection=TrainingParameterSelection(
            shared=True,
            expert_0=True,
            expert_1=False,
            embedding_lm_head=False,
        ),
    )

    named_parameters = dict(model.named_parameters())
    assert named_parameters["model.layers.0.self_attn.weight"].requires_grad
    assert named_parameters["model.layers.0.experts.gate_up_proj"].requires_grad
    assert named_parameters["model.layers.0.experts.down_proj"].requires_grad
    assert named_parameters["model.layers.0.router.proj.weight"].requires_grad
    assert not named_parameters["model.embed_tokens.weight"].requires_grad
    assert not named_parameters["lm_head.weight"].requires_grad

    loss = (
        model.model.layers[0].experts.gate_up_proj.sum()
        + model.model.layers[0].experts.down_proj.sum()
    )
    loss.backward()

    assert torch.count_nonzero(model.model.layers[0].experts.gate_up_proj.grad[0]) > 0
    assert torch.count_nonzero(model.model.layers[0].experts.down_proj.grad[0]) > 0
    assert torch.count_nonzero(model.model.layers[0].experts.gate_up_proj.grad[1]) == 0
    assert torch.count_nonzero(model.model.layers[0].experts.down_proj.grad[1]) == 0


def test_prepare_student_for_distillation_can_train_full_embedding_lm_head() -> None:
    model = _DummyModel()
    token_groups = TokenGroupMetadata(
        original_token_ids=(0, 2, 4, 5),
        added_token_ids=(1, 3),
        requested_added_token_ids=(3,),
        intermediate_added_token_ids=(1,),
        special_token_ids=(0,),
        added_tokens=("yz", "ayz"),
        requested_added_tokens=("ayz",),
        intermediate_added_tokens=("yz",),
        special_tokens=("<pad>",),
    )

    prepare_student_for_distillation(
        model,
        token_groups=token_groups,
        torch_module=torch,
        parameter_selection=TrainingParameterSelection(
            shared=False,
            expert_0=False,
            expert_1=False,
            embedding_lm_head=True,
            full_embedding_lm_head=True,
        ),
    )

    named_parameters = dict(model.named_parameters())
    assert named_parameters["model.embed_tokens.weight"].requires_grad
    assert named_parameters["lm_head.weight"].requires_grad
    assert not named_parameters["model.layers.0.self_attn.weight"].requires_grad

    loss = model.model.embed_tokens.weight.sum() + model.lm_head.weight.sum()
    loss.backward()

    assert torch.all(named_parameters["model.embed_tokens.weight"].grad == 1)
    assert torch.all(named_parameters["lm_head.weight"].grad == 1)


def test_compute_expert_pair_weight_diff_metrics() -> None:
    model = _DummyModel()
    with torch.no_grad():
        model.model.layers[0].router.proj.weight[0].fill_(0.0)
        model.model.layers[0].router.proj.weight[1].fill_(1.0)
        model.model.layers[0].router.per_expert_scale[0].fill_(0.0)
        model.model.layers[0].router.per_expert_scale[1].fill_(2.0)
        model.model.layers[0].experts.gate_up_proj[0].fill_(1.0)
        model.model.layers[0].experts.gate_up_proj[1].fill_(3.0)
        model.model.layers[0].experts.down_proj[0].fill_(1.0)
        model.model.layers[0].experts.down_proj[1].fill_(-1.0)

    metrics = _compute_expert_pair_weight_diff_metrics(model=model, torch_module=torch)

    assert metrics["router_expert_weight_diff_mean_abs"] == pytest.approx(1.2)
    assert metrics["router_expert_weight_diff_mean_squared"] == pytest.approx(1.6)
    assert metrics["expert_weight_diff_mean_abs"] == pytest.approx(2.0)
    assert metrics["expert_weight_diff_mean_squared"] == pytest.approx(4.0)


def test_load_corpus_examples_supports_jsonl_gz(tmp_path: Path) -> None:
    corpus_path = tmp_path / "ind.jsonl.gz"
    with gzip.open(corpus_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": "first"}) + "\n")
        handle.write(json.dumps({"text": "second"}) + "\n")

    examples = _load_corpus_examples((corpus_path,))

    assert examples == ("first", "second")


def test_load_corpus_examples_supports_line_based_gz(tmp_path: Path) -> None:
    corpus_path = tmp_path / "ood.txt.gz"
    with gzip.open(corpus_path, "wt", encoding="utf-8") as handle:
        handle.write("første\n")
        handle.write("\n")
        handle.write("anden\n")

    examples = _load_corpus_examples((corpus_path,))

    assert examples == ("første", "anden")


def test_load_corpus_examples_supports_jinx(tmp_path: Path) -> None:
    dataset_writer_module = pytest.importorskip("mldataforge.jinx.dataset_writer")
    corpus_path = tmp_path / "ind.jinx"
    with dataset_writer_module.JinxDatasetWriter(str(corpus_path), shard_size=None) as writer:
        writer.write({"text": "first"})
        writer.write({"text": "second"})

    examples = _load_corpus_examples((corpus_path,))

    assert examples == ("first", "second")


def test_count_corpus_examples_uses_jinx_footer_metadata(tmp_path: Path) -> None:
    dataset_writer_module = pytest.importorskip("mldataforge.jinx.dataset_writer")
    corpus_path = tmp_path / "ind.jinx"
    with dataset_writer_module.JinxDatasetWriter(str(corpus_path), shard_size=None) as writer:
        writer.write({"text": "first"})
        writer.write({"text": "second"})
        writer.write({"text": "third"})

    count = _count_corpus_examples((corpus_path,))

    assert count == 3


def test_tokenize_batch_uses_fixed_padding_on_mps() -> None:
    tokenizer = _RecordingTokenizer()

    encoded = _tokenize_batch(
        tokenizer=tokenizer,
        batch_texts=("hej",),
        max_length=64,
        device="mps",
        pad_to_max_length=None,
    )

    assert encoded["padding"] == "max_length"
    assert tokenizer.calls[-1]["padding"] == "max_length"


def test_tokenize_batch_uses_dynamic_padding_off_mps() -> None:
    tokenizer = _RecordingTokenizer()

    encoded = _tokenize_batch(
        tokenizer=tokenizer,
        batch_texts=("hello",),
        max_length=64,
        device="cpu",
        pad_to_max_length=None,
    )

    assert encoded["padding"] is True
    assert tokenizer.calls[-1]["padding"] is True


def test_tokenize_batch_can_disable_fixed_padding_on_mps() -> None:
    tokenizer = _RecordingTokenizer()

    encoded = _tokenize_batch(
        tokenizer=tokenizer,
        batch_texts=("hej",),
        max_length=64,
        device="mps",
        pad_to_max_length=False,
    )

    assert encoded["padding"] is True
    assert tokenizer.calls[-1]["padding"] is True


def test_tokenize_batch_can_force_fixed_padding_off_mps() -> None:
    tokenizer = _RecordingTokenizer()

    encoded = _tokenize_batch(
        tokenizer=tokenizer,
        batch_texts=("hello",),
        max_length=64,
        device="cpu",
        pad_to_max_length=True,
    )

    assert encoded["padding"] == "max_length"
    assert tokenizer.calls[-1]["padding"] == "max_length"


def test_evaluate_language_model_loss_respects_max_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_path = tmp_path / "eval.txt"
    eval_path.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    model = _EvalModel()
    tokenizer = object()
    seen_batches: list[tuple[str, ...]] = []

    def fake_eval_batch_language_model_loss(**kwargs: object) -> tuple[float, float]:
        batch_texts = kwargs["batch_texts"]
        assert isinstance(batch_texts, tuple)
        seen_batches.append(batch_texts)
        return float(len(batch_texts)), float(len(batch_texts))

    monkeypatch.setattr(
        "tokcleanse.train._evaluate_batch_language_model_loss",
        fake_eval_batch_language_model_loss,
    )

    loss = _evaluate_language_model_loss(
        model=model,
        tokenizer=tokenizer,
        eval_files=(eval_path,),
        batch_size=2,
        max_batches=2,
        max_length=32,
        device="cpu",
        pad_to_max_length=False,
        torch_module=torch,
    )

    assert loss == pytest.approx(1.0)
    assert seen_batches == [("a", "b"), ("c", "d")]
    assert model.training is True


def test_evaluate_language_model_loss_uses_full_eval_set_when_max_batches_is_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_path = tmp_path / "eval.txt"
    eval_path.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    model = _EvalModel()
    tokenizer = object()
    seen_batches: list[tuple[str, ...]] = []

    def fake_eval_batch_language_model_loss(**kwargs: object) -> tuple[float, float]:
        batch_texts = kwargs["batch_texts"]
        assert isinstance(batch_texts, tuple)
        seen_batches.append(batch_texts)
        return float(len(batch_texts)), float(len(batch_texts))

    monkeypatch.setattr(
        "tokcleanse.train._evaluate_batch_language_model_loss",
        fake_eval_batch_language_model_loss,
    )

    loss = _evaluate_language_model_loss(
        model=model,
        tokenizer=tokenizer,
        eval_files=(eval_path,),
        batch_size=2,
        max_batches=0,
        max_length=32,
        device="cpu",
        pad_to_max_length=False,
        torch_module=torch,
    )

    assert loss == pytest.approx(1.0)
    assert seen_batches == [("a", "b"), ("c", "d"), ("e",)]


def test_resolve_torch_compile_defaults_to_cuda_only() -> None:
    assert _resolve_torch_compile(requested=None, device="cuda") is True
    assert _resolve_torch_compile(requested=None, device="mps") is False
    assert _resolve_torch_compile(requested=None, device="cpu") is False


def test_resolve_torch_compile_respects_explicit_override() -> None:
    assert _resolve_torch_compile(requested=True, device="mps") is True
    assert _resolve_torch_compile(requested=False, device="cuda") is False


def test_should_run_distillation_respects_language_and_cadence() -> None:
    assert _should_run_distillation(
        step=1,
        language="ind",
        distill_ind=True,
        distill_ood=True,
        distill_every=2,
        distill_weight=1.0,
    ) is False
    assert _should_run_distillation(
        step=2,
        language="ind",
        distill_ind=True,
        distill_ood=True,
        distill_every=2,
        distill_weight=1.0,
    ) is True
    assert _should_run_distillation(
        step=2,
        language="ood",
        distill_ind=True,
        distill_ood=False,
        distill_every=2,
        distill_weight=1.0,
    ) is False


def test_should_run_distillation_skips_zero_weight() -> None:
    assert _should_run_distillation(
        step=10,
        language="ood",
        distill_ind=True,
        distill_ood=True,
        distill_every=1,
        distill_weight=0.0,
    ) is False


def test_resolve_training_steps_supports_negative_epoch_counts(tmp_path: Path) -> None:
    ind_path = tmp_path / "ind.txt"
    ood_path = tmp_path / "ood.txt"
    ind_path.write_text("\n".join(f"ind-{index}" for index in range(10)) + "\n", encoding="utf-8")
    ood_path.write_text("\n".join(f"ood-{index}" for index in range(4)) + "\n", encoding="utf-8")
    scheduler = _LanguageScheduler(
        ind_enabled=True,
        ood_enabled=True,
        ind_batches_per_cycle=1,
        ood_batches_per_cycle=1,
    )

    resolved_steps, resolved_epochs = _resolve_training_steps(
        requested_steps=-2,
        batch_size=1,
        gradient_accumulation_steps=8,
        ind_files=(ind_path,),
        ood_files=(ood_path,),
        scheduler=scheduler,
    )

    assert resolved_steps == 5
    assert resolved_epochs == 2


def test_masked_row_parameter_ids_selects_embedding_and_lm_head_weights() -> None:
    model = _DummyModel()

    parameter_ids = _masked_row_parameter_ids(model)

    assert parameter_ids == frozenset(
        {
            id(model.model.embed_tokens.weight),
            id(model.lm_head.weight),
        }
    )


def test_build_training_dry_run_report_lists_trainable_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    metadata_dir = tmp_path / "model"
    metadata_dir.mkdir()
    (metadata_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (metadata_dir / "tokenizer_token_groups.json").write_text(
        json.dumps(
            {
                "original_token_ids": [0, 2, 4, 5],
                "added_token_ids": [1, 3],
                "requested_added_token_ids": [3],
                "intermediate_added_token_ids": [1],
                "special_token_ids": [0],
                "added_tokens": ["yz", "ayz"],
                "requested_added_tokens": ["ayz"],
                "intermediate_added_tokens": ["yz"],
                "special_tokens": ["<pad>"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ind_path = tmp_path / "ind.txt"
    ind_path.write_text("a\nb\n", encoding="utf-8")
    model = _DummyModel()
    monkeypatch.setattr(
        transformers,
        "AutoModelForCausalLM",
        _DummyAutoModelFactory(model),
    )

    report = build_training_dry_run_report(
        metadata_dir,
        ind_files=(ind_path,),
        steps=-1,
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        weight_decay=0.1,
    )

    assert report.added_token_ids == (1, 3)
    assert report.gradient_checkpointing is True
    assert report.checkpoint_every == 0
    assert report.checkpoint_dir is None
    assert report.lr_warmup_steps == 0
    assert "model.embed_tokens.weight" in report.masked_row_parameter_names
    assert "lm_head.weight" in report.masked_row_parameter_names
    assert "model.layers.0.experts.gate_up_proj" in report.trainable_parameter_names
    assert any(group["weight_decay"] == 0.0 for group in report.optimizer_groups)
    assert any(group["weight_decay"] == 0.1 for group in report.optimizer_groups)


def test_build_training_dry_run_report_requires_checkpoint_dir_for_periodic_saves(
    tmp_path: Path,
) -> None:
    ind_path = tmp_path / "ind.txt"
    ind_path.write_text("a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="--checkpoint-dir is required"):
        build_training_dry_run_report(
            tmp_path / "model",
            ind_files=(ind_path,),
            checkpoint_every=10,
        )


def test_print_training_metrics_includes_mix_and_per_language_losses() -> None:
    output = io.StringIO()

    with redirect_stdout(output):
        _print_training_metrics(
            {
                "step": 10,
                "mix": "4ind/4ood",
                "train_loss": 5.0,
                "train_lm_loss": 6.0,
                "train_distill_loss": 3.0,
                "learning_rate": 5e-5,
                "ind_train_loss": 4.5,
                "ood_train_loss": 5.5,
                "eval_lm_loss": None,
                "latest_eval_lm_loss": 7.0,
            }
        )

    assert output.getvalue().strip() == (
        "step=10 mix=4ind/4ood train_loss=5.0000 train_lm_loss=6.0000 "
        "train_distill_loss=3.0000 lr=5.00e-05 ind_train_loss=4.5000 ood_train_loss=5.5000 "
        "latest_eval_lm_loss=7.0000"
    )


def test_learning_rate_scheduler_warms_up_then_decays() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = _build_learning_rate_scheduler(
        optimizer=optimizer,
        warmup_steps=2,
        total_steps=6,
        min_learning_rate=1e-8,
        base_learning_rate=1.0,
        torch_module=torch,
    )

    learning_rates = []
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])

    assert learning_rates[0] == pytest.approx(1.0)
    assert learning_rates[1] < learning_rates[0]
    assert learning_rates[1] > learning_rates[2] > learning_rates[3]
    assert learning_rates[-1] == pytest.approx(1e-8, rel=1e-6, abs=1e-8)


def test_learning_rate_scheduler_state_dict_restores_progress() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = _build_learning_rate_scheduler(
        optimizer=optimizer,
        warmup_steps=2,
        total_steps=6,
        min_learning_rate=1e-8,
        base_learning_rate=1.0,
        torch_module=torch,
    )
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    saved_lr = optimizer.param_groups[0]["lr"]
    state_dict = scheduler.state_dict()

    restored_parameter = torch.nn.Parameter(torch.tensor(1.0))
    restored_optimizer = torch.optim.AdamW([restored_parameter], lr=1.0)
    restored_scheduler = _build_learning_rate_scheduler(
        optimizer=restored_optimizer,
        warmup_steps=2,
        total_steps=6,
        min_learning_rate=1e-8,
        base_learning_rate=1.0,
        torch_module=torch,
    )
    restored_scheduler.load_state_dict(state_dict)

    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(saved_lr)


def test_corpus_stream_state_dict_restores_position(tmp_path: Path) -> None:
    first_path = tmp_path / "a.txt"
    second_path = tmp_path / "b.txt"
    first_path.write_text("a1\na2\na3\n", encoding="utf-8")
    second_path.write_text("b1\nb2\nb3\n", encoding="utf-8")

    stream = _CorpusStream((first_path, second_path), rng=random.Random(0))
    consumed = stream.next_batch(4)
    state_dict = stream.state_dict()
    expected_next = stream.next_batch(4)
    stream.close()

    restored = _CorpusStream((first_path, second_path), rng=random.Random(123))
    restored.load_state_dict(state_dict)
    actual_next = restored.next_batch(4)
    restored.close()

    assert consumed
    assert actual_next == expected_next


def test_training_random_generator_state_restores_python_and_torch() -> None:
    random.seed(123)
    torch.manual_seed(123)
    state_dict = _capture_training_random_generator_state(torch_module=torch)
    expected_python_values = [random.random() for _ in range(3)]
    expected_torch_values = torch.rand(3)

    random.seed(999)
    torch.manual_seed(999)
    _restore_training_random_generator_state(
        resume_state={"random_generator_state_dict": state_dict},
        torch_module=torch,
    )

    assert [random.random() for _ in range(3)] == expected_python_values
    assert torch.equal(torch.rand(3), expected_torch_values)


def test_route_bias_annealer_pauses_above_threshold_and_advances_below() -> None:
    annealer = _RouteBiasAnnealer(anneal_steps=4, loss_threshold=0.05)

    assert annealer.scale() == pytest.approx(1.0)
    assert annealer.observe_route_loss(0.051) is False
    assert annealer.completed_anneal_steps == 0
    assert annealer.scale() == pytest.approx(1.0)

    assert annealer.observe_route_loss(0.05) is True
    assert annealer.completed_anneal_steps == 1
    assert annealer.scale() == pytest.approx(0.75)

    state = annealer.state_dict()
    restored = _RouteBiasAnnealer(anneal_steps=4, loss_threshold=0.05)
    restored.load_state_dict(state)
    assert restored.completed_anneal_steps == 1
    assert restored.scale() == pytest.approx(0.75)


def test_resume_training_configuration_rejects_different_anneal_steps(tmp_path: Path) -> None:
    configuration = _build_training_configuration_state(
        resolved_steps=100,
        resolved_epochs=None,
        teacher_model=tmp_path / "teacher",
        ind_files=(tmp_path / "ind.txt",),
        ood_files=(tmp_path / "ood.txt",),
        eval_files=(),
        resolved_device="cpu",
        dtype="float32",
        batch_size=4,
        eval_batch_size=4,
        eval_max_batches=32,
        weight_diff_every=25,
        pad_to_max_length=True,
        torch_compile=False,
        distill_ind=False,
        distill_ood=False,
        distill_original_tokens_only=False,
        distill_every=1,
        lr_warmup_steps=25,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        max_length=256,
        learning_rate=1e-3,
        weight_decay=0.0,
        distill_kl_vocab_chunk_size=0,
        checkpoint_every=25,
        checkpoint_dir=tmp_path / "checkpoints",
        eval_every_steps=25,
        log_every=1,
        print_every=1,
        ind_batches_per_cycle=1,
        ood_batches_per_cycle=1,
        ind_lm_weight=1.0,
        ind_distill_weight=0.0,
        ind_route_weight=0.02,
        ind_route_logit_bias=-1.25,
        ood_lm_weight=1.0,
        ood_distill_weight=0.0,
        ood_route_weight=0.02,
        ood_route_logit_bias=1.25,
        route_logit_bias_anneal_steps=750,
        route_logit_bias_anneal_loss_threshold=5e-2,
        router_learning_rate_multiplier=1.0,
        parameter_selection=TrainingParameterSelection(
            shared=True,
            expert_0=True,
            expert_1=True,
            embedding_lm_head=True,
            full_embedding_lm_head=True,
        ),
        seed=0,
    )
    changed_configuration = dict(configuration)
    changed_configuration["route_logit_bias_anneal_steps"] = 500

    _validate_resume_training_configuration(
        resume_state={"training_configuration": configuration},
        training_configuration_state=configuration,
    )
    with pytest.raises(ValueError, match="route_logit_bias_anneal_steps"):
        _validate_resume_training_configuration(
            resume_state={"training_configuration": configuration},
            training_configuration_state=changed_configuration,
        )


def test_print_training_metrics_includes_timing_fields_when_present() -> None:
    output = io.StringIO()

    with redirect_stdout(output):
        _print_training_metrics(
            {
                "step": 12,
                "mix": "4ind/4ood",
                "train_loss": 5.0,
                "train_lm_loss": 6.0,
                "train_distill_loss": 3.0,
                "learning_rate": 5e-5,
                "ind_train_loss": 4.5,
                "ood_train_loss": 5.5,
                "latest_eval_lm_loss": 7.0,
                "step_total_s": 9.876,
                "data_s": 0.12,
                "tokenize_s": 0.34,
                "move_s": 0.56,
                "student_forward_s": 1.23,
                "teacher_forward_s": 2.34,
                "lm_loss_s": 0.45,
                "distill_loss_s": 0.67,
                "backward_s": 3.21,
                "optimizer_s": 0.89,
                "eval_s": 4.56,
            }
        )

    assert output.getvalue().strip() == (
        "step=12 mix=4ind/4ood train_loss=5.0000 train_lm_loss=6.0000 "
        "train_distill_loss=3.0000 lr=5.00e-05 ind_train_loss=4.5000 ood_train_loss=5.5000 "
        "step_s=9.88s data_s=0.12s tok_s=0.34s move_s=0.56s stu_s=1.23s "
        "tea_s=2.34s lm_s=0.45s kl_s=0.67s back_s=3.21s opt_s=0.89s eval_s=4.56s "
        "latest_eval_lm_loss=7.0000"
    )


def test_release_mps_working_set_synchronizes_and_clears_mps_cache() -> None:
    torch_module = _FakeTorchModule()

    _release_mps_working_set(
        torch_module=torch_module,
        device="mps",
    )

    assert torch_module.mps.synchronize_calls == 1
    assert torch_module.mps.empty_cache_calls == 1


def test_release_mps_working_set_skips_non_mps_devices() -> None:
    torch_module = _FakeTorchModule()

    _release_mps_working_set(
        torch_module=torch_module,
        device="cpu",
    )

    assert torch_module.mps.synchronize_calls == 0
    assert torch_module.mps.empty_cache_calls == 0


def test_capture_mps_memory_snapshot_reads_mps_counters() -> None:
    torch_module = _FakeTorchModule()

    snapshot = _capture_mps_memory_snapshot(
        torch_module=torch_module,
        device="mps",
    )

    assert snapshot is not None
    assert snapshot.allocated_bytes == 3 * 1024 ** 3
    assert snapshot.driver_allocated_bytes == 5 * 1024 ** 3
    assert snapshot.recommended_max_bytes == 8 * 1024 ** 3


def test_compute_language_model_loss_components_matches_cross_entropy() -> None:
    logits = torch.tensor(
        [
            [
                [1.0, 0.5, -0.5],
                [0.2, 1.3, -0.7],
                [0.1, -0.2, 0.4],
                [0.9, -0.1, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[0, 1, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    loss_sum, token_count = _compute_language_model_loss_components(
        logits=logits,
        input_ids=input_ids,
        attention_mask=attention_mask,
        torch_module=torch,
    )

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].to(dtype=logits.dtype)
    expected_per_token = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
    ).reshape_as(shift_labels)
    expected_loss_sum = (expected_per_token * shift_mask).sum()
    expected_token_count = shift_mask.sum()

    assert torch.allclose(loss_sum, expected_loss_sum)
    assert torch.allclose(token_count, expected_token_count)


def test_compute_distillation_loss_matches_kl_div_reference() -> None:
    student_logits = torch.tensor(
        [
            [
                [0.1, 0.3, -0.5, 1.2],
                [0.7, -0.4, 0.6, 0.2],
                [-0.1, 0.9, 0.5, -0.3],
                [0.2, 0.0, -0.8, 0.4],
            ]
        ],
        dtype=torch.float32,
    )
    teacher_logits = torch.tensor(
        [
            [
                [0.4, -0.2, 0.1, 0.8],
                [0.3, 0.5, -0.6, 0.1],
                [0.0, 0.7, 0.2, -0.4],
                [-0.3, 0.6, 0.4, 0.2],
            ]
        ],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 0, 0]])

    actual = _compute_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        attention_mask=attention_mask,
        vocab_index=None,
        vocab_chunk_size=4096,
        torch_module=torch,
    )

    shift_student = student_logits[:, :-1, :]
    shift_teacher = teacher_logits[:, :-1, :]
    shift_mask = attention_mask[:, 1:].to(dtype=student_logits.dtype)
    expected_per_token = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(shift_student, dim=-1),
        torch.nn.functional.softmax(shift_teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    expected = (expected_per_token * shift_mask).sum() / shift_mask.sum().clamp_min(1.0)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_print_training_metrics_includes_mps_memory_fields() -> None:
    output = io.StringIO()

    with redirect_stdout(output):
        _print_training_metrics(
            {
                "step": 11,
                "mix": "4ind/4ood",
                "train_loss": 5.0,
                "train_lm_loss": 6.0,
                "train_distill_loss": 3.0,
                "learning_rate": 5e-5,
                "ind_train_loss": 4.5,
                "ood_train_loss": 5.5,
                "eval_lm_loss": None,
                "latest_eval_lm_loss": 7.0,
                "mps_allocated_gib": 3.0,
                "mps_driver_allocated_gib": 5.0,
                "mps_headroom_gib": 3.0,
            }
        )

    assert output.getvalue().strip() == (
        "step=11 mix=4ind/4ood train_loss=5.0000 train_lm_loss=6.0000 "
        "train_distill_loss=3.0000 lr=5.00e-05 ind_train_loss=4.5000 ood_train_loss=5.5000 "
        "latest_eval_lm_loss=7.0000 mps_allocated=3.00GiB mps_driver=5.00GiB "
        "mps_headroom=3.00GiB"
    )


def test_compute_distillation_loss_matches_between_chunked_and_unchunked_modes() -> None:
    student_logits = torch.tensor(
        [[[0.1, 0.4, -0.3], [0.6, -0.2, 0.0], [0.3, 0.2, -0.4]]],
        dtype=torch.float32,
    )
    teacher_logits = torch.tensor(
        [[[0.2, 0.1, -0.4], [0.3, 0.0, -0.1], [0.5, -0.2, -0.3]]],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 1]])

    chunked = _compute_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        attention_mask=attention_mask,
        vocab_index=None,
        vocab_chunk_size=2,
        torch_module=torch,
    )
    unchunked = _compute_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        attention_mask=attention_mask,
        vocab_index=None,
        vocab_chunk_size=0,
        torch_module=torch,
    )

    assert torch.allclose(chunked, unchunked, atol=1e-6, rtol=1e-6)


def test_compute_distillation_loss_can_restrict_to_original_token_subset() -> None:
    student_logits = torch.tensor(
        [[[0.3, -0.2, 1.1, 0.7], [0.4, 0.8, -0.5, 0.1], [0.2, -0.4, 0.5, 0.0]]],
        dtype=torch.float32,
    )
    teacher_logits = torch.tensor(
        [[[0.1, 0.5, 0.9, -0.3], [0.6, -0.2, 0.0, 0.7], [-0.1, 0.3, 0.4, 0.2]]],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor([[1, 1, 1]])
    vocab_index = torch.tensor([0, 2], dtype=torch.long)

    actual = _compute_distillation_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        attention_mask=attention_mask,
        vocab_index=vocab_index,
        vocab_chunk_size=0,
        torch_module=torch,
    )

    shift_student = student_logits[:, :-1, :].index_select(dim=-1, index=vocab_index)
    shift_teacher = teacher_logits[:, :-1, :].index_select(dim=-1, index=vocab_index)
    shift_mask = attention_mask[:, 1:].to(dtype=student_logits.dtype)
    expected_per_token = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(shift_student, dim=-1),
        torch.nn.functional.softmax(shift_teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    expected = (expected_per_token * shift_mask).sum() / shift_mask.sum().clamp_min(1.0)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_build_training_dry_run_report_requires_original_token_metadata_for_subset_kl(
    tmp_path: Path,
) -> None:
    ind_path = tmp_path / "ind.txt"
    ind_path.write_text("a\n", encoding="utf-8")

    with pytest.raises(ValueError, match="--distill-original-tokens-only requires"):
        build_training_dry_run_report(
            tmp_path / "model",
            ind_files=(ind_path,),
            distill_original_tokens_only=True,
        )
