from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tokcleanse._moe import (
    _configure_moe_implementation_for_device,
    _load_causal_lm_for_runtime,
    _load_tokenizer_for_runtime,
)
from tokcleanse.compare import (
    ComparisonSummary,
    DEFAULT_COMPARE_PROMPT,
    PromptComparison,
    compare_model_logits,
    load_prompts,
    serialize_comparisons,
    summarize_comparisons,
)
from tokcleanse.odin_moe import OdinMOETextDecoderLayer


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


def test_configure_moe_implementation_for_device_switches_grouped_mm_on_mps() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            _experts_implementation="grouped_mm",
            text_config=SimpleNamespace(_experts_implementation="grouped_mm"),
        )
    )

    changed = _configure_moe_implementation_for_device(model=model, device="mps")

    assert changed == ("config", "text_config")
    assert model.config._experts_implementation == "eager"
    assert model.config.text_config._experts_implementation == "eager"


def test_configure_moe_implementation_for_device_leaves_non_mps_unchanged() -> None:
    model = SimpleNamespace(config=SimpleNamespace(_experts_implementation="grouped_mm"))

    changed = _configure_moe_implementation_for_device(model=model, device="cpu")

    assert changed == ()
    assert model.config._experts_implementation == "grouped_mm"


def test_load_causal_lm_for_runtime_uses_odin_moe_for_drop_norm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "odin-moe"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "OdinMoE",
                "architectures": ["OdinMOEForCausalLM"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[tuple[str, object, object]] = []
    sentinel_model = SimpleNamespace(config=SimpleNamespace(_experts_implementation="eager"))

    def fake_from_pretrained(model_path: str, *, config: object, dtype: object) -> object:
        calls.append((model_path, config, dtype))
        return sentinel_model

    monkeypatch.setattr("tokcleanse.odin_moe.OdinMOEForCausalLM.from_pretrained", fake_from_pretrained)

    loaded_model = _load_causal_lm_for_runtime(
        model_path=str(model_dir),
        dtype="float32",
        device="cpu",
    )

    assert loaded_model is sentinel_model
    assert calls and calls[0][0] == str(model_dir)


def test_load_tokenizer_for_runtime_uses_odin_moe_tokenizer_for_odin_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "odin-moe"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "OdinMoE",
                "architectures": ["OdinMOEForCausalLM"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    calls: list[tuple[str, bool]] = []
    sentinel_tokenizer = object()

    def fake_from_pretrained(model_path: str, *, fix_mistral_regex: bool) -> object:
        calls.append((model_path, fix_mistral_regex))
        return sentinel_tokenizer

    monkeypatch.setattr(
        "tokcleanse.odin_moe_tokenizer.OdinMoETokenizerFast.from_pretrained",
        fake_from_pretrained,
    )

    tokenizer = _load_tokenizer_for_runtime(str(model_dir))

    assert tokenizer is sentinel_tokenizer
    assert calls == [(str(model_dir), True)]


class _IdentityModule(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class _ScaleModule(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.scale


class _ZeroAttention(torch.nn.Module):
    def forward(self, *, hidden_states: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, None]:
        return torch.zeros_like(hidden_states), None


class _SingleExpertRouter(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_count = hidden_states.shape[0]
        return (
            torch.ones((token_count, 1), dtype=hidden_states.dtype, device=hidden_states.device),
            torch.ones((token_count, 1), dtype=hidden_states.dtype, device=hidden_states.device),
            torch.zeros((token_count, 1), dtype=torch.long, device=hidden_states.device),
        )


class _IdentityExperts(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        del top_k_index, top_k_weights
        return hidden_states


class _ToyRoutedExpertLayer(torch.nn.Module):
    forward = OdinMOETextDecoderLayer.forward

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size_per_layer_input = 0
        self.input_layernorm = _IdentityModule()
        self.self_attn = _ZeroAttention()
        self.post_attention_layernorm = _IdentityModule()
        self.pre_feedforward_layernorm = _IdentityModule()
        self.router = _SingleExpertRouter()
        self.experts = _IdentityExperts()
        self.post_feedforward_layernorm = _IdentityModule()
        self.layer_scalar = torch.ones(1)


def test_odin_moe_text_decoder_layer_bypasses_shared_mlp_when_moe_enabled() -> None:
    layer = _ToyRoutedExpertLayer()
    hidden_states = torch.tensor([[[2.0, 4.0, 6.0, 8.0]]], dtype=torch.float32)

    output = layer(
        hidden_states,
        shared_kv_states={},
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
    )

    assert torch.equal(output, hidden_states * 2.0)


class _ToyTokenizer:
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        self._input_ids = input_ids
        self._attention_mask = attention_mask
        self.pad_token_id = 0

    def __call__(
        self,
        texts: list[str],
        *,
        return_tensors: str,
        padding: bool,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        del texts, return_tensors, padding, truncation, max_length
        return {
            "input_ids": self._input_ids.clone(),
            "attention_mask": self._attention_mask.clone(),
        }


class _ToyLogitModel(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def eval(self) -> "_ToyLogitModel":
        return self

    def to(self, device: str) -> "_ToyLogitModel":
        del device
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        logits = input_ids.float().unsqueeze(-1).repeat(1, 1, 3) + self.offset
        return SimpleNamespace(logits=logits)


def test_compare_model_logits_reports_per_prompt_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
    tokenizers = {
        "model-a": _ToyTokenizer(input_ids, attention_mask),
        "model-b": _ToyTokenizer(input_ids, attention_mask),
    }
    models = {
        "model-a": _ToyLogitModel(0.0),
        "model-b": _ToyLogitModel(0.5),
    }
    monkeypatch.setattr(
        "tokcleanse.compare._load_tokenizer_for_runtime",
        lambda model_path: tokenizers[model_path],
    )
    monkeypatch.setattr(
        "tokcleanse.compare._load_causal_lm_for_runtime",
        lambda *, model_path, dtype, device: models[model_path],
    )

    diffs = compare_model_logits(
        "model-a",
        "model-b",
        prompts=("first", "second"),
        completion=True,
        batch_size=2,
        device="cpu",
        dtype="float32",
        max_length=16,
        atol=1e-6,
        rtol=1e-6,
    )

    assert len(diffs) == 2
    assert diffs[0].token_count == 2
    assert diffs[0].max_abs_diff == pytest.approx(0.5)
    assert diffs[0].mean_abs_diff == pytest.approx(0.5)
    assert diffs[0].rmse == pytest.approx(0.5)
    assert diffs[0].allclose is False
    assert diffs[1].token_count == 3
    assert diffs[1].max_abs_diff == pytest.approx(0.5)


def test_compare_model_logits_rejects_misaligned_tokenization(monkeypatch: pytest.MonkeyPatch) -> None:
    attention_mask = torch.tensor([[1, 1]], dtype=torch.long)
    tokenizers = {
        "model-a": _ToyTokenizer(torch.tensor([[1, 2]], dtype=torch.long), attention_mask),
        "model-b": _ToyTokenizer(torch.tensor([[1, 3]], dtype=torch.long), attention_mask),
    }
    models = {
        "model-a": _ToyLogitModel(0.0),
        "model-b": _ToyLogitModel(0.0),
    }
    monkeypatch.setattr(
        "tokcleanse.compare._load_tokenizer_for_runtime",
        lambda model_path: tokenizers[model_path],
    )
    monkeypatch.setattr(
        "tokcleanse.compare._load_causal_lm_for_runtime",
        lambda *, model_path, dtype, device: models[model_path],
    )

    with pytest.raises(ValueError, match="aligned tokenization"):
        compare_model_logits(
            "model-a",
            "model-b",
            prompts=("prompt",),
            completion=True,
            device="cpu",
            dtype="float32",
        )
