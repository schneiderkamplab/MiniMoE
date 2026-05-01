"""Training helpers for MoE upcycling distillation workflows."""

from __future__ import annotations

import gc
import gzip
import math
import json
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from io import TextIOWrapper
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ._moe import _load_causal_lm_for_runtime, _load_tokenizer_for_runtime
from .compare import CompareDevice, CompareDType, resolve_compare_device

DEFAULT_TRAIN_BATCH_SIZE = 1
DEFAULT_TRAIN_DTYPE: CompareDType = "bfloat16"
DEFAULT_TRAIN_STEPS = -1
DEFAULT_TRAIN_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_TRAIN_TORCH_COMPILE: bool | None = None
DEFAULT_TRAIN_LEARNING_RATE = 5e-5
DEFAULT_TRAIN_MIN_LEARNING_RATE = 1e-8
DEFAULT_TRAIN_LR_WARMUP_STEPS = 0
DEFAULT_TRAIN_MAX_GRAD_NORM = 10.0
DEFAULT_ROUTER_LEARNING_RATE = 1e-3
DEFAULT_ROUTER_LR_WARMUP_STEPS = 0
DEFAULT_ROUTER_LR_ANNEAL_STEPS = 100
DEFAULT_ROUTER_MIN_LEARNING_RATE = 1e-6
DEFAULT_TRAIN_EVAL_EVERY = 50
DEFAULT_TRAIN_EVAL_MAX_BATCHES = 32
DEFAULT_LOG_EVERY = 1
DEFAULT_PRINT_EVERY = 10
DEFAULT_WEIGHT_DIFF_EVERY = 25
_DISTILL_KL_VOCAB_CHUNK_SIZE = 4096
DEFAULT_DISTILL_KL_VOCAB_CHUNK_SIZE = 0
_TOKENIZER_TOKEN_GROUPS_FILENAME = "tokenizer_token_groups.json"
_TRAINING_STATE_FILENAME = "training_state.pt"
_TRAINING_STATE_VERSION = 3
_AUXILIARY_TOKENIZER_FILENAMES = (
    "tokenizer_mapping.json",
    "tokenizer_token_groups.json",
    "tokenizer_added_token_initializers.json",
    "original_tokenizer.json",
    "original_tokenizer_config.json",
)


@dataclass(frozen=True, slots=True)
class TokenGroupMetadata:
    """Saved tokenizer-id groups used for selective token-row training."""

    original_token_ids: tuple[int, ...]
    added_token_ids: tuple[int, ...]
    requested_added_token_ids: tuple[int, ...]
    intermediate_added_token_ids: tuple[int, ...]
    special_token_ids: tuple[int, ...]
    added_tokens: tuple[str, ...]
    requested_added_tokens: tuple[str, ...]
    intermediate_added_tokens: tuple[str, ...]
    special_tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingLossWeights:
    """Loss weights for one language stream."""

    lm: float
    distill: float
    route: float = 0.0


@dataclass(frozen=True, slots=True)
class TrainingParameterSelection:
    """Parameter groups selected for training."""

    shared: bool = False
    expert_0: bool = False
    expert_1: bool = True
    embedding_lm_head: bool = True
    full_embedding_lm_head: bool = False


@dataclass(frozen=True, slots=True)
class TrainingDryRunReport:
    """Inspectable summary of what the trainer would optimize."""

    resolved_device: str
    dtype: str
    resume_from: str | None
    resume_step: int | None
    requested_steps: int
    resolved_steps: int
    resolved_epochs: int | None
    batch_size: int
    eval_batch_size: int
    eval_max_batches: int
    weight_diff_every: int
    pad_to_max_length: bool | None
    torch_compile: bool
    distill_ind: bool
    distill_ood: bool
    distill_original_tokens_only: bool
    distill_every: int
    lr_warmup_steps: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    min_learning_rate: float
    max_grad_norm: float
    weight_decay: float
    ind_route_weight: float
    ind_route_logit_bias: float
    ood_route_weight: float
    ood_route_logit_bias: float
    route_logit_bias_anneal_steps: int
    route_logit_bias_anneal_offset_steps: int
    route_logit_bias_anneal_loss_threshold: float
    router_learning_rate: float
    router_lr_warmup_steps: int
    router_lr_anneal_steps: int
    router_min_learning_rate: float
    train_shared: bool
    train_expert_0: bool
    train_expert_1: bool
    train_embedding_lm_head: bool
    train_full_embedding_lm_head: bool
    distill_kl_vocab_chunk_size: int
    checkpoint_every: int
    checkpoint_dir: str | None
    masked_row_parameter_names: tuple[str, ...]
    added_token_ids: tuple[int, ...]
    requested_added_token_ids: tuple[int, ...]
    intermediate_added_token_ids: tuple[int, ...]
    optimizer_groups: tuple[dict[str, Any], ...]
    trainable_parameter_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _MpsMemorySnapshot:
    allocated_bytes: int
    driver_allocated_bytes: int
    recommended_max_bytes: int | None


@dataclass(slots=True)
class _RouterRoutingMetrics:
    probability_sum: float = 0.0
    unbiased_probability_sum: float = 0.0
    weight_sum: float = 0.0
    token_layer_count: int = 0


@dataclass(slots=True)
class _LearningRateScheduler:
    optimizer: Any
    lr_group: str
    warmup_steps: int
    total_steps: int
    min_learning_rate: float
    base_learning_rate: float
    completed_steps: int = 0

    def __post_init__(self) -> None:
        if self.total_steps < 1:
            raise ValueError("total_steps must be at least 1")
        if self.min_learning_rate <= 0:
            raise ValueError("min_learning_rate must be positive")
        if self.base_learning_rate <= 0:
            raise ValueError("base_learning_rate must be positive")
        self._apply_learning_rate()

    def step(self) -> None:
        self.completed_steps += 1
        self._apply_learning_rate()

    def state_dict(self) -> dict[str, int | float]:
        return {
            "completed_steps": self.completed_steps,
            "lr_group": self.lr_group,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_learning_rate": self.min_learning_rate,
            "base_learning_rate": self.base_learning_rate,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        completed_steps = state_dict.get("completed_steps", 0)
        if not isinstance(completed_steps, int) or completed_steps < 0:
            raise ValueError("training-state lr_scheduler completed_steps must be a non-negative integer")
        expected_values = {
            "lr_group": self.lr_group,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_learning_rate": self.min_learning_rate,
            "base_learning_rate": self.base_learning_rate,
        }
        for key, expected_value in expected_values.items():
            saved_value = state_dict.get(key, expected_value)
            if saved_value != expected_value:
                raise ValueError(f"training-state lr_scheduler {key} does not match current run")
        self.completed_steps = completed_steps
        self._apply_learning_rate()

    def _apply_learning_rate(self) -> None:
        learning_rate = self._current_learning_rate()
        for group in self.optimizer.param_groups:
            if group.get("lr_group", "main") == self.lr_group:
                group["lr"] = learning_rate

    def _current_learning_rate(self) -> float:
        step_index = self.completed_steps + 1
        if self.warmup_steps > 0 and step_index <= self.warmup_steps:
            return self.base_learning_rate * (step_index / self.warmup_steps)
        decay_start = self.warmup_steps
        decay_steps = max(self.total_steps - decay_start, 1)
        decay_progress = min(max(step_index - decay_start, 0), decay_steps) / decay_steps
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        min_ratio = min(self.min_learning_rate / self.base_learning_rate, 1.0)
        return self.base_learning_rate * (min_ratio + (1.0 - min_ratio) * cosine_scale)


@dataclass(slots=True)
class _RouteBiasAnnealer:
    anneal_steps: int
    loss_threshold: float
    offset_steps: int = 0
    completed_anneal_steps: int = 0
    observed_steps: int = 0

    def __post_init__(self) -> None:
        if self.anneal_steps < 0:
            raise ValueError("anneal_steps cannot be negative")
        if self.offset_steps < 0:
            raise ValueError("offset_steps cannot be negative")
        if self.loss_threshold < 0:
            raise ValueError("loss_threshold cannot be negative")

    def scale(self) -> float:
        if self.anneal_steps <= 0:
            return 1.0
        if self.anneal_steps <= 1 or self.completed_anneal_steps >= self.anneal_steps - 1:
            return 0.0
        progress = self.completed_anneal_steps / (self.anneal_steps - 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def observe_route_loss(self, route_loss: float) -> bool:
        if self.anneal_steps <= 0:
            return False
        self.observed_steps += 1
        if self.observed_steps <= self.offset_steps:
            return False
        if math.isfinite(route_loss) and route_loss <= self.loss_threshold:
            self.completed_anneal_steps = min(self.completed_anneal_steps + 1, self.anneal_steps)
            return True
        return False

    def state_dict(self) -> dict[str, int | float]:
        return {
            "anneal_steps": self.anneal_steps,
            "loss_threshold": self.loss_threshold,
            "offset_steps": self.offset_steps,
            "completed_anneal_steps": self.completed_anneal_steps,
            "observed_steps": self.observed_steps,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        anneal_steps = state_dict.get("anneal_steps", self.anneal_steps)
        loss_threshold = state_dict.get("loss_threshold", self.loss_threshold)
        offset_steps = state_dict.get("offset_steps", self.offset_steps)
        if anneal_steps != self.anneal_steps:
            raise ValueError("training-state route_bias_annealer anneal_steps does not match current run")
        if loss_threshold != self.loss_threshold:
            raise ValueError("training-state route_bias_annealer loss_threshold does not match current run")
        if offset_steps != self.offset_steps:
            raise ValueError("training-state route_bias_annealer offset_steps does not match current run")
        completed_anneal_steps = state_dict.get("completed_anneal_steps", 0)
        if not isinstance(completed_anneal_steps, int) or completed_anneal_steps < 0:
            raise ValueError("training-state route_bias_annealer completed_anneal_steps is invalid")
        observed_steps = state_dict.get("observed_steps", 0)
        if not isinstance(observed_steps, int) or observed_steps < 0:
            raise ValueError("training-state route_bias_annealer observed_steps is invalid")
        self.completed_anneal_steps = min(completed_anneal_steps, max(self.anneal_steps, 0))
        self.observed_steps = observed_steps


def load_token_group_metadata(model_path: str | Path) -> TokenGroupMetadata:
    """Load saved tokenizer token-group metadata for a local model directory."""

    metadata_path = Path(model_path).expanduser() / _TOKENIZER_TOKEN_GROUPS_FILENAME
    if not metadata_path.exists():
        return TokenGroupMetadata(
            original_token_ids=(),
            added_token_ids=(),
            requested_added_token_ids=(),
            intermediate_added_token_ids=(),
            special_token_ids=(),
            added_tokens=(),
            requested_added_tokens=(),
            intermediate_added_tokens=(),
            special_tokens=(),
        )
    data = _load_json_object(metadata_path)
    return TokenGroupMetadata(
        original_token_ids=_load_int_tuple(data, "original_token_ids"),
        added_token_ids=_load_int_tuple(data, "added_token_ids"),
        requested_added_token_ids=_load_int_tuple(data, "requested_added_token_ids"),
        intermediate_added_token_ids=_load_int_tuple(data, "intermediate_added_token_ids"),
        special_token_ids=_load_int_tuple(data, "special_token_ids"),
        added_tokens=_load_str_tuple(data, "added_tokens"),
        requested_added_tokens=_load_str_tuple(data, "requested_added_tokens"),
        intermediate_added_tokens=_load_str_tuple(data, "intermediate_added_tokens"),
        special_tokens=_load_str_tuple(data, "special_tokens"),
    )


def prepare_student_for_distillation(
    model: Any,
    *,
    token_groups: TokenGroupMetadata,
    torch_module: Any,
    parameter_selection: TrainingParameterSelection = TrainingParameterSelection(),
) -> None:
    """Freeze the shared backbone and leave MoE-specific plus added rows trainable."""

    masked_row_parameter_ids = _masked_row_parameter_ids(model)
    for _, parameter in model.named_parameters():
        parameter.requires_grad_(False)

    for name, parameter in model.named_parameters():
        if id(parameter) in masked_row_parameter_ids:
            continue
        if _is_expert_weight_parameter(name):
            if parameter_selection.expert_0 or parameter_selection.expert_1:
                parameter.requires_grad_(True)
                _mask_frozen_expert_gradients(
                    name=name,
                    parameter=parameter,
                    torch_module=torch_module,
                    train_expert_0=parameter_selection.expert_0,
                    train_expert_1=parameter_selection.expert_1,
                )
        elif _is_router_parameter(name):
            parameter.requires_grad_(True)
        elif parameter_selection.shared:
            parameter.requires_grad_(True)

    if parameter_selection.embedding_lm_head and parameter_selection.full_embedding_lm_head:
        _enable_all_embedding_lm_head(model=model)
    elif parameter_selection.embedding_lm_head:
        _enable_added_token_rows(
            model=model,
            token_ids=token_groups.added_token_ids,
            torch_module=torch_module,
        )


class _RouterRoutingMetricCollector:
    def __init__(self, *, torch_module: Any, expert_index: int = 1) -> None:
        self._torch = torch_module
        self._expert_index = expert_index
        self._active_language: str | None = None
        self._metrics = {
            "ind": _RouterRoutingMetrics(),
            "ood": _RouterRoutingMetrics(),
        }
        self._losses = {
            "ind": [],
            "ood": [],
        }
        self._routers: list[Any] = []
        self._handles: list[Any] = []

    def register(self, model: Any) -> None:
        for name, module in model.named_modules():
            if name.endswith(".router"):
                self._routers.append(module)
                self._handles.append(module.register_forward_hook(self._record_router_output))

    def close(self) -> None:
        self._set_expert1_logit_bias(None)
        for handle in self._handles:
            handle.remove()
        self._routers.clear()
        self._handles.clear()

    def reset(self) -> None:
        for language in self._metrics:
            self._metrics[language] = _RouterRoutingMetrics()
            self._losses[language] = []

    def activate(self, language: str, *, expert1_logit_bias: float = 0.0) -> None:
        self._active_language = language
        self._set_expert1_logit_bias(expert1_logit_bias)

    def deactivate(self) -> None:
        self._active_language = None
        self._set_expert1_logit_bias(None)

    def averages(self) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for language, metrics in self._metrics.items():
            if metrics.token_layer_count > 0:
                denominator = float(metrics.token_layer_count)
                probability = metrics.probability_sum / denominator
                unbiased_probability = metrics.unbiased_probability_sum / denominator
                result[f"{language}_router_prob_expert{self._expert_index}"] = probability
                result[f"{language}_router_unbiased_prob_expert{self._expert_index}"] = unbiased_probability
                result[f"{language}_router_bias_prob_delta_expert{self._expert_index}"] = (
                    probability - unbiased_probability
                )
                result[f"{language}_router_weight_expert{self._expert_index}"] = metrics.weight_sum / denominator
            else:
                result[f"{language}_router_prob_expert{self._expert_index}"] = None
                result[f"{language}_router_unbiased_prob_expert{self._expert_index}"] = None
                result[f"{language}_router_bias_prob_delta_expert{self._expert_index}"] = None
                result[f"{language}_router_weight_expert{self._expert_index}"] = None
        return result

    def consume_loss(self, *, language: str) -> Any | None:
        losses = self._losses.get(language)
        if not losses:
            return None
        loss = self._torch.stack(tuple(losses)).sum()
        losses.clear()
        return loss

    def _record_router_output(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        language = self._active_language
        if language not in self._metrics:
            return
        if not isinstance(output, tuple) or len(output) < 3:
            return
        router_probabilities, top_k_weights, top_k_index = output[:3]
        if not all(hasattr(tensor, "detach") for tensor in (router_probabilities, top_k_weights, top_k_index)):
            return
        with self._torch.no_grad():
            probabilities = router_probabilities.detach()
            unbiased_router_probabilities = getattr(_module, "_odin_unbiased_router_probabilities", None)
            if (
                unbiased_router_probabilities is not None
                and hasattr(unbiased_router_probabilities, "detach")
                and unbiased_router_probabilities.shape == router_probabilities.shape
            ):
                unbiased_probabilities = unbiased_router_probabilities.detach()
            else:
                unbiased_probabilities = probabilities
            weights = top_k_weights.detach()
            indices = top_k_index.detach()
            if probabilities.ndim != 2 or weights.ndim != 2 or indices.ndim != 2:
                return
            if self._expert_index >= int(probabilities.shape[-1]):
                return
            expert_probability_sum = float(probabilities[:, self._expert_index].sum().cpu())
            unbiased_expert_probability_sum = float(
                unbiased_probabilities[:, self._expert_index].sum().cpu()
            )
            expert_weight_mask = indices == self._expert_index
            expert_weight_sum = float(weights.masked_select(expert_weight_mask).sum().cpu())
            token_count = int(probabilities.shape[0])
        metrics = self._metrics[language]
        metrics.probability_sum += expert_probability_sum
        metrics.unbiased_probability_sum += unbiased_expert_probability_sum
        metrics.weight_sum += expert_weight_sum
        metrics.token_layer_count += token_count
        target_expert_index = 0 if language == "ind" else self._expert_index
        if target_expert_index < int(router_probabilities.shape[-1]):
            target_probability = router_probabilities[:, target_expert_index].clamp_min(1e-8)
            self._losses[language].append(-target_probability.log().mean())

    def _set_expert1_logit_bias(self, value: float | None) -> None:
        for router in self._routers:
            if value is None:
                if hasattr(router, "_odin_expert1_logit_bias"):
                    delattr(router, "_odin_expert1_logit_bias")
            else:
                setattr(router, "_odin_expert1_logit_bias", float(value))


def build_training_dry_run_report(
    student_model: str | Path,
    *,
    resume_from: Path | None = None,
    ind_files: tuple[Path, ...] = (),
    ood_files: tuple[Path, ...] = (),
    device: CompareDevice = "auto",
    dtype: CompareDType = DEFAULT_TRAIN_DTYPE,
    steps: int = DEFAULT_TRAIN_STEPS,
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    eval_batch_size: int | None = None,
    eval_max_batches: int = DEFAULT_TRAIN_EVAL_MAX_BATCHES,
    weight_diff_every: int = DEFAULT_WEIGHT_DIFF_EVERY,
    pad_to_max_length: bool | None = None,
    torch_compile: bool | None = DEFAULT_TRAIN_TORCH_COMPILE,
    distill_ind: bool = True,
    distill_ood: bool = True,
    distill_original_tokens_only: bool = False,
    distill_every: int = 1,
    lr_warmup_steps: int = DEFAULT_TRAIN_LR_WARMUP_STEPS,
    gradient_accumulation_steps: int = DEFAULT_TRAIN_GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing: bool = True,
    learning_rate: float = DEFAULT_TRAIN_LEARNING_RATE,
    min_learning_rate: float = DEFAULT_TRAIN_MIN_LEARNING_RATE,
    max_grad_norm: float = DEFAULT_TRAIN_MAX_GRAD_NORM,
    weight_decay: float = 0.0,
    distill_kl_vocab_chunk_size: int = DEFAULT_DISTILL_KL_VOCAB_CHUNK_SIZE,
    checkpoint_every: int = 0,
    checkpoint_dir: Path | None = None,
    ind_batches_per_cycle: int = 1,
    ood_batches_per_cycle: int = 1,
    ind_route_weight: float = 0.0,
    ind_route_logit_bias: float = 0.0,
    ood_route_weight: float = 0.0,
    ood_route_logit_bias: float = 0.0,
    route_logit_bias_anneal_steps: int = 0,
    route_logit_bias_anneal_offset_steps: int = 0,
    route_logit_bias_anneal_loss_threshold: float = 5e-2,
    router_learning_rate: float = DEFAULT_ROUTER_LEARNING_RATE,
    router_lr_warmup_steps: int = DEFAULT_ROUTER_LR_WARMUP_STEPS,
    router_lr_anneal_steps: int = DEFAULT_ROUTER_LR_ANNEAL_STEPS,
    router_min_learning_rate: float = DEFAULT_ROUTER_MIN_LEARNING_RATE,
    train_shared: bool = False,
    train_expert_0: bool = False,
    train_expert_1: bool = True,
    train_embedding_lm_head: bool = True,
    train_full_embedding_lm_head: bool = False,
) -> TrainingDryRunReport:
    """Describe which parameters and rows would be trained without running training."""

    import torch
    if steps == 0:
        raise ValueError("--steps must not be 0")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if eval_batch_size is not None and eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be at least 1")
    if eval_max_batches < 0:
        raise ValueError("--eval-max-batches cannot be negative")
    if weight_diff_every < 0:
        raise ValueError("--weight-diff-every cannot be negative")
    if distill_every < 1:
        raise ValueError("--distill-every must be at least 1")
    if lr_warmup_steps < 0:
        raise ValueError("--lr-warmup-steps cannot be negative")
    if gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation must be at least 1")
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if min_learning_rate <= 0:
        raise ValueError("--min-learning-rate must be positive")
    if max_grad_norm < 0:
        raise ValueError("--max-grad-norm cannot be negative")
    if ind_route_weight < 0 or ood_route_weight < 0:
        raise ValueError("--ind-route-weight and --ood-route-weight cannot be negative")
    if route_logit_bias_anneal_steps < 0:
        raise ValueError("--route-logit-bias-anneal-steps cannot be negative")
    if route_logit_bias_anneal_offset_steps < 0:
        raise ValueError("--route-logit-bias-anneal-offset-steps cannot be negative")
    if route_logit_bias_anneal_loss_threshold < 0:
        raise ValueError("--route-logit-bias-anneal-loss-threshold cannot be negative")
    if router_learning_rate <= 0:
        raise ValueError("--router-learning-rate must be positive")
    if router_lr_warmup_steps < 0:
        raise ValueError("--router-lr-warmup-steps cannot be negative")
    if router_lr_anneal_steps < 1:
        raise ValueError("--router-lr-anneal-steps must be at least 1")
    if router_min_learning_rate <= 0:
        raise ValueError("--router-min-learning-rate must be positive")
    if distill_kl_vocab_chunk_size < 0:
        raise ValueError("--distill-kl-vocab-chunk-size cannot be negative")
    if checkpoint_every < 0:
        raise ValueError("--checkpoint-every cannot be negative")
    if checkpoint_every > 0 and checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required when --checkpoint-every is greater than 0")
    if ind_batches_per_cycle < 0 or ood_batches_per_cycle < 0:
        raise ValueError("--ind-batches-per-cycle and --ood-batches-per-cycle cannot be negative")
    if not ind_files and not ood_files:
        raise ValueError("At least one corpus file must be provided")
    parameter_selection = TrainingParameterSelection(
        shared=train_shared,
        expert_0=train_expert_0,
        expert_1=train_expert_1,
        embedding_lm_head=train_embedding_lm_head,
        full_embedding_lm_head=train_full_embedding_lm_head,
    )

    scheduler = _LanguageScheduler(
        ind_enabled=bool(ind_files),
        ood_enabled=bool(ood_files),
        ind_batches_per_cycle=ind_batches_per_cycle,
        ood_batches_per_cycle=ood_batches_per_cycle,
    )
    resolved_steps, resolved_epochs = _resolve_training_steps(
        requested_steps=steps,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ind_files=ind_files,
        ood_files=ood_files,
        scheduler=scheduler,
    )
    resolved_device = resolve_compare_device(device)
    resolved_torch_compile = _resolve_torch_compile(
        requested=torch_compile,
        device=resolved_device,
    )
    resolved_dtype = _resolve_dtype(torch, dtype)
    resolved_eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size
    student_source, resume_state = _resolve_resume_source(
        student_model=student_model,
        resume_from=resume_from,
        torch_module=torch,
    )
    token_groups = load_token_group_metadata(student_source)
    _validate_distillation_token_groups(
        token_groups=token_groups,
        distill_original_tokens_only=distill_original_tokens_only,
    )

    model = None
    try:
        model = _load_causal_lm_for_runtime(
            model_path=str(student_source),
            dtype=resolved_dtype,
            device=resolved_device,
        )
        prepare_student_for_distillation(
            model,
            token_groups=token_groups,
            torch_module=torch,
            parameter_selection=parameter_selection,
        )
        masked_row_parameter_ids = _masked_row_parameter_ids(model)
        masked_row_parameter_names: list[str] = []
        masked_row_parameters: list[Any] = []
        router_trainable_parameters: list[Any] = []
        standard_trainable_parameters: list[Any] = []
        trainable_parameter_names: list[str] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            trainable_parameter_names.append(name)
            if ".router." in name:
                router_trainable_parameters.append(parameter)
            elif id(parameter) in masked_row_parameter_ids:
                masked_row_parameter_names.append(name)
                masked_row_parameters.append(parameter)
            else:
                standard_trainable_parameters.append(parameter)
        optimizer_groups = _serialize_optimizer_groups(
            standard_trainable_parameters=standard_trainable_parameters,
            router_trainable_parameters=router_trainable_parameters,
            masked_row_parameters=masked_row_parameters,
            named_parameters=dict(model.named_parameters()),
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            router_learning_rate=router_learning_rate,
        )
        return TrainingDryRunReport(
            resolved_device=resolved_device,
            dtype=dtype,
            resume_from=str(student_source) if resume_from is not None else None,
            resume_step=_resume_state_step(resume_state),
            requested_steps=steps,
            resolved_steps=resolved_steps,
            resolved_epochs=resolved_epochs,
            batch_size=batch_size,
            eval_batch_size=resolved_eval_batch_size,
            eval_max_batches=eval_max_batches,
            weight_diff_every=weight_diff_every,
            pad_to_max_length=pad_to_max_length,
            torch_compile=resolved_torch_compile,
            distill_ind=distill_ind,
            distill_ood=distill_ood,
            distill_original_tokens_only=distill_original_tokens_only,
            distill_every=distill_every,
            lr_warmup_steps=lr_warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            ind_route_weight=ind_route_weight,
            ind_route_logit_bias=ind_route_logit_bias,
            ood_route_weight=ood_route_weight,
            ood_route_logit_bias=ood_route_logit_bias,
            route_logit_bias_anneal_steps=route_logit_bias_anneal_steps,
            route_logit_bias_anneal_offset_steps=route_logit_bias_anneal_offset_steps,
            route_logit_bias_anneal_loss_threshold=route_logit_bias_anneal_loss_threshold,
            router_learning_rate=router_learning_rate,
            router_lr_warmup_steps=router_lr_warmup_steps,
            router_lr_anneal_steps=router_lr_anneal_steps,
            router_min_learning_rate=router_min_learning_rate,
            distill_kl_vocab_chunk_size=distill_kl_vocab_chunk_size,
            train_shared=train_shared,
            train_expert_0=train_expert_0,
            train_expert_1=train_expert_1,
            train_embedding_lm_head=train_embedding_lm_head,
            train_full_embedding_lm_head=train_full_embedding_lm_head,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=str(checkpoint_dir.expanduser()) if checkpoint_dir is not None else None,
            masked_row_parameter_names=tuple(masked_row_parameter_names),
            added_token_ids=token_groups.added_token_ids,
            requested_added_token_ids=token_groups.requested_added_token_ids,
            intermediate_added_token_ids=token_groups.intermediate_added_token_ids,
            optimizer_groups=optimizer_groups,
            trainable_parameter_names=tuple(trainable_parameter_names),
        )
    finally:
        if model is not None:
            del model
        _release_resources(torch_module=torch)


def train_distilled_model(
    student_model: str | Path,
    teacher_model: str | Path,
    output_dir: str | Path,
    *,
    resume_from: Path | None = None,
    ind_files: tuple[Path, ...] = (),
    ood_files: tuple[Path, ...] = (),
    eval_files: tuple[Path, ...] = (),
    device: CompareDevice = "auto",
    dtype: CompareDType = DEFAULT_TRAIN_DTYPE,
    steps: int = DEFAULT_TRAIN_STEPS,
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    eval_batch_size: int | None = None,
    eval_max_batches: int = DEFAULT_TRAIN_EVAL_MAX_BATCHES,
    weight_diff_every: int = DEFAULT_WEIGHT_DIFF_EVERY,
    pad_to_max_length: bool | None = None,
    torch_compile: bool | None = DEFAULT_TRAIN_TORCH_COMPILE,
    distill_ind: bool = True,
    distill_ood: bool = True,
    distill_original_tokens_only: bool = False,
    distill_every: int = 1,
    lr_warmup_steps: int = DEFAULT_TRAIN_LR_WARMUP_STEPS,
    gradient_accumulation_steps: int = DEFAULT_TRAIN_GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing: bool = True,
    max_length: int = 1024,
    learning_rate: float = DEFAULT_TRAIN_LEARNING_RATE,
    min_learning_rate: float = DEFAULT_TRAIN_MIN_LEARNING_RATE,
    max_grad_norm: float = DEFAULT_TRAIN_MAX_GRAD_NORM,
    weight_decay: float = 0.0,
    distill_kl_vocab_chunk_size: int = DEFAULT_DISTILL_KL_VOCAB_CHUNK_SIZE,
    checkpoint_every: int = 0,
    checkpoint_dir: Path | None = None,
    eval_every_steps: int = DEFAULT_TRAIN_EVAL_EVERY,
    log_every: int = DEFAULT_LOG_EVERY,
    print_every: int = DEFAULT_PRINT_EVERY,
    ind_batches_per_cycle: int = 1,
    ood_batches_per_cycle: int = 1,
    ind_lm_weight: float = 0.05,
    ind_distill_weight: float = 1.0,
    ind_route_weight: float = 0.0,
    ind_route_logit_bias: float = 0.0,
    ood_lm_weight: float = 1.0,
    ood_distill_weight: float = 0.25,
    ood_route_weight: float = 0.0,
    ood_route_logit_bias: float = 0.0,
    route_logit_bias_anneal_steps: int = 0,
    route_logit_bias_anneal_offset_steps: int = 0,
    route_logit_bias_anneal_loss_threshold: float = 5e-2,
    router_learning_rate: float = DEFAULT_ROUTER_LEARNING_RATE,
    router_lr_warmup_steps: int = DEFAULT_ROUTER_LR_WARMUP_STEPS,
    router_lr_anneal_steps: int = DEFAULT_ROUTER_LR_ANNEAL_STEPS,
    router_min_learning_rate: float = DEFAULT_ROUTER_MIN_LEARNING_RATE,
    train_shared: bool = False,
    train_expert_0: bool = False,
    train_expert_1: bool = True,
    train_embedding_lm_head: bool = True,
    train_full_embedding_lm_head: bool = False,
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Train an upcycled student model against a frozen teacher with aligned tokenizers."""

    import torch

    _seed_training_random_generators(seed=seed, torch_module=torch)
    if steps == 0:
        raise ValueError("--steps must not be 0")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if eval_batch_size is not None and eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be at least 1")
    if eval_max_batches < 0:
        raise ValueError("--eval-max-batches cannot be negative")
    if weight_diff_every < 0:
        raise ValueError("--weight-diff-every cannot be negative")
    if distill_every < 1:
        raise ValueError("--distill-every must be at least 1")
    if lr_warmup_steps < 0:
        raise ValueError("--lr-warmup-steps cannot be negative")
    if gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation must be at least 1")
    if max_length < 2:
        raise ValueError("--max-length must be at least 2")
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if min_learning_rate <= 0:
        raise ValueError("--min-learning-rate must be positive")
    if max_grad_norm < 0:
        raise ValueError("--max-grad-norm cannot be negative")
    if ind_route_weight < 0 or ood_route_weight < 0:
        raise ValueError("--ind-route-weight and --ood-route-weight cannot be negative")
    if route_logit_bias_anneal_steps < 0:
        raise ValueError("--route-logit-bias-anneal-steps cannot be negative")
    if route_logit_bias_anneal_offset_steps < 0:
        raise ValueError("--route-logit-bias-anneal-offset-steps cannot be negative")
    if route_logit_bias_anneal_loss_threshold < 0:
        raise ValueError("--route-logit-bias-anneal-loss-threshold cannot be negative")
    if router_learning_rate <= 0:
        raise ValueError("--router-learning-rate must be positive")
    if router_lr_warmup_steps < 0:
        raise ValueError("--router-lr-warmup-steps cannot be negative")
    if router_lr_anneal_steps < 1:
        raise ValueError("--router-lr-anneal-steps must be at least 1")
    if router_min_learning_rate <= 0:
        raise ValueError("--router-min-learning-rate must be positive")
    if distill_kl_vocab_chunk_size < 0:
        raise ValueError("--distill-kl-vocab-chunk-size cannot be negative")
    if checkpoint_every < 0:
        raise ValueError("--checkpoint-every cannot be negative")
    if eval_every_steps < 0:
        raise ValueError("--eval-every cannot be negative")
    if log_every < 1:
        raise ValueError("--log-every must be at least 1")
    if print_every < 1:
        raise ValueError("--print-every must be at least 1")
    if ind_batches_per_cycle < 0 or ood_batches_per_cycle < 0:
        raise ValueError("--ind-batches-per-cycle and --ood-batches-per-cycle cannot be negative")
    if not ind_files and not ood_files:
        raise ValueError("At least one corpus file must be provided")
    parameter_selection = TrainingParameterSelection(
        shared=train_shared,
        expert_0=train_expert_0,
        expert_1=train_expert_1,
        embedding_lm_head=train_embedding_lm_head,
        full_embedding_lm_head=train_full_embedding_lm_head,
    )

    rng = random.Random(seed)
    needs_teacher = (
        (distill_ind and ind_distill_weight > 0)
        or (distill_ood and ood_distill_weight > 0)
    )
    ind_stream = (
        _CorpusStream(ind_files, rng=random.Random(rng.randrange(1 << 30)))
        if ind_files
        else None
    )
    ood_stream = (
        _CorpusStream(ood_files, rng=random.Random(rng.randrange(1 << 30)))
        if ood_files
        else None
    )
    scheduler = _LanguageScheduler(
        ind_enabled=bool(ind_files),
        ood_enabled=bool(ood_files),
        ind_batches_per_cycle=ind_batches_per_cycle,
        ood_batches_per_cycle=ood_batches_per_cycle,
    )
    resolved_steps, resolved_epochs = _resolve_training_steps(
        requested_steps=steps,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ind_files=ind_files,
        ood_files=ood_files,
        scheduler=scheduler,
    )
    resolved_device = resolve_compare_device(device)
    resolved_torch_compile = _resolve_torch_compile(
        requested=torch_compile,
        device=resolved_device,
    )
    resolved_dtype = _resolve_dtype(torch, dtype)
    resolved_eval_batch_size = batch_size if eval_batch_size is None else eval_batch_size
    student_source, resume_state = _resolve_resume_source(
        student_model=student_model,
        resume_from=resume_from,
        torch_module=torch,
    )
    token_groups = load_token_group_metadata(student_source)
    _validate_distillation_token_groups(
        token_groups=token_groups,
        distill_original_tokens_only=distill_original_tokens_only,
    )
    distillation_vocab_index = _build_distillation_vocab_index(
        token_groups=token_groups,
        distill_original_tokens_only=distill_original_tokens_only,
        device=resolved_device,
        torch_module=torch,
    )
    output_path = Path(output_dir).expanduser()
    resume_step = _resume_state_step(resume_state) or 0
    training_configuration_state = _build_training_configuration_state(
        resolved_steps=resolved_steps,
        resolved_epochs=resolved_epochs,
        teacher_model=teacher_model,
        ind_files=ind_files,
        ood_files=ood_files,
        eval_files=eval_files,
        resolved_device=resolved_device,
        dtype=dtype,
        batch_size=batch_size,
        eval_batch_size=resolved_eval_batch_size,
        eval_max_batches=eval_max_batches,
        weight_diff_every=weight_diff_every,
        pad_to_max_length=pad_to_max_length,
        torch_compile=resolved_torch_compile,
        distill_ind=distill_ind,
        distill_ood=distill_ood,
        distill_original_tokens_only=distill_original_tokens_only,
        distill_every=distill_every,
        lr_warmup_steps=lr_warmup_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        max_length=max_length,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
        distill_kl_vocab_chunk_size=distill_kl_vocab_chunk_size,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir.expanduser() if checkpoint_dir is not None else None,
        eval_every_steps=eval_every_steps,
        log_every=log_every,
        print_every=print_every,
        ind_batches_per_cycle=ind_batches_per_cycle,
        ood_batches_per_cycle=ood_batches_per_cycle,
        ind_lm_weight=ind_lm_weight,
        ind_distill_weight=ind_distill_weight,
        ind_route_weight=ind_route_weight,
        ind_route_logit_bias=ind_route_logit_bias,
        ood_lm_weight=ood_lm_weight,
        ood_distill_weight=ood_distill_weight,
        ood_route_weight=ood_route_weight,
        ood_route_logit_bias=ood_route_logit_bias,
        route_logit_bias_anneal_steps=route_logit_bias_anneal_steps,
        route_logit_bias_anneal_offset_steps=route_logit_bias_anneal_offset_steps,
        route_logit_bias_anneal_loss_threshold=route_logit_bias_anneal_loss_threshold,
        router_learning_rate=router_learning_rate,
        router_lr_warmup_steps=router_lr_warmup_steps,
        router_lr_anneal_steps=router_lr_anneal_steps,
        router_min_learning_rate=router_min_learning_rate,
        parameter_selection=parameter_selection,
        seed=seed,
    )
    _validate_resume_training_configuration(
        resume_state=resume_state,
        training_configuration_state=training_configuration_state,
    )
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {output_path}")
        if resume_from is not None and _path_contains(output_path, student_source):
            raise ValueError("--resume-from cannot point inside --output-dir when --overwrite is used")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpointing_enabled = checkpoint_every > 0
    resolved_checkpoint_dir = checkpoint_dir.expanduser() if checkpoint_dir is not None else None
    if checkpointing_enabled and resolved_checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required when checkpointing is enabled")
    if checkpointing_enabled and resolved_checkpoint_dir is not None:
        if resolved_checkpoint_dir.exists() and resume_from is None:
            if not overwrite:
                raise FileExistsError(f"Checkpoint directory already exists: {resolved_checkpoint_dir}")
            shutil.rmtree(resolved_checkpoint_dir)
        resolved_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if resume_step >= resolved_steps:
        raise ValueError(
            f"Resume checkpoint is already at step {resume_step}, which leaves no work under resolved_steps={resolved_steps}"
        )
    metrics_log_path = output_path / "training_metrics.jsonl"
    ind_weights = TrainingLossWeights(lm=ind_lm_weight, distill=ind_distill_weight, route=ind_route_weight)
    ood_weights = TrainingLossWeights(lm=ood_lm_weight, distill=ood_distill_weight, route=ood_route_weight)
    route_bias_annealer = _RouteBiasAnnealer(
        anneal_steps=route_logit_bias_anneal_steps,
        loss_threshold=route_logit_bias_anneal_loss_threshold,
        offset_steps=route_logit_bias_anneal_offset_steps,
    )

    tokenizer = None
    teacher_tokenizer = None
    student = None
    student_runtime = None
    teacher = None
    optimizer = None
    lr_scheduler = None
    router_metric_collector = None
    try:
        tokenizer = _load_tokenizer_for_runtime(str(student_source))
        if needs_teacher:
            teacher_tokenizer = _load_tokenizer_for_runtime(str(teacher_model))
            _validate_matching_tokenizers(tokenizer=tokenizer, teacher_tokenizer=teacher_tokenizer)
        _prepare_training_tokenizer(tokenizer)
        student = _load_causal_lm_for_runtime(
            model_path=str(student_source),
            dtype=resolved_dtype,
            device=resolved_device,
        )
        if gradient_checkpointing:
            _enable_gradient_checkpointing(student)
        if needs_teacher:
            teacher = _load_causal_lm_for_runtime(
                model_path=str(teacher_model),
                dtype=resolved_dtype,
                device=resolved_device,
            )
        student.to(resolved_device)
        if teacher is not None:
            teacher.to(resolved_device)
            teacher.eval()
        prepare_student_for_distillation(
            student,
            token_groups=token_groups,
            torch_module=torch,
            parameter_selection=parameter_selection,
        )
        student.train()

        masked_row_parameter_ids = _masked_row_parameter_ids(student)
        masked_row_parameters = []
        router_trainable_parameters = []
        standard_trainable_parameters = []
        for name, parameter in student.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".router." in name:
                router_trainable_parameters.append(parameter)
            elif id(parameter) in masked_row_parameter_ids:
                masked_row_parameters.append(parameter)
            else:
                standard_trainable_parameters.append(parameter)
        if not masked_row_parameters and not standard_trainable_parameters and not router_trainable_parameters:
            raise ValueError("No trainable parameters were selected for training")
        task_trainable_parameters = standard_trainable_parameters + masked_row_parameters
        optimizer_param_groups = _build_optimizer_param_groups(
            standard_trainable_parameters=standard_trainable_parameters,
            router_trainable_parameters=router_trainable_parameters,
            masked_row_parameters=masked_row_parameters,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            router_learning_rate=router_learning_rate,
        )
        optimizer = torch.optim.AdamW(
            optimizer_param_groups,
            lr=learning_rate,
        )
        optimizer_weight_decays = tuple(
            float(group["weight_decay"]) for group in optimizer_param_groups
        )
        lr_scheduler = _build_learning_rate_scheduler(
            optimizer=optimizer,
            lr_group="main",
            warmup_steps=lr_warmup_steps,
            total_steps=resolved_steps,
            min_learning_rate=min_learning_rate,
            base_learning_rate=learning_rate,
            torch_module=torch,
        )
        router_lr_scheduler = None
        if router_trainable_parameters:
            router_lr_scheduler = _build_learning_rate_scheduler(
                optimizer=optimizer,
                lr_group="router",
                warmup_steps=router_lr_warmup_steps,
                total_steps=router_lr_warmup_steps + router_lr_anneal_steps,
                min_learning_rate=router_min_learning_rate,
                base_learning_rate=router_learning_rate,
                torch_module=torch,
            )
        _restore_resume_state(
            resume_state=resume_state,
            optimizer=optimizer,
            optimizer_weight_decays=optimizer_weight_decays,
            learning_rates_by_group={
                "main": learning_rate,
                "router": router_learning_rate,
            },
            lr_scheduler=lr_scheduler,
            router_lr_scheduler=router_lr_scheduler,
            ind_stream=ind_stream,
            ood_stream=ood_stream,
            scheduler=scheduler,
            route_bias_annealer=route_bias_annealer,
        )
        optimizer.zero_grad(set_to_none=True)
        student_runtime = _compile_model_for_runtime(
            model=student,
            enabled=resolved_torch_compile,
            torch_module=torch,
        )
        router_metric_collector = _RouterRoutingMetricCollector(torch_module=torch)
        router_metric_collector.register(student)
        student_runtime.train()
        _restore_training_random_generator_state(
            resume_state=resume_state,
            torch_module=torch,
        )

        progress = tqdm(range(resume_step + 1, resolved_steps + 1), desc="Training", unit="step")
        latest_eval_loss = _resume_state_latest_eval_loss(resume_state)
        for step in progress:
            step_started_at = time.perf_counter()
            route_logit_bias_scale = route_bias_annealer.scale()
            effective_ind_route_logit_bias = ind_route_logit_bias * route_logit_bias_scale
            effective_ood_route_logit_bias = ood_route_logit_bias * route_logit_bias_scale
            step_metrics = {
                "loss": 0.0,
                "task_loss": 0.0,
                "lm_loss": 0.0,
                "distill_loss": 0.0,
                "route_loss": 0.0,
                "weighted_lm_loss": 0.0,
                "weighted_distill_loss": 0.0,
                "weighted_route_loss": 0.0,
            }
            timing_metrics = {
                "data_s": 0.0,
                "tokenize_s": 0.0,
                "move_s": 0.0,
                "student_forward_s": 0.0,
                "teacher_forward_s": 0.0,
                "lm_loss_s": 0.0,
                "distill_loss_s": 0.0,
                "backward_s": 0.0,
                "optimizer_s": 0.0,
                "eval_s": 0.0,
                "mps_cleanup_s": 0.0,
            }
            language_metrics = {
                "ind": {
                    "batches": 0,
                    "loss": 0.0,
                    "task_loss": 0.0,
                    "lm_loss": 0.0,
                    "distill_loss": 0.0,
                    "route_loss": 0.0,
                    "weighted_lm_loss": 0.0,
                    "weighted_distill_loss": 0.0,
                    "weighted_route_loss": 0.0,
                },
                "ood": {
                    "batches": 0,
                    "loss": 0.0,
                    "task_loss": 0.0,
                    "lm_loss": 0.0,
                    "distill_loss": 0.0,
                    "route_loss": 0.0,
                    "weighted_lm_loss": 0.0,
                    "weighted_distill_loss": 0.0,
                    "weighted_route_loss": 0.0,
                },
            }
            if router_metric_collector is not None:
                router_metric_collector.reset()
            for _ in range(gradient_accumulation_steps):
                language = scheduler.next_language()
                t0 = time.perf_counter()
                batch_texts = _next_text_batch(
                    language=language,
                    batch_size=batch_size,
                    ind_stream=ind_stream,
                    ood_stream=ood_stream,
                )
                timing_metrics["data_s"] += time.perf_counter() - t0
                weights = ind_weights if language == "ind" else ood_weights
                t0 = time.perf_counter()
                encoded = _tokenize_batch(
                    tokenizer=tokenizer,
                    batch_texts=batch_texts,
                    max_length=max_length,
                    device=resolved_device,
                    pad_to_max_length=pad_to_max_length,
                )
                timing_metrics["tokenize_s"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                encoded = _move_to_device(encoded, resolved_device)
                timing_metrics["move_s"] += time.perf_counter() - t0
                student_outputs = None
                student_logits = None
                teacher_outputs = None
                teacher_logits = None
                lm_loss = None
                distill_loss = None
                route_loss = None
                weighted_route_loss = None
                total_loss = None
                try:
                    t0 = time.perf_counter()
                    if router_metric_collector is not None:
                        expert1_logit_bias = (
                            effective_ind_route_logit_bias
                            if language == "ind"
                            else effective_ood_route_logit_bias
                        )
                        router_metric_collector.activate(
                            language,
                            expert1_logit_bias=expert1_logit_bias,
                        )
                    student_outputs = student_runtime(
                        **encoded,
                        use_cache=False,
                    )
                    if router_metric_collector is not None:
                        router_metric_collector.deactivate()
                    timing_metrics["student_forward_s"] += time.perf_counter() - t0
                    student_logits = student_outputs.logits
                    if weights.lm > 0:
                        t0 = time.perf_counter()
                        lm_loss = _compute_language_model_loss(
                            logits=student_logits,
                            input_ids=encoded["input_ids"],
                            attention_mask=encoded["attention_mask"],
                            torch_module=torch,
                        )
                        timing_metrics["lm_loss_s"] += time.perf_counter() - t0
                    else:
                        lm_loss = student_logits.new_zeros(())
                    should_distill = _should_run_distillation(
                        step=step,
                        language=language,
                        distill_ind=distill_ind,
                        distill_ood=distill_ood,
                        distill_every=distill_every,
                        distill_weight=weights.distill,
                    )
                    if should_distill:
                        if teacher is None:
                            raise ValueError("Teacher model is required when distillation loss weight is positive")
                        t0 = time.perf_counter()
                        with torch.inference_mode():
                            teacher_outputs = teacher(
                                **encoded,
                                use_cache=False,
                            )
                        timing_metrics["teacher_forward_s"] += time.perf_counter() - t0
                        teacher_logits = teacher_outputs.logits
                        t0 = time.perf_counter()
                        distill_loss = _compute_distillation_loss(
                            student_logits=student_logits,
                            teacher_logits=teacher_logits,
                            attention_mask=encoded["attention_mask"],
                            vocab_index=distillation_vocab_index,
                            vocab_chunk_size=distill_kl_vocab_chunk_size,
                            torch_module=torch,
                        )
                        timing_metrics["distill_loss_s"] += time.perf_counter() - t0
                    else:
                        distill_loss = student_logits.new_zeros(())
                    if router_metric_collector is not None and weights.route > 0:
                        route_loss = router_metric_collector.consume_loss(language=language)
                    if route_loss is None:
                        route_loss = student_logits.new_zeros(())
                    task_loss = weights.lm * lm_loss + weights.distill * distill_loss
                    weighted_route_loss = weights.route * route_loss
                    total_loss = task_loss + weighted_route_loss
                    t0 = time.perf_counter()
                    has_route_backward = _loss_requires_grad(weighted_route_loss) and bool(router_trainable_parameters)
                    _backward_loss_for_parameters(
                        loss=task_loss,
                        parameters=task_trainable_parameters,
                        gradient_scale=1.0 / gradient_accumulation_steps,
                        retain_graph=has_route_backward,
                        torch_module=torch,
                    )
                    _backward_loss_for_parameters(
                        loss=weighted_route_loss,
                        parameters=router_trainable_parameters,
                        gradient_scale=1.0 / gradient_accumulation_steps,
                        retain_graph=False,
                        torch_module=torch,
                    )
                    timing_metrics["backward_s"] += time.perf_counter() - t0
                    total_loss_value = float(total_loss.detach().cpu())
                    task_loss_value = float(task_loss.detach().cpu())
                    lm_loss_value = float(lm_loss.detach().cpu())
                    distill_loss_value = float(distill_loss.detach().cpu())
                    route_loss_value = float(route_loss.detach().cpu())
                    weighted_lm_loss_value = weights.lm * lm_loss_value
                    weighted_distill_loss_value = weights.distill * distill_loss_value
                    weighted_route_loss_value = weights.route * route_loss_value
                    step_metrics["loss"] += total_loss_value
                    step_metrics["task_loss"] += task_loss_value
                    step_metrics["lm_loss"] += lm_loss_value
                    step_metrics["distill_loss"] += distill_loss_value
                    step_metrics["route_loss"] += route_loss_value
                    step_metrics["weighted_lm_loss"] += weighted_lm_loss_value
                    step_metrics["weighted_distill_loss"] += weighted_distill_loss_value
                    step_metrics["weighted_route_loss"] += weighted_route_loss_value
                    language_metrics[language]["batches"] += 1
                    language_metrics[language]["loss"] += total_loss_value
                    language_metrics[language]["task_loss"] += task_loss_value
                    language_metrics[language]["lm_loss"] += lm_loss_value
                    language_metrics[language]["distill_loss"] += distill_loss_value
                    language_metrics[language]["route_loss"] += route_loss_value
                    language_metrics[language]["weighted_lm_loss"] += weighted_lm_loss_value
                    language_metrics[language]["weighted_distill_loss"] += weighted_distill_loss_value
                    language_metrics[language]["weighted_route_loss"] += weighted_route_loss_value
                finally:
                    if router_metric_collector is not None:
                        router_metric_collector.deactivate()
                    total_loss = None
                    weighted_route_loss = None
                    task_loss = None
                    route_loss = None
                    distill_loss = None
                    lm_loss = None
                    teacher_logits = None
                    teacher_outputs = None
                    student_logits = None
                    student_outputs = None
                    encoded = None
                    t0 = time.perf_counter()
                    _release_mps_working_set(
                        torch_module=torch,
                        device=resolved_device,
                    )
                    timing_metrics["mps_cleanup_s"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            main_grad_norm = _parameter_grad_norm(task_trainable_parameters, torch_module=torch)
            router_grad_norm = _parameter_grad_norm(router_trainable_parameters, torch_module=torch)
            _clip_parameter_grad_norm(
                task_trainable_parameters,
                max_grad_norm=max_grad_norm,
                torch_module=torch,
            )
            _clip_parameter_grad_norm(
                router_trainable_parameters,
                max_grad_norm=max_grad_norm,
                torch_module=torch,
            )
            main_grad_norm_post_clip = _parameter_grad_norm(task_trainable_parameters, torch_module=torch)
            router_grad_norm_post_clip = _parameter_grad_norm(router_trainable_parameters, torch_module=torch)
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if router_lr_scheduler is not None:
                router_lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            _release_mps_working_set(
                torch_module=torch,
                device=resolved_device,
            )
            timing_metrics["optimizer_s"] += time.perf_counter() - t0
            ind_batches = int(language_metrics["ind"]["batches"])
            ood_batches = int(language_metrics["ood"]["batches"])
            mix_label = f"{ind_batches}ind/{ood_batches}ood"
            ind_loss = _average_language_metric(language_metrics["ind"], "loss")
            ood_loss = _average_language_metric(language_metrics["ood"], "loss")
            router_metrics = router_metric_collector.averages() if router_metric_collector is not None else {}
            average_route_loss = step_metrics["route_loss"] / gradient_accumulation_steps
            average_weighted_route_loss = step_metrics["weighted_route_loss"] / gradient_accumulation_steps
            route_logit_bias_annealed = route_bias_annealer.observe_route_loss(average_weighted_route_loss)
            postfix = {
                "mix": mix_label,
                "obj": f"{step_metrics['loss'] / gradient_accumulation_steps:.4f}",
                "task": f"{step_metrics['task_loss'] / gradient_accumulation_steps:.4f}",
                "route": f"{average_route_loss:.4f}",
                "ind": _format_optional_metric(ind_loss),
                "ood": _format_optional_metric(ood_loss),
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
            current_router_learning_rate = _optimizer_group_learning_rate(
                optimizer=optimizer,
                lr_group="router",
            )
            if current_router_learning_rate is not None:
                postfix["rlr"] = f"{current_router_learning_rate:.2e}"
            if route_logit_bias_anneal_steps > 0:
                postfix["bias"] = f"{route_logit_bias_scale:.3f}"
                postfix["bias_n"] = str(route_bias_annealer.completed_anneal_steps)
            for key, postfix_key in (
                ("ind_router_weight_expert1", "ind_r1"),
                ("ood_router_weight_expert1", "ood_r1"),
            ):
                value = router_metrics.get(key)
                if isinstance(value, float):
                    postfix[postfix_key] = f"{value:.6f}"
            current_eval_loss: float | None = None
            if eval_files and eval_every_steps > 0 and step % eval_every_steps == 0:
                t0 = time.perf_counter()
                current_eval_loss = _evaluate_language_model_loss(
                    model=student_runtime,
                    tokenizer=tokenizer,
                    eval_files=eval_files,
                    batch_size=resolved_eval_batch_size,
                    max_batches=eval_max_batches,
                    max_length=max_length,
                    device=resolved_device,
                    pad_to_max_length=pad_to_max_length,
                    torch_module=torch,
                )
                timing_metrics["eval_s"] += time.perf_counter() - t0
                latest_eval_loss = current_eval_loss
                postfix["eval_lm"] = f"{current_eval_loss:.4f}"
            step_total_s = time.perf_counter() - step_started_at
            postfix["t"] = f"{step_total_s:.1f}s"
            mps_memory_snapshot = _capture_mps_memory_snapshot(
                torch_module=torch,
                device=resolved_device,
            )
            if mps_memory_snapshot is not None:
                postfix["mps"] = _format_mps_progress_value(mps_memory_snapshot)
            weight_diff_metrics = {}
            if weight_diff_every > 0 and step % weight_diff_every == 0:
                weight_diff_metrics = _compute_expert_pair_weight_diff_metrics(
                    model=student,
                    torch_module=torch,
                )
            metric_record = {
                "step": step,
                "mix": mix_label,
                "ind_batches": ind_batches,
                "ood_batches": ood_batches,
                "train_loss": step_metrics["loss"] / gradient_accumulation_steps,
                "train_reported_objective": step_metrics["loss"] / gradient_accumulation_steps,
                "train_task_loss": step_metrics["task_loss"] / gradient_accumulation_steps,
                "train_task_objective": step_metrics["task_loss"] / gradient_accumulation_steps,
                "train_lm_loss": step_metrics["lm_loss"] / gradient_accumulation_steps,
                "train_distill_loss": step_metrics["distill_loss"] / gradient_accumulation_steps,
                "train_route_loss": average_route_loss,
                "train_router_objective": average_weighted_route_loss,
                "train_weighted_lm_loss": step_metrics["weighted_lm_loss"] / gradient_accumulation_steps,
                "train_weighted_distill_loss": step_metrics["weighted_distill_loss"] / gradient_accumulation_steps,
                "train_weighted_route_loss": average_weighted_route_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "main_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "min_learning_rate": min_learning_rate,
                "router_learning_rate": current_router_learning_rate,
                "router_lr_warmup_steps": router_lr_warmup_steps,
                "max_grad_norm": max_grad_norm,
                "main_grad_norm": main_grad_norm,
                "router_grad_norm": router_grad_norm,
                "main_grad_norm_pre_clip": main_grad_norm,
                "router_grad_norm_pre_clip": router_grad_norm,
                "main_grad_norm_post_clip": main_grad_norm_post_clip,
                "router_grad_norm_post_clip": router_grad_norm_post_clip,
                "ind_train_loss": ind_loss,
                "ind_train_task_loss": _average_language_metric(language_metrics["ind"], "task_loss"),
                "ind_train_task_objective": _average_language_metric(language_metrics["ind"], "task_loss"),
                "ind_train_lm_loss": _average_language_metric(language_metrics["ind"], "lm_loss"),
                "ind_train_distill_loss": _average_language_metric(language_metrics["ind"], "distill_loss"),
                "ind_train_route_loss": _average_language_metric(language_metrics["ind"], "route_loss"),
                "ind_train_router_objective": _average_language_metric(language_metrics["ind"], "weighted_route_loss"),
                "ind_train_weighted_lm_loss": _average_language_metric(language_metrics["ind"], "weighted_lm_loss"),
                "ind_train_weighted_distill_loss": _average_language_metric(
                    language_metrics["ind"],
                    "weighted_distill_loss",
                ),
                "ind_train_weighted_route_loss": _average_language_metric(
                    language_metrics["ind"],
                    "weighted_route_loss",
                ),
                "ood_train_loss": ood_loss,
                "ood_train_task_loss": _average_language_metric(language_metrics["ood"], "task_loss"),
                "ood_train_task_objective": _average_language_metric(language_metrics["ood"], "task_loss"),
                "ood_train_lm_loss": _average_language_metric(language_metrics["ood"], "lm_loss"),
                "ood_train_distill_loss": _average_language_metric(language_metrics["ood"], "distill_loss"),
                "ood_train_route_loss": _average_language_metric(language_metrics["ood"], "route_loss"),
                "ood_train_router_objective": _average_language_metric(language_metrics["ood"], "weighted_route_loss"),
                "ood_train_weighted_lm_loss": _average_language_metric(language_metrics["ood"], "weighted_lm_loss"),
                "ood_train_weighted_distill_loss": _average_language_metric(
                    language_metrics["ood"],
                    "weighted_distill_loss",
                ),
                "ood_train_weighted_route_loss": _average_language_metric(
                    language_metrics["ood"],
                    "weighted_route_loss",
                ),
                "eval_lm_loss": current_eval_loss,
                "latest_eval_lm_loss": latest_eval_loss,
                "route_logit_bias_scale": route_logit_bias_scale,
                "route_logit_bias_anneal_progress": route_bias_annealer.completed_anneal_steps,
                "route_logit_bias_anneal_observed_steps": route_bias_annealer.observed_steps,
                "route_logit_bias_anneal_offset_steps": route_logit_bias_anneal_offset_steps,
                "route_logit_bias_anneal_objective": average_weighted_route_loss,
                "route_logit_bias_anneal_loss_threshold": route_logit_bias_anneal_loss_threshold,
                "route_logit_bias_annealed": route_logit_bias_annealed,
                "ind_route_logit_bias": effective_ind_route_logit_bias,
                "ood_route_logit_bias": effective_ood_route_logit_bias,
                "step_total_s": step_total_s,
            }
            metric_record.update(router_metrics)
            metric_record.update(weight_diff_metrics)
            metric_record.update(timing_metrics)
            if mps_memory_snapshot is not None:
                metric_record.update(_serialize_mps_memory_snapshot(mps_memory_snapshot))
            if step % log_every == 0:
                _append_jsonl(metrics_log_path, metric_record)
            if step % print_every == 0:
                _print_training_metrics(metric_record)
            progress.set_postfix(**postfix)
            if (
                checkpointing_enabled
                and resolved_checkpoint_dir is not None
                and step % checkpoint_every == 0
            ):
                _save_training_checkpoint(
                    model=student,
                    tokenizer=tokenizer,
                    source_model=student_source,
                    checkpoint_path=resolved_checkpoint_dir / f"step{step}",
                    step_label=f"step {step}",
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    router_lr_scheduler=router_lr_scheduler,
                    ind_stream=ind_stream,
                    ood_stream=ood_stream,
                    scheduler=scheduler,
                    route_bias_annealer=route_bias_annealer,
                    step=step,
                    latest_eval_loss=latest_eval_loss,
                    training_configuration_state=training_configuration_state,
                    torch_module=torch,
                )

        student.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        _copy_auxiliary_tokenizer_files(student_model=student_source, output_path=output_path)
        _save_training_state(
            checkpoint_path=output_path,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            router_lr_scheduler=router_lr_scheduler,
            ind_stream=ind_stream,
            ood_stream=ood_stream,
            scheduler=scheduler,
            route_bias_annealer=route_bias_annealer,
            step=resolved_steps,
            latest_eval_loss=latest_eval_loss,
            training_configuration_state=training_configuration_state,
            torch_module=torch,
        )
        if checkpointing_enabled and resolved_checkpoint_dir is not None:
            _save_training_checkpoint(
                model=student,
                tokenizer=tokenizer,
                source_model=student_source,
                checkpoint_path=resolved_checkpoint_dir / "final",
                step_label="final",
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                router_lr_scheduler=router_lr_scheduler,
                ind_stream=ind_stream,
                ood_stream=ood_stream,
                scheduler=scheduler,
                route_bias_annealer=route_bias_annealer,
                step=resolved_steps,
                latest_eval_loss=latest_eval_loss,
                training_configuration_state=training_configuration_state,
                torch_module=torch,
            )
        _write_training_recipe(
            output_path=output_path,
            student_model=student_model,
            teacher_model=teacher_model,
            resume_from=resume_from,
            requested_steps=steps,
            resolved_steps=resolved_steps,
            resolved_epochs=resolved_epochs,
            ind_files=ind_files,
            ood_files=ood_files,
            eval_files=eval_files,
            resolved_device=resolved_device,
            dtype=dtype,
            steps=resolved_steps,
            batch_size=batch_size,
            eval_batch_size=resolved_eval_batch_size,
            eval_max_batches=eval_max_batches,
            weight_diff_every=weight_diff_every,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            torch_compile=resolved_torch_compile,
            lr_warmup_steps=lr_warmup_steps,
            max_length=max_length,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            ind_route_weight=ind_route_weight,
            ind_route_logit_bias=ind_route_logit_bias,
            ood_route_weight=ood_route_weight,
            ood_route_logit_bias=ood_route_logit_bias,
            route_logit_bias_anneal_steps=route_logit_bias_anneal_steps,
            route_logit_bias_anneal_offset_steps=route_logit_bias_anneal_offset_steps,
            route_logit_bias_anneal_loss_threshold=route_logit_bias_anneal_loss_threshold,
            router_learning_rate=router_learning_rate,
            router_lr_warmup_steps=router_lr_warmup_steps,
            router_lr_anneal_steps=router_lr_anneal_steps,
            router_min_learning_rate=router_min_learning_rate,
            parameter_selection=parameter_selection,
            distill_kl_vocab_chunk_size=distill_kl_vocab_chunk_size,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=resolved_checkpoint_dir,
            eval_every_steps=eval_every_steps,
            log_every=log_every,
            print_every=print_every,
            pad_to_max_length=pad_to_max_length,
            distill_ind=distill_ind,
            distill_ood=distill_ood,
            distill_original_tokens_only=distill_original_tokens_only,
            distill_every=distill_every,
            metrics_log_path=metrics_log_path,
            ind_batches_per_cycle=ind_batches_per_cycle,
            ood_batches_per_cycle=ood_batches_per_cycle,
            ind_weights=ind_weights,
            ood_weights=ood_weights,
            seed=seed,
            token_groups=token_groups,
        )
        return output_path.resolve()
    finally:
        if router_metric_collector is not None:
            router_metric_collector.close()
            router_metric_collector = None
        if optimizer is not None:
            del optimizer
            optimizer = None
        if lr_scheduler is not None:
            del lr_scheduler
            lr_scheduler = None
        if teacher is not None:
            del teacher
            teacher = None
        if student_runtime is not None:
            del student_runtime
            student_runtime = None
        if student is not None:
            del student
            student = None
        if teacher_tokenizer is not None:
            del teacher_tokenizer
            teacher_tokenizer = None
        if tokenizer is not None:
            del tokenizer
            tokenizer = None
        if ind_stream is not None:
            ind_stream.close()
        if ood_stream is not None:
            ood_stream.close()
        _release_resources(torch_module=torch)


@dataclass(slots=True)
class _CorpusStream:
    paths: tuple[Path, ...]
    rng: random.Random
    _path_order: list[Path] = field(init=False, repr=False)
    _path_index: int = field(init=False, repr=False)
    _current_handle: TextIOWrapper | Any | None = field(init=False, repr=False)
    _current_path: Path | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.paths:
            raise ValueError("Corpus stream requires at least one input file")
        self._path_order = [path.expanduser() for path in self.paths]
        self.rng.shuffle(self._path_order)
        self._path_index = 0
        self._current_handle = None
        self._current_path = None

    def next_batch(self, batch_size: int) -> tuple[str, ...]:
        return tuple(self._next_example() for _ in range(batch_size))

    def close(self) -> None:
        if self._current_handle is not None:
            self._current_handle.close()
            self._current_handle = None
            self._current_path = None

    def state_dict(self) -> dict[str, Any]:
        current_offset: int | None = None
        if self._current_handle is not None and hasattr(self._current_handle, "tell"):
            current_offset = int(self._current_handle.tell())
        return {
            "paths": [str(path) for path in self.paths],
            "path_order": [str(path) for path in self._path_order],
            "path_index": self._path_index,
            "current_path": str(self._current_path) if self._current_path is not None else None,
            "current_offset": current_offset,
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        saved_paths = state_dict.get("paths")
        expected_paths = [str(path.expanduser()) for path in self.paths]
        if saved_paths != expected_paths:
            raise ValueError("Cannot resume with different corpus files")
        path_order = state_dict.get("path_order")
        if not isinstance(path_order, list) or not all(isinstance(item, str) for item in path_order):
            raise ValueError("training-state path_order must be a list of strings")
        if set(path_order) != set(expected_paths):
            raise ValueError("training-state path_order does not match corpus files")
        path_index = state_dict.get("path_index")
        if not isinstance(path_index, int) or not (0 <= path_index < len(path_order)):
            raise ValueError("training-state path_index is invalid")
        current_path = state_dict.get("current_path")
        if current_path is not None and current_path not in path_order:
            raise ValueError("training-state current_path is invalid")
        current_offset = state_dict.get("current_offset")
        if current_offset is not None and (not isinstance(current_offset, int) or current_offset < 0):
            raise ValueError("training-state current_offset must be a non-negative integer or null")
        self.close()
        self._path_order = [Path(path) for path in path_order]
        self._path_index = path_index
        self._current_path = None
        self._current_handle = None
        rng_state = state_dict.get("rng_state")
        if rng_state is None:
            raise ValueError("training-state rng_state is missing")
        self.rng.setstate(rng_state)
        if current_path is not None and current_offset is not None:
            self._current_path = Path(current_path)
            self._current_handle = _open_text_handle(self._current_path)
            self._current_handle.seek(current_offset)

    def _next_example(self) -> str:
        visited_paths = 0
        while visited_paths < len(self._path_order):
            handle = self._ensure_open_handle()
            raw_line = handle.readline()
            if raw_line == "":
                self._advance_path()
                visited_paths += 1
                continue
            parsed = _parse_corpus_line(self._current_path, raw_line)
            if parsed is None:
                continue
            return parsed
        raise ValueError("Corpus files do not contain any usable examples")

    def _ensure_open_handle(self) -> Any:
        if self._current_handle is None:
            self._current_path = self._path_order[self._path_index]
            self._current_handle = _open_text_handle(self._current_path)
        return self._current_handle

    def _advance_path(self) -> None:
        self.close()
        self._path_index = (self._path_index + 1) % len(self._path_order)
        if self._path_index == 0:
            self.rng.shuffle(self._path_order)


@dataclass(slots=True)
class _LanguageScheduler:
    ind_enabled: bool
    ood_enabled: bool
    ind_batches_per_cycle: int
    ood_batches_per_cycle: int
    _cycle: tuple[str, ...] = field(init=False, repr=False)
    _position: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        cycle: list[str] = []
        if self.ind_enabled and self.ind_batches_per_cycle > 0:
            cycle.extend(["ind"] * self.ind_batches_per_cycle)
        if self.ood_enabled and self.ood_batches_per_cycle > 0:
            cycle.extend(["ood"] * self.ood_batches_per_cycle)
        if not cycle:
            raise ValueError("No enabled language streams are available")
        self._cycle = tuple(cycle)
        self._position = 0

    def next_language(self) -> str:
        language = self._cycle[self._position]
        self._position = (self._position + 1) % len(self._cycle)
        return language

    def language_fraction(self, language: str) -> float:
        return sum(entry == language for entry in self._cycle) / len(self._cycle)

    def state_dict(self) -> dict[str, Any]:
        return {
            "cycle": list(self._cycle),
            "position": self._position,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> bool:
        cycle = state_dict.get("cycle")
        position = state_dict.get("position")
        if cycle != list(self._cycle):
            return False
        if not isinstance(position, int) or not (0 <= position < len(self._cycle)):
            raise ValueError("training-state scheduler position is invalid")
        self._position = position
        return True


def _load_corpus_examples(paths: tuple[Path, ...]) -> tuple[str, ...]:
    examples: list[str] = []
    for path in paths:
        resolved_path = path.expanduser()
        if _is_jsonl_path(resolved_path):
            examples.extend(_load_jsonl_texts(resolved_path))
        else:
            examples.extend(_load_line_texts(resolved_path))
    return tuple(example for example in examples if example)


def _load_jsonl_texts(path: Path) -> list[str]:
    return list(_iterate_examples_in_path(path))


def _load_line_texts(path: Path) -> list[str]:
    return list(_iterate_examples_in_path(path))


def _is_jsonl_path(path: Path) -> bool:
    name = path.name.casefold()
    return name.endswith(".jsonl") or name.endswith(".jsonl.gz")


def _iterate_examples_once(paths: tuple[Path, ...]) -> Any:
    for path in paths:
        yield from _iterate_examples_in_path(path.expanduser())


def _iterate_examples_in_path(path: Path) -> Any:
    if _is_jinx_path(path):
        yield from _iterate_examples_in_jinx_path(path)
        return
    with _open_text_handle(path) as handle:
        for raw_line in handle:
            parsed = _parse_corpus_line(path, raw_line)
            if parsed is not None:
                yield parsed


def _iterate_examples_in_jinx_path(path: Path) -> Any:
    reader = _open_jinx_reader(path)
    try:
        for sample in reader:
            yield _parse_jinx_sample(path, sample)
    finally:
        _close_jinx_reader(reader)


def _open_text_handle(path: Path) -> Any:
    if path.name.casefold().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_jinx_reader(path: Path) -> Any:
    from mldataforge.jinx.dataset_reader import JinxDatasetReader

    return JinxDatasetReader(path, lazy=False)


def _close_jinx_reader(reader: Any) -> None:
    for shard in getattr(reader, "shards", ()):
        _close_jinx_shard(shard)


def _close_jinx_shard(shard: Any) -> None:
    mmap_object = getattr(shard, "mmap", None)
    if mmap_object:
        mmap_object.close()
    else:
        file_object = getattr(shard, "file", None)
        if file_object is not None:
            file_object.close()
    index = getattr(shard, "offsets", None)
    index_mmap = getattr(index, "_mmap", None)
    if index_mmap is not None:
        index_mmap.close()
    index_tmp = getattr(shard, "_index_tmp", None)
    if isinstance(index_tmp, str) and os.path.exists(index_tmp):
        os.remove(index_tmp)
    bin_file = getattr(shard, "bin", None)
    if bin_file is not None:
        bin_file.close()


def _is_jinx_path(path: Path) -> bool:
    if path.name.casefold().endswith(".jinx"):
        return True
    return path.is_dir() and any(path.glob("shard-*.jinx"))


def _parse_corpus_line(path: Path | None, raw_line: str) -> str | None:
    stripped = raw_line.strip()
    if not stripped:
        return None
    if path is not None and _is_jsonl_path(path):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            raise ValueError(f"Expected JSON object with string text in {path}")
        return data["text"]
    return stripped


def _parse_jinx_sample(path: Path, sample: Any) -> str:
    if not isinstance(sample, dict) or not isinstance(sample.get("text"), str):
        raise ValueError(f"Expected JINX sample with string text in {path}")
    return sample["text"]


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_int_tuple(data: dict[str, Any], key: str) -> tuple[int, ...]:
    raw_value = data.get(key, [])
    if not isinstance(raw_value, list) or not all(isinstance(item, int) for item in raw_value):
        raise ValueError(f"{key} must be a list of integers")
    return tuple(raw_value)


def _load_str_tuple(data: dict[str, Any], key: str) -> tuple[str, ...]:
    raw_value = data.get(key, [])
    if not isinstance(raw_value, list) or not all(isinstance(item, str) for item in raw_value):
        raise ValueError(f"{key} must be a list of strings")
    return tuple(raw_value)


def _validate_matching_tokenizers(*, tokenizer: Any, teacher_tokenizer: Any) -> None:
    student_vocab = tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    if student_vocab != teacher_vocab:
        raise ValueError(
            "Student and teacher tokenizers must match exactly for distillation; "
            "use a teacher with the same tokenizer as the student"
        )


def _prepare_training_tokenizer(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "right"


def _enable_gradient_checkpointing(model: Any) -> None:
    if not getattr(model, "supports_gradient_checkpointing", False):
        return
    if not hasattr(model, "gradient_checkpointing_enable"):
        return
    model.gradient_checkpointing_enable()


def _is_expert_weight_parameter(name: str) -> bool:
    return ".experts.gate_up_proj" in name or ".experts.down_proj" in name


def _is_router_parameter(name: str) -> bool:
    return ".router." in name


def _mask_frozen_expert_gradients(
    *,
    name: str,
    parameter: Any,
    torch_module: Any,
    train_expert_0: bool,
    train_expert_1: bool,
) -> None:
    if not _is_expert_weight_parameter(name):
        return
    if parameter.ndim < 1 or int(parameter.shape[0]) < 2:
        return
    mask = torch_module.zeros_like(parameter)
    if train_expert_0:
        mask[0] = 1
    if train_expert_1:
        mask[1] = 1
    parameter.register_hook(lambda grad, expert_mask=mask: grad * expert_mask.to(dtype=grad.dtype))


def _enable_added_token_rows(
    *,
    model: Any,
    token_ids: tuple[int, ...],
    torch_module: Any,
) -> None:
    if not token_ids:
        return
    handled_parameter_ids: set[int] = set()
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    for module in (input_embeddings, output_embeddings):
        if module is None or not hasattr(module, "weight"):
            continue
        parameter = module.weight
        parameter_id = id(parameter)
        if parameter_id in handled_parameter_ids:
            continue
        handled_parameter_ids.add(parameter_id)
        parameter.requires_grad_(True)
        mask = _build_row_mask(
            row_count=parameter.shape[0],
            token_ids=token_ids,
            torch_module=torch_module,
            device=parameter.device,
        )
        parameter.register_hook(lambda grad, row_mask=mask: grad * row_mask.to(dtype=grad.dtype))


def _enable_all_embedding_lm_head(*, model: Any) -> None:
    handled_parameter_ids: set[int] = set()
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    for module in (input_embeddings, output_embeddings):
        if module is None or not hasattr(module, "weight"):
            continue
        parameter = module.weight
        parameter_id = id(parameter)
        if parameter_id in handled_parameter_ids:
            continue
        handled_parameter_ids.add(parameter_id)
        parameter.requires_grad_(True)


def _masked_row_parameter_ids(model: Any) -> frozenset[int]:
    parameter_ids: set[int] = set()
    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is None or not hasattr(module, "weight"):
            continue
        parameter_ids.add(id(module.weight))
    return frozenset(parameter_ids)


def _build_optimizer_param_groups(
    *,
    standard_trainable_parameters: list[Any],
    router_trainable_parameters: list[Any],
    masked_row_parameters: list[Any],
    weight_decay: float,
    learning_rate: float,
    router_learning_rate: float,
) -> list[dict[str, Any]]:
    optimizer_param_groups: list[dict[str, Any]] = []
    if standard_trainable_parameters:
        optimizer_param_groups.append(
            {
                "params": standard_trainable_parameters,
                "weight_decay": weight_decay,
                "lr": learning_rate,
                "lr_group": "main",
            }
        )
    if router_trainable_parameters:
        optimizer_param_groups.append(
            {
                "params": router_trainable_parameters,
                "weight_decay": weight_decay,
                "lr": router_learning_rate,
                "lr_group": "router",
            }
        )
    if masked_row_parameters:
        # AdamW applies decoupled weight decay to whole tensors, so keep
        # the row-masked embedding/head matrices at zero decay.
        optimizer_param_groups.append(
            {
                "params": masked_row_parameters,
                "weight_decay": 0.0,
                "lr": learning_rate,
                "lr_group": "main",
            }
        )
    return optimizer_param_groups


def _serialize_optimizer_groups(
    *,
    standard_trainable_parameters: list[Any],
    router_trainable_parameters: list[Any],
    masked_row_parameters: list[Any],
    named_parameters: dict[str, Any],
    weight_decay: float,
    learning_rate: float,
    router_learning_rate: float,
) -> tuple[dict[str, Any], ...]:
    parameter_name_by_id = {id(parameter): name for name, parameter in named_parameters.items()}
    groups = _build_optimizer_param_groups(
        standard_trainable_parameters=standard_trainable_parameters,
        router_trainable_parameters=router_trainable_parameters,
        masked_row_parameters=masked_row_parameters,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        router_learning_rate=router_learning_rate,
    )
    serialized_groups = []
    for group in groups:
        serialized_groups.append(
            {
                "weight_decay": group["weight_decay"],
                "learning_rate": group["lr"],
                "lr_group": group["lr_group"],
                "parameter_names": sorted(
                    parameter_name_by_id[id(parameter)]
                    for parameter in group["params"]
                ),
            }
        )
    return tuple(serialized_groups)


def _optimizer_group_learning_rate(*, optimizer: Any, lr_group: str) -> float | None:
    for group in optimizer.param_groups:
        if group.get("lr_group") == lr_group:
            return float(group["lr"])
    return None


def _loss_requires_grad(loss: Any) -> bool:
    return bool(getattr(loss, "requires_grad", False))


def _backward_loss_for_parameters(
    *,
    loss: Any,
    parameters: list[Any],
    gradient_scale: float,
    retain_graph: bool,
    torch_module: Any,
) -> None:
    if not parameters or not _loss_requires_grad(loss):
        return
    trainable_parameters = tuple(
        parameter
        for parameter in parameters
        if bool(getattr(parameter, "requires_grad", False))
    )
    if not trainable_parameters:
        return
    torch_module.autograd.backward(
        loss * gradient_scale,
        retain_graph=retain_graph,
        inputs=trainable_parameters,
    )


def _parameter_grad_norm(parameters: list[Any], *, torch_module: Any) -> float | None:
    squared_norm = None
    for parameter in parameters:
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        detached_gradient = gradient.detach()
        contribution = detached_gradient.float().pow(2).sum()
        squared_norm = contribution if squared_norm is None else squared_norm + contribution
    if squared_norm is None:
        return None
    return float(torch_module.sqrt(squared_norm).cpu())


def _clip_parameter_grad_norm(
    parameters: list[Any],
    *,
    max_grad_norm: float,
    torch_module: Any,
) -> None:
    if max_grad_norm <= 0 or not parameters:
        return
    trainable_parameters = [
        parameter
        for parameter in parameters
        if bool(getattr(parameter, "requires_grad", False)) and getattr(parameter, "grad", None) is not None
    ]
    if not trainable_parameters:
        return
    torch_module.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=max_grad_norm)


def _build_row_mask(
    *,
    row_count: int,
    token_ids: tuple[int, ...],
    torch_module: Any,
    device: Any,
) -> Any:
    mask = torch_module.zeros((row_count, 1), device=device)
    for token_id in token_ids:
        if token_id < 0 or token_id >= row_count:
            raise ValueError(f"Token id {token_id} is out of range for a matrix with {row_count} rows")
        mask[token_id, 0] = 1
    return mask


def _next_text_batch(
    *,
    language: str,
    batch_size: int,
    ind_stream: _CorpusStream | None,
    ood_stream: _CorpusStream | None,
) -> tuple[str, ...]:
    if language == "ind":
        if ind_stream is None:
            raise ValueError("In-distribution stream is not configured")
        return ind_stream.next_batch(batch_size)
    if ood_stream is None:
        raise ValueError("Out-of-distribution stream is not configured")
    return ood_stream.next_batch(batch_size)


def _compute_language_model_loss(
    *,
    logits: Any,
    input_ids: Any,
    attention_mask: Any,
    torch_module: Any,
) -> Any:
    loss_sum, token_count = _compute_language_model_loss_components(
        logits=logits,
        input_ids=input_ids,
        attention_mask=attention_mask,
        torch_module=torch_module,
    )
    return loss_sum / token_count.clamp_min(1.0)


def _compute_language_model_loss_components(
    *,
    logits: Any,
    input_ids: Any,
    attention_mask: Any,
    torch_module: Any,
) -> tuple[Any, Any]:
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:].to(dtype=shift_logits.dtype)
    target_logits = shift_logits.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    per_token_loss = torch_module.logsumexp(shift_logits, dim=-1) - target_logits
    return (per_token_loss * shift_mask).sum(), shift_mask.sum()


def _compute_distillation_loss(
    *,
    student_logits: Any,
    teacher_logits: Any,
    attention_mask: Any,
    vocab_index: Any | None,
    vocab_chunk_size: int,
    torch_module: Any,
) -> Any:
    shift_student = student_logits[:, :-1, :]
    shift_teacher = teacher_logits[:, :-1, :]
    if vocab_index is not None:
        shift_student = shift_student.index_select(dim=-1, index=vocab_index)
        shift_teacher = shift_teacher.index_select(dim=-1, index=vocab_index)
    shift_mask = attention_mask[:, 1:].to(dtype=shift_student.dtype)
    if vocab_chunk_size == 0:
        student_log_probs = torch_module.log_softmax(shift_student, dim=-1)
        teacher_log_probs = torch_module.log_softmax(shift_teacher, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        per_token_kl = (
            teacher_probs * (teacher_log_probs - student_log_probs)
        ).sum(dim=-1)
    else:
        student_logsumexp = torch_module.logsumexp(shift_student, dim=-1, keepdim=True)
        teacher_logsumexp = torch_module.logsumexp(shift_teacher, dim=-1, keepdim=True)
        per_token_kl = shift_student.new_zeros(shift_student.shape[:-1])
        vocab_size = shift_student.shape[-1]
        for start in range(0, vocab_size, vocab_chunk_size):
            end = min(start + vocab_chunk_size, vocab_size)
            student_log_probs = shift_student[..., start:end] - student_logsumexp
            teacher_log_probs = shift_teacher[..., start:end] - teacher_logsumexp
            teacher_probs = teacher_log_probs.exp()
            per_token_kl = per_token_kl + (
                teacher_probs * (teacher_log_probs - student_log_probs)
            ).sum(dim=-1)
    return (per_token_kl * shift_mask).sum() / shift_mask.sum().clamp_min(1.0)


def _move_to_device(batch: Any, device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in dict(batch).items()
    }


def _tokenize_batch(
    *,
    tokenizer: Any,
    batch_texts: tuple[str, ...],
    max_length: int,
    device: str,
    pad_to_max_length: bool | None,
) -> Any:
    padding: bool | str = True
    # MPS tends to retain per-shape execution state, so keep sequence shapes fixed
    # unless the caller explicitly opts out.
    if pad_to_max_length is True or (pad_to_max_length is None and device == "mps"):
        padding = "max_length"
    return tokenizer(
        list(batch_texts),
        return_tensors="pt",
        padding=padding,
        truncation=True,
        max_length=max_length,
    )


def _resolve_torch_compile(*, requested: bool | None, device: str) -> bool:
    if requested is not None:
        return requested
    return device == "cuda"


def _compile_model_for_runtime(*, model: Any, enabled: bool, torch_module: Any) -> Any:
    if not enabled:
        return model
    compile_fn = getattr(torch_module, "compile", None)
    if not callable(compile_fn):
        raise ValueError("torch.compile is not available in this PyTorch build")
    return compile_fn(model)


def _build_learning_rate_scheduler(
    *,
    optimizer: Any,
    lr_group: str = "main",
    warmup_steps: int,
    total_steps: int,
    min_learning_rate: float,
    base_learning_rate: float,
    torch_module: Any,
) -> Any | None:
    if total_steps < 1:
        return None
    return _LearningRateScheduler(
        optimizer=optimizer,
        lr_group=lr_group,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_learning_rate=min_learning_rate,
        base_learning_rate=base_learning_rate,
    )


def _should_run_distillation(
    *,
    step: int,
    language: str,
    distill_ind: bool,
    distill_ood: bool,
    distill_every: int,
    distill_weight: float,
) -> bool:
    if distill_weight <= 0:
        return False
    if language == "ind":
        language_enabled = distill_ind
    else:
        language_enabled = distill_ood
    if not language_enabled:
        return False
    return step % distill_every == 0


def _validate_distillation_token_groups(
    *,
    token_groups: TokenGroupMetadata,
    distill_original_tokens_only: bool,
) -> None:
    if distill_original_tokens_only and not token_groups.original_token_ids:
        raise ValueError(
            "--distill-original-tokens-only requires tokenizer_token_groups.json "
            "with non-empty original_token_ids"
        )


def _build_distillation_vocab_index(
    *,
    token_groups: TokenGroupMetadata,
    distill_original_tokens_only: bool,
    device: str,
    torch_module: Any,
) -> Any | None:
    if not distill_original_tokens_only:
        return None
    return torch_module.tensor(
        token_groups.original_token_ids,
        device=device,
        dtype=torch_module.long,
    )


def _resolve_dtype(torch_module: Any, dtype: CompareDType) -> Any:
    if dtype == "auto":
        return "auto"
    return getattr(torch_module, dtype)


def _resolve_training_steps(
    *,
    requested_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    ind_files: tuple[Path, ...],
    ood_files: tuple[Path, ...],
    scheduler: _LanguageScheduler,
) -> tuple[int, int | None]:
    if requested_steps > 0:
        return requested_steps, None

    epochs = -requested_steps
    if epochs < 1:
        raise ValueError("--steps must not be 0")
    step_candidates: list[int] = []
    if ind_files:
        ind_examples = _count_corpus_examples_with_progress(
            ind_files,
            progress_label="Counting IND examples",
        )
        ind_examples_per_step = (
            batch_size
            * gradient_accumulation_steps
            * scheduler.language_fraction("ind")
        )
        step_candidates.append(
            _steps_needed_for_examples(
                example_count=ind_examples,
                examples_per_step=ind_examples_per_step,
                epochs=epochs,
            )
        )
    if ood_files:
        ood_examples = _count_corpus_examples_with_progress(
            ood_files,
            progress_label="Counting OOD examples",
        )
        ood_examples_per_step = (
            batch_size
            * gradient_accumulation_steps
            * scheduler.language_fraction("ood")
        )
        step_candidates.append(
            _steps_needed_for_examples(
                example_count=ood_examples,
                examples_per_step=ood_examples_per_step,
                epochs=epochs,
            )
        )
    if not step_candidates:
        raise ValueError("At least one corpus file must be provided")
    return max(step_candidates), epochs


def _steps_needed_for_examples(
    *,
    example_count: int,
    examples_per_step: float,
    epochs: int,
) -> int:
    if example_count < 1:
        raise ValueError("Corpus files do not contain any usable examples")
    if examples_per_step <= 0:
        raise ValueError("examples_per_step must be positive")
    return max(1, math.ceil((example_count * epochs) / examples_per_step))


def _count_corpus_examples(paths: tuple[Path, ...]) -> int:
    return _count_corpus_examples_with_progress(
        paths,
        progress_label="Counting examples",
    )


def _count_corpus_examples_with_progress(
    paths: tuple[Path, ...],
    *,
    progress_label: str,
) -> int:
    progress = _create_progress_bar(
        desc=progress_label,
        unit="example",
    )
    count = 0
    pending_updates = 0
    try:
        for path in paths:
            resolved_path = path.expanduser()
            if _is_jinx_path(resolved_path):
                jinx_count = _count_jinx_examples(resolved_path)
                count += jinx_count
                progress.update(jinx_count)
                continue
            for _ in _iterate_examples_in_path(resolved_path):
                count += 1
                pending_updates += 1
                if pending_updates >= 1024:
                    progress.update(pending_updates)
                    pending_updates = 0
        if pending_updates:
            progress.update(pending_updates)
        return count
    finally:
        progress.close()


def _count_jinx_examples(path: Path) -> int:
    reader = _open_jinx_reader(path)
    try:
        return len(reader)
    finally:
        _close_jinx_reader(reader)


def _resolve_resume_source(
    *,
    student_model: str | Path,
    resume_from: Path | None,
    torch_module: Any,
) -> tuple[Path, dict[str, Any] | None]:
    if resume_from is None:
        return Path(student_model).expanduser(), None
    checkpoint_path = resume_from.expanduser()
    if not checkpoint_path.exists():
        raise ValueError(f"--resume-from does not exist: {checkpoint_path}")
    state_path = checkpoint_path / _TRAINING_STATE_FILENAME
    if not state_path.exists():
        raise ValueError(
            f"--resume-from is missing {_TRAINING_STATE_FILENAME}: {checkpoint_path}. "
            "Please resume from a newer tokcleanse training checkpoint."
        )
    state = torch_module.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"{state_path} must contain a training-state dictionary")
    return checkpoint_path, state


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _resume_state_step(resume_state: dict[str, Any] | None) -> int | None:
    if resume_state is None:
        return None
    step = resume_state.get("step")
    if not isinstance(step, int) or step < 0:
        raise ValueError("training-state step must be a non-negative integer")
    return step


def _resume_state_latest_eval_loss(resume_state: dict[str, Any] | None) -> float | None:
    if resume_state is None:
        return None
    latest_eval_loss = resume_state.get("latest_eval_loss")
    if latest_eval_loss is None:
        return None
    if not isinstance(latest_eval_loss, (int, float)):
        raise ValueError("training-state latest_eval_loss must be numeric or null")
    return float(latest_eval_loss)


def _restore_resume_state(
    *,
    resume_state: dict[str, Any] | None,
    optimizer: Any,
    optimizer_weight_decays: tuple[float, ...],
    learning_rates_by_group: dict[str, float],
    lr_scheduler: _LearningRateScheduler | None,
    router_lr_scheduler: _LearningRateScheduler | None,
    ind_stream: _CorpusStream | None,
    ood_stream: _CorpusStream | None,
    scheduler: _LanguageScheduler,
    route_bias_annealer: _RouteBiasAnnealer,
) -> None:
    if resume_state is None:
        return
    optimizer_state = resume_state.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict):
        raise ValueError("training-state optimizer_state_dict is missing")
    optimizer.load_state_dict(optimizer_state)
    _move_optimizer_state_to_parameter_devices(optimizer)
    _reapply_optimizer_hyperparameters(
        optimizer=optimizer,
        optimizer_weight_decays=optimizer_weight_decays,
        learning_rates_by_group=learning_rates_by_group,
    )
    if lr_scheduler is not None:
        scheduler_state = resume_state.get("lr_scheduler_state_dict")
        if not isinstance(scheduler_state, dict):
            raise ValueError("training-state lr_scheduler_state_dict is missing")
        lr_scheduler.load_state_dict(scheduler_state)
    if router_lr_scheduler is not None:
        router_scheduler_state = resume_state.get("router_lr_scheduler_state_dict")
        if not isinstance(router_scheduler_state, dict):
            raise ValueError("training-state router_lr_scheduler_state_dict is missing")
        router_lr_scheduler.load_state_dict(router_scheduler_state)
    if ind_stream is not None:
        ind_stream_state = resume_state.get("ind_stream_state_dict")
        if not isinstance(ind_stream_state, dict):
            raise ValueError("training-state ind_stream_state_dict is missing")
        ind_stream.load_state_dict(ind_stream_state)
    if ood_stream is not None:
        ood_stream_state = resume_state.get("ood_stream_state_dict")
        if not isinstance(ood_stream_state, dict):
            raise ValueError("training-state ood_stream_state_dict is missing")
        ood_stream.load_state_dict(ood_stream_state)
    language_scheduler_state = resume_state.get("language_scheduler_state_dict")
    if not isinstance(language_scheduler_state, dict):
        raise ValueError("training-state language_scheduler_state_dict is missing")
    scheduler.load_state_dict(language_scheduler_state)
    route_bias_annealer_state = resume_state.get("route_bias_annealer_state_dict")
    if not isinstance(route_bias_annealer_state, dict):
        raise ValueError("training-state route_bias_annealer_state_dict is missing")
    route_bias_annealer.load_state_dict(route_bias_annealer_state)


def _seed_training_random_generators(*, seed: int, torch_module: Any) -> None:
    random.seed(seed)
    manual_seed = getattr(torch_module, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)
    cuda_module = getattr(torch_module, "cuda", None)
    cuda_manual_seed_all = getattr(cuda_module, "manual_seed_all", None)
    cuda_is_available = getattr(cuda_module, "is_available", None)
    if callable(cuda_manual_seed_all) and callable(cuda_is_available) and cuda_is_available():
        cuda_manual_seed_all(seed)
    mps_module = getattr(torch_module, "mps", None)
    mps_manual_seed = getattr(mps_module, "manual_seed", None)
    mps_is_available = getattr(mps_module, "is_available", None)
    if callable(mps_manual_seed) and callable(mps_is_available) and mps_is_available():
        mps_manual_seed(seed)


def _capture_training_random_generator_state(*, torch_module: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "torch_cpu_rng_state": None,
        "torch_cuda_rng_state_all": None,
        "torch_mps_rng_state": None,
    }
    get_rng_state = getattr(torch_module, "get_rng_state", None)
    if callable(get_rng_state):
        state["torch_cpu_rng_state"] = get_rng_state()
    cuda_module = getattr(torch_module, "cuda", None)
    cuda_is_available = getattr(cuda_module, "is_available", None)
    cuda_get_rng_state_all = getattr(cuda_module, "get_rng_state_all", None)
    if callable(cuda_is_available) and cuda_is_available() and callable(cuda_get_rng_state_all):
        state["torch_cuda_rng_state_all"] = cuda_get_rng_state_all()
    mps_module = getattr(torch_module, "mps", None)
    mps_is_available = getattr(mps_module, "is_available", None)
    mps_get_rng_state = getattr(mps_module, "get_rng_state", None)
    if callable(mps_is_available) and mps_is_available() and callable(mps_get_rng_state):
        state["torch_mps_rng_state"] = mps_get_rng_state()
    return state


def _restore_training_random_generator_state(
    *,
    resume_state: dict[str, Any] | None,
    torch_module: Any,
) -> None:
    if resume_state is None:
        return
    rng_state = resume_state.get("random_generator_state_dict")
    if not isinstance(rng_state, dict):
        raise ValueError("training-state random_generator_state_dict is missing")
    python_random_state = rng_state.get("python_random_state")
    if python_random_state is None:
        raise ValueError("training-state python_random_state is missing")
    random.setstate(python_random_state)
    torch_cpu_rng_state = rng_state.get("torch_cpu_rng_state")
    set_rng_state = getattr(torch_module, "set_rng_state", None)
    if torch_cpu_rng_state is not None and callable(set_rng_state):
        set_rng_state(torch_cpu_rng_state)
    torch_cuda_rng_state_all = rng_state.get("torch_cuda_rng_state_all")
    cuda_module = getattr(torch_module, "cuda", None)
    cuda_is_available = getattr(cuda_module, "is_available", None)
    cuda_set_rng_state_all = getattr(cuda_module, "set_rng_state_all", None)
    if (
        torch_cuda_rng_state_all is not None
        and callable(cuda_is_available)
        and cuda_is_available()
        and callable(cuda_set_rng_state_all)
    ):
        cuda_set_rng_state_all(torch_cuda_rng_state_all)
    torch_mps_rng_state = rng_state.get("torch_mps_rng_state")
    mps_module = getattr(torch_module, "mps", None)
    mps_is_available = getattr(mps_module, "is_available", None)
    mps_set_rng_state = getattr(mps_module, "set_rng_state", None)
    if (
        torch_mps_rng_state is not None
        and callable(mps_is_available)
        and mps_is_available()
        and callable(mps_set_rng_state)
    ):
        mps_set_rng_state(torch_mps_rng_state)


def _build_training_configuration_state(
    *,
    resolved_steps: int,
    resolved_epochs: int | None,
    teacher_model: str | Path,
    ind_files: tuple[Path, ...],
    ood_files: tuple[Path, ...],
    eval_files: tuple[Path, ...],
    resolved_device: str,
    dtype: CompareDType,
    batch_size: int,
    eval_batch_size: int,
    eval_max_batches: int,
    weight_diff_every: int,
    pad_to_max_length: bool | None,
    torch_compile: bool,
    distill_ind: bool,
    distill_ood: bool,
    distill_original_tokens_only: bool,
    distill_every: int,
    lr_warmup_steps: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    max_length: int,
    learning_rate: float,
    min_learning_rate: float,
    max_grad_norm: float,
    weight_decay: float,
    distill_kl_vocab_chunk_size: int,
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
    route_logit_bias_anneal_offset_steps: int,
    route_logit_bias_anneal_loss_threshold: float,
    router_learning_rate: float,
    router_lr_warmup_steps: int,
    router_lr_anneal_steps: int,
    router_min_learning_rate: float,
    parameter_selection: TrainingParameterSelection,
    seed: int,
) -> dict[str, Any]:
    return {
        "resolved_steps": resolved_steps,
        "resolved_epochs": resolved_epochs,
        "teacher_model": str(teacher_model),
        "ind_files": [str(path) for path in ind_files],
        "ood_files": [str(path) for path in ood_files],
        "eval_files": [str(path) for path in eval_files],
        "device": resolved_device,
        "dtype": dtype,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "eval_max_batches": eval_max_batches,
        "weight_diff_every": weight_diff_every,
        "pad_to_max_length": pad_to_max_length,
        "torch_compile": torch_compile,
        "distill_ind": distill_ind,
        "distill_ood": distill_ood,
        "distill_original_tokens_only": distill_original_tokens_only,
        "distill_every": distill_every,
        "lr_warmup_steps": lr_warmup_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "learning_rate_schedule": "linear_warmup_cosine_decay",
        "max_grad_norm": max_grad_norm,
        "weight_decay": weight_decay,
        "distill_kl_vocab_chunk_size": distill_kl_vocab_chunk_size,
        "checkpoint_every": checkpoint_every,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "eval_every_steps": eval_every_steps,
        "log_every": log_every,
        "print_every": print_every,
        "ind_batches_per_cycle": ind_batches_per_cycle,
        "ood_batches_per_cycle": ood_batches_per_cycle,
        "ind_loss_weights": {
            "lm": ind_lm_weight,
            "distill": ind_distill_weight,
            "route": ind_route_weight,
        },
        "ind_route_logit_bias": ind_route_logit_bias,
        "ood_loss_weights": {
            "lm": ood_lm_weight,
            "distill": ood_distill_weight,
            "route": ood_route_weight,
        },
        "ood_route_logit_bias": ood_route_logit_bias,
        "route_logit_bias_anneal_steps": route_logit_bias_anneal_steps,
        "route_logit_bias_anneal_offset_steps": route_logit_bias_anneal_offset_steps,
        "route_logit_bias_anneal_loss_threshold": route_logit_bias_anneal_loss_threshold,
        "router_learning_rate": router_learning_rate,
        "router_lr_warmup_steps": router_lr_warmup_steps,
        "router_lr_anneal_steps": router_lr_anneal_steps,
        "router_min_learning_rate": router_min_learning_rate,
        "parameter_selection": asdict(parameter_selection),
        "seed": seed,
    }


def _validate_resume_training_configuration(
    *,
    resume_state: dict[str, Any] | None,
    training_configuration_state: dict[str, Any],
) -> None:
    if resume_state is None:
        return
    saved_configuration = resume_state.get("training_configuration")
    if not isinstance(saved_configuration, dict):
        raise ValueError("training-state training_configuration is missing")
    if saved_configuration == training_configuration_state:
        return
    mismatched_keys = tuple(
        key
        for key in sorted(set(saved_configuration) | set(training_configuration_state))
        if saved_configuration.get(key) != training_configuration_state.get(key)
    )
    preview = ", ".join(mismatched_keys[:8])
    suffix = "" if len(mismatched_keys) <= 8 else f", ... ({len(mismatched_keys)} total)"
    raise ValueError(
        "Resume checkpoint was created with different training parameters: "
        f"{preview}{suffix}"
    )


def _reapply_optimizer_hyperparameters(
    *,
    optimizer: Any,
    optimizer_weight_decays: tuple[float, ...],
    learning_rates_by_group: dict[str, float],
) -> None:
    if len(optimizer.param_groups) != len(optimizer_weight_decays):
        raise ValueError("optimizer state is incompatible with current parameter groups")
    for group, weight_decay in zip(optimizer.param_groups, optimizer_weight_decays, strict=True):
        lr_group = group.get("lr_group")
        if not isinstance(lr_group, str) or lr_group not in learning_rates_by_group:
            raise ValueError("optimizer state is incompatible with current learning-rate groups")
        group["lr"] = learning_rates_by_group[lr_group]
        group["weight_decay"] = weight_decay


def _move_optimizer_state_to_parameter_devices(optimizer: Any) -> None:
    for parameter, state in optimizer.state.items():
        parameter_device = getattr(parameter, "device", None)
        if parameter_device is None:
            continue
        for key, value in tuple(state.items()):
            if hasattr(value, "to"):
                state[key] = value.to(device=parameter_device)


def _copy_auxiliary_tokenizer_files(*, student_model: str | Path, output_path: Path) -> None:
    student_path = Path(student_model).expanduser()
    if not student_path.exists():
        return
    for filename in _AUXILIARY_TOKENIZER_FILENAMES:
        source = student_path / filename
        if source.exists():
            shutil.copy2(source, output_path / filename)


def _save_training_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    source_model: str | Path,
    checkpoint_path: Path,
    step_label: str,
    optimizer: Any,
    lr_scheduler: _LearningRateScheduler | None,
    router_lr_scheduler: _LearningRateScheduler | None,
    ind_stream: _CorpusStream | None,
    ood_stream: _CorpusStream | None,
    scheduler: _LanguageScheduler,
    route_bias_annealer: _RouteBiasAnnealer,
    step: int,
    latest_eval_loss: float | None,
    training_configuration_state: dict[str, Any],
    torch_module: Any,
) -> None:
    tqdm.write(f"Saving checkpoint ({step_label}) to {checkpoint_path}", file=sys.stderr)
    if checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    _copy_auxiliary_tokenizer_files(student_model=source_model, output_path=checkpoint_path)
    _save_training_state(
        checkpoint_path=checkpoint_path,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        router_lr_scheduler=router_lr_scheduler,
        ind_stream=ind_stream,
        ood_stream=ood_stream,
        scheduler=scheduler,
        route_bias_annealer=route_bias_annealer,
        step=step,
        latest_eval_loss=latest_eval_loss,
        training_configuration_state=training_configuration_state,
        torch_module=torch_module,
    )


def _save_training_state(
    *,
    checkpoint_path: Path,
    optimizer: Any,
    lr_scheduler: _LearningRateScheduler | None,
    router_lr_scheduler: _LearningRateScheduler | None,
    ind_stream: _CorpusStream | None,
    ood_stream: _CorpusStream | None,
    scheduler: _LanguageScheduler,
    route_bias_annealer: _RouteBiasAnnealer,
    step: int,
    latest_eval_loss: float | None,
    training_configuration_state: dict[str, Any],
    torch_module: Any,
) -> None:
    state = {
        "version": _TRAINING_STATE_VERSION,
        "step": step,
        "latest_eval_loss": latest_eval_loss,
        "training_configuration": training_configuration_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": None if lr_scheduler is None else lr_scheduler.state_dict(),
        "router_lr_scheduler_state_dict": (
            None if router_lr_scheduler is None else router_lr_scheduler.state_dict()
        ),
        "ind_stream_state_dict": None if ind_stream is None else ind_stream.state_dict(),
        "ood_stream_state_dict": None if ood_stream is None else ood_stream.state_dict(),
        "language_scheduler_state_dict": scheduler.state_dict(),
        "route_bias_annealer_state_dict": route_bias_annealer.state_dict(),
        "random_generator_state_dict": _capture_training_random_generator_state(
            torch_module=torch_module,
        ),
    }
    torch_module.save(state, checkpoint_path / _TRAINING_STATE_FILENAME)


def _write_training_recipe(
    *,
    output_path: Path,
    student_model: str | Path,
    teacher_model: str | Path,
    resume_from: Path | None,
    requested_steps: int,
    resolved_steps: int,
    resolved_epochs: int | None,
    ind_files: tuple[Path, ...],
    ood_files: tuple[Path, ...],
    eval_files: tuple[Path, ...],
    resolved_device: str,
    dtype: CompareDType,
    steps: int,
    batch_size: int,
    eval_batch_size: int,
    eval_max_batches: int,
    weight_diff_every: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    torch_compile: bool,
    lr_warmup_steps: int,
    distill_ind: bool,
    distill_ood: bool,
    distill_original_tokens_only: bool,
    distill_every: int,
    max_length: int,
    learning_rate: float,
    min_learning_rate: float,
    max_grad_norm: float,
    weight_decay: float,
    ind_route_weight: float,
    ind_route_logit_bias: float,
    ood_route_weight: float,
    ood_route_logit_bias: float,
    route_logit_bias_anneal_steps: int,
    route_logit_bias_anneal_offset_steps: int,
    route_logit_bias_anneal_loss_threshold: float,
    router_learning_rate: float,
    router_lr_warmup_steps: int,
    router_lr_anneal_steps: int,
    router_min_learning_rate: float,
    parameter_selection: TrainingParameterSelection,
    distill_kl_vocab_chunk_size: int,
    checkpoint_every: int,
    checkpoint_dir: Path | None,
    eval_every_steps: int,
    log_every: int,
    print_every: int,
    pad_to_max_length: bool | None,
    metrics_log_path: Path,
    ind_batches_per_cycle: int,
    ood_batches_per_cycle: int,
    ind_weights: TrainingLossWeights,
    ood_weights: TrainingLossWeights,
    seed: int,
    token_groups: TokenGroupMetadata,
) -> None:
    recipe = {
        "student_model": str(student_model),
        "teacher_model": str(teacher_model),
        "resume_from": None if resume_from is None else str(resume_from),
        "requested_steps": requested_steps,
        "resolved_steps": resolved_steps,
        "resolved_epochs": resolved_epochs,
        "ind_files": [str(path) for path in ind_files],
        "ood_files": [str(path) for path in ood_files],
        "eval_files": [str(path) for path in eval_files],
        "device": resolved_device,
        "dtype": dtype,
        "steps": steps,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "eval_max_batches": eval_max_batches,
        "weight_diff_every": weight_diff_every,
        "gradient_accumulation": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "torch_compile": torch_compile,
        "lr_warmup_steps": lr_warmup_steps,
        "distill_ind": distill_ind,
        "distill_ood": distill_ood,
        "distill_original_tokens_only": distill_original_tokens_only,
        "distill_every": distill_every,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "learning_rate_schedule": "linear_warmup_cosine_decay",
        "max_grad_norm": max_grad_norm,
        "weight_decay": weight_decay,
        "ind_route_weight": ind_route_weight,
        "ind_route_logit_bias": ind_route_logit_bias,
        "ood_route_weight": ood_route_weight,
        "ood_route_logit_bias": ood_route_logit_bias,
        "route_logit_bias_anneal_steps": route_logit_bias_anneal_steps,
        "route_logit_bias_anneal_offset_steps": route_logit_bias_anneal_offset_steps,
        "route_logit_bias_anneal_loss_threshold": route_logit_bias_anneal_loss_threshold,
        "router_learning_rate": router_learning_rate,
        "router_lr_warmup_steps": router_lr_warmup_steps,
        "router_lr_anneal_steps": router_lr_anneal_steps,
        "router_min_learning_rate": router_min_learning_rate,
        "parameter_selection": asdict(parameter_selection),
        "distill_kl_vocab_chunk_size": distill_kl_vocab_chunk_size,
        "checkpoint_every": checkpoint_every,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "eval_every": eval_every_steps,
        "log_every": log_every,
        "print_every": print_every,
        "pad_to_max_length": pad_to_max_length,
        "metrics_log_path": str(metrics_log_path),
        "ind_batches_per_cycle": ind_batches_per_cycle,
        "ood_batches_per_cycle": ood_batches_per_cycle,
        "ind_loss_weights": asdict(ind_weights),
        "ood_loss_weights": asdict(ood_weights),
        "seed": seed,
        "token_groups": asdict(token_groups),
    }
    (output_path / "training_recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _release_resources(*, torch_module: Any) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    mps_module = getattr(torch_module, "mps", None)
    is_available = getattr(mps_module, "is_available", None)
    empty_cache = getattr(mps_module, "empty_cache", None)
    if callable(is_available) and is_available() and callable(empty_cache):
        empty_cache()


def _release_mps_working_set(*, torch_module: Any, device: str) -> None:
    if device != "mps":
        return
    mps_module = getattr(torch_module, "mps", None)
    if mps_module is None:
        return
    synchronize = getattr(mps_module, "synchronize", None)
    if callable(synchronize):
        synchronize()
    empty_cache = getattr(mps_module, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _capture_mps_memory_snapshot(*, torch_module: Any, device: str) -> _MpsMemorySnapshot | None:
    if device != "mps":
        return None
    mps_module = getattr(torch_module, "mps", None)
    if mps_module is None:
        return None
    current_allocated_memory = getattr(mps_module, "current_allocated_memory", None)
    driver_allocated_memory = getattr(mps_module, "driver_allocated_memory", None)
    if not callable(current_allocated_memory) or not callable(driver_allocated_memory):
        return None
    recommended_max_memory = getattr(mps_module, "recommended_max_memory", None)
    return _MpsMemorySnapshot(
        allocated_bytes=int(current_allocated_memory()),
        driver_allocated_bytes=int(driver_allocated_memory()),
        recommended_max_bytes=int(recommended_max_memory()) if callable(recommended_max_memory) else None,
    )


def _serialize_mps_memory_snapshot(snapshot: _MpsMemorySnapshot) -> dict[str, float | int]:
    record: dict[str, float | int] = {
        "mps_allocated_bytes": snapshot.allocated_bytes,
        "mps_allocated_gib": _bytes_to_gib(snapshot.allocated_bytes),
        "mps_driver_allocated_bytes": snapshot.driver_allocated_bytes,
        "mps_driver_allocated_gib": _bytes_to_gib(snapshot.driver_allocated_bytes),
    }
    if snapshot.recommended_max_bytes is not None:
        record["mps_recommended_max_bytes"] = snapshot.recommended_max_bytes
        record["mps_recommended_max_gib"] = _bytes_to_gib(snapshot.recommended_max_bytes)
        record["mps_headroom_gib"] = _bytes_to_gib(
            max(0, snapshot.recommended_max_bytes - snapshot.driver_allocated_bytes)
        )
    return record


def _bytes_to_gib(value: int) -> float:
    return value / float(1024 ** 3)


def _format_gib(value: float) -> str:
    return f"{value:.2f}GiB"


def _format_mps_progress_value(snapshot: _MpsMemorySnapshot) -> str:
    driver_gib = _bytes_to_gib(snapshot.driver_allocated_bytes)
    if snapshot.recommended_max_bytes is None:
        return _format_gib(driver_gib)
    return f"{driver_gib:.1f}/{_bytes_to_gib(snapshot.recommended_max_bytes):.1f}GiB"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _average_language_metric(metrics: dict[str, float | int], key: str) -> float | None:
    batch_count = int(metrics["batches"])
    if batch_count < 1:
        return None
    return float(metrics[key]) / batch_count


def _compute_expert_pair_weight_diff_metrics(*, model: Any, torch_module: Any) -> dict[str, float]:
    router_stats = _accumulate_expert_pair_parameter_diffs(
        model=model,
        torch_module=torch_module,
        include_parameter=_is_router_expert_pair_parameter,
    )
    expert_stats = _accumulate_expert_pair_parameter_diffs(
        model=model,
        torch_module=torch_module,
        include_parameter=_is_moe_expert_pair_parameter,
    )
    metrics: dict[str, float] = {}
    if router_stats is not None:
        metrics.update(
            {
                "router_expert_weight_diff_mean_abs": router_stats["mean_abs"],
                "router_expert_weight_diff_mean_squared": router_stats["mean_squared"],
            }
        )
    if expert_stats is not None:
        metrics.update(
            {
                "expert_weight_diff_mean_abs": expert_stats["mean_abs"],
                "expert_weight_diff_mean_squared": expert_stats["mean_squared"],
            }
        )
    return metrics


def _accumulate_expert_pair_parameter_diffs(
    *,
    model: Any,
    torch_module: Any,
    include_parameter: Any,
) -> dict[str, float] | None:
    abs_sum = 0.0
    squared_sum = 0.0
    element_count = 0
    with torch_module.no_grad():
        for name, parameter in model.named_parameters():
            if not include_parameter(name, parameter):
                continue
            diff = parameter.detach()[0] - parameter.detach()[1]
            diff = diff.float()
            abs_sum += float(diff.abs().sum().cpu())
            squared_sum += float(diff.square().sum().cpu())
            element_count += int(diff.numel())
    if element_count < 1:
        return None
    return {
        "mean_abs": abs_sum / element_count,
        "mean_squared": squared_sum / element_count,
    }


def _is_router_expert_pair_parameter(name: str, parameter: Any) -> bool:
    return ".router." in name and _has_expert_pair_axis(parameter)


def _is_moe_expert_pair_parameter(name: str, parameter: Any) -> bool:
    return (
        (".experts.gate_up_proj" in name or ".experts.down_proj" in name)
        and _has_expert_pair_axis(parameter)
    )


def _has_expert_pair_axis(parameter: Any) -> bool:
    shape = getattr(parameter, "shape", ())
    return len(shape) >= 1 and int(shape[0]) == 2


def _format_optional_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _format_optional_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}s"


def _print_training_metrics(record: dict[str, Any]) -> None:
    reported_objective = record.get("train_reported_objective", record["train_loss"])
    parts = [
        f"step={record['step']}",
        f"mix={record['mix']}",
        f"train_objective={reported_objective:.4f}",
        f"train_lm_loss={record['train_lm_loss']:.4f}",
        f"train_distill_loss={record['train_distill_loss']:.4f}",
        f"ind_train_loss={_format_optional_metric(record.get('ind_train_loss'))}",
        f"ood_train_loss={_format_optional_metric(record.get('ood_train_loss'))}",
    ]
    task_loss = record.get("train_task_loss")
    if isinstance(task_loss, float):
        parts.insert(3, f"train_task_loss={task_loss:.4f}")
    route_loss = record.get("train_route_loss")
    if isinstance(route_loss, float):
        parts.insert(5, f"train_route_loss={route_loss:.4f}")
    router_objective = record.get("train_router_objective")
    if isinstance(router_objective, float):
        parts.insert(6, f"train_router_objective={router_objective:.4f}")
    learning_rate = record.get("learning_rate")
    if isinstance(learning_rate, float):
        parts.insert(7, f"main_lr={learning_rate:.2e}")
    router_learning_rate = record.get("router_learning_rate")
    if isinstance(router_learning_rate, float):
        parts.insert(8, f"router_lr={router_learning_rate:.2e}")
    main_grad_norm_pre_clip = record.get("main_grad_norm_pre_clip", record.get("main_grad_norm"))
    if isinstance(main_grad_norm_pre_clip, float):
        parts.append(f"main_grad_pre={main_grad_norm_pre_clip:.4e}")
    main_grad_norm_post_clip = record.get("main_grad_norm_post_clip")
    if isinstance(main_grad_norm_post_clip, float):
        parts.append(f"main_grad_clip={main_grad_norm_post_clip:.4e}")
    router_grad_norm_pre_clip = record.get("router_grad_norm_pre_clip", record.get("router_grad_norm"))
    if isinstance(router_grad_norm_pre_clip, float):
        parts.append(f"router_grad_pre={router_grad_norm_pre_clip:.4e}")
    router_grad_norm_post_clip = record.get("router_grad_norm_post_clip")
    if isinstance(router_grad_norm_post_clip, float):
        parts.append(f"router_grad_clip={router_grad_norm_post_clip:.4e}")
    step_total_s = record.get("step_total_s")
    if isinstance(step_total_s, float):
        parts.append(f"step_s={_format_optional_seconds(step_total_s)}")
    for key, label in (
        ("data_s", "data_s"),
        ("tokenize_s", "tok_s"),
        ("move_s", "move_s"),
        ("student_forward_s", "stu_s"),
        ("teacher_forward_s", "tea_s"),
        ("lm_loss_s", "lm_s"),
        ("distill_loss_s", "kl_s"),
        ("backward_s", "back_s"),
        ("optimizer_s", "opt_s"),
        ("eval_s", "eval_s"),
    ):
        value = record.get(key)
        if isinstance(value, float):
            parts.append(f"{label}={_format_optional_seconds(value)}")
    eval_lm_loss = record.get("eval_lm_loss")
    latest_eval_lm_loss = record.get("latest_eval_lm_loss")
    if isinstance(eval_lm_loss, float):
        parts.append(f"eval_lm_loss={eval_lm_loss:.4f}")
    elif isinstance(latest_eval_lm_loss, float):
        parts.append(f"latest_eval_lm_loss={latest_eval_lm_loss:.4f}")
    for key, label in (
        ("ind_router_weight_expert1", "ind_route_e1"),
        ("ood_router_weight_expert1", "ood_route_e1"),
        ("ind_router_prob_expert1", "ind_prob_e1"),
        ("ood_router_prob_expert1", "ood_prob_e1"),
        ("route_logit_bias_scale", "route_bias_scale"),
        ("ind_route_logit_bias", "ind_route_bias"),
        ("ood_route_logit_bias", "ood_route_bias"),
    ):
        value = record.get(key)
        if isinstance(value, float):
            parts.append(f"{label}={value:.6f}")
    for key, label in (
        ("router_expert_weight_diff_mean_abs", "router_wdiff_abs"),
        ("router_expert_weight_diff_mean_squared", "router_wdiff_msq"),
        ("expert_weight_diff_mean_abs", "expert_wdiff_abs"),
        ("expert_weight_diff_mean_squared", "expert_wdiff_msq"),
    ):
        value = record.get(key)
        if isinstance(value, float):
            parts.append(f"{label}={value:.6e}")
    mps_allocated_gib = record.get("mps_allocated_gib")
    mps_driver_allocated_gib = record.get("mps_driver_allocated_gib")
    mps_headroom_gib = record.get("mps_headroom_gib")
    if isinstance(mps_allocated_gib, float):
        parts.append(f"mps_allocated={_format_gib(mps_allocated_gib)}")
    if isinstance(mps_driver_allocated_gib, float):
        parts.append(f"mps_driver={_format_gib(mps_driver_allocated_gib)}")
    if isinstance(mps_headroom_gib, float):
        parts.append(f"mps_headroom={_format_gib(mps_headroom_gib)}")
    print(" ".join(parts), file=sys.stdout, flush=True)


def _evaluate_language_model_loss(
    *,
    model: Any,
    tokenizer: Any,
    eval_files: tuple[Path, ...],
    batch_size: int,
    max_batches: int,
    max_length: int,
    device: str,
    pad_to_max_length: bool | None,
    torch_module: Any,
) -> float:
    was_training = bool(model.training)
    model.eval()
    loss_sum = 0.0
    token_count = 0.0
    batch_count = 0
    batch: list[str] = []
    progress = _create_progress_bar(
        desc="Evaluating",
        unit="batch",
    )
    try:
        for example in _iterate_examples_once(eval_files):
            batch.append(example)
            if len(batch) < batch_size:
                continue
            if max_batches > 0 and batch_count >= max_batches:
                break
            batch_loss_sum, batch_token_count = _evaluate_batch_language_model_loss(
                model=model,
                tokenizer=tokenizer,
                batch_texts=tuple(batch),
                max_length=max_length,
                device=device,
                pad_to_max_length=pad_to_max_length,
                torch_module=torch_module,
            )
            loss_sum += batch_loss_sum
            token_count += batch_token_count
            batch.clear()
            batch_count += 1
            progress.update(1)
            if max_batches > 0 and batch_count >= max_batches:
                break
        if batch and (max_batches == 0 or batch_count < max_batches):
            batch_loss_sum, batch_token_count = _evaluate_batch_language_model_loss(
                model=model,
                tokenizer=tokenizer,
                batch_texts=tuple(batch),
                max_length=max_length,
                device=device,
                pad_to_max_length=pad_to_max_length,
                torch_module=torch_module,
            )
            loss_sum += batch_loss_sum
            token_count += batch_token_count
            batch_count += 1
            progress.update(1)
        if token_count == 0:
            raise ValueError("Evaluation files do not contain any usable examples")
        return loss_sum / token_count
    finally:
        progress.close()
        if was_training:
            model.train()


def _create_progress_bar(
    *,
    desc: str,
    unit: str,
) -> tqdm[Any]:
    return tqdm(
        desc=desc,
        total=None,
        unit=unit,
        dynamic_ncols=True,
        leave=False,
        mininterval=0.5,
        file=sys.stderr,
        disable=not sys.stderr.isatty(),
    )


def _evaluate_batch_language_model_loss(
    *,
    model: Any,
    tokenizer: Any,
    batch_texts: tuple[str, ...],
    max_length: int,
    device: str,
    pad_to_max_length: bool | None,
    torch_module: Any,
) -> tuple[float, float]:
    encoded = _tokenize_batch(
        tokenizer=tokenizer,
        batch_texts=batch_texts,
        max_length=max_length,
        device=device,
        pad_to_max_length=pad_to_max_length,
    )
    encoded = _move_to_device(encoded, device)
    outputs = None
    logits = None
    loss_sum = None
    token_count = None
    try:
        with torch_module.inference_mode():
            outputs = model(**encoded, use_cache=False)
        logits = outputs.logits
        loss_sum, token_count = _compute_language_model_loss_components(
            logits=logits,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            torch_module=torch_module,
        )
        return float(loss_sum.detach().cpu()), float(token_count.detach().cpu())
    finally:
        token_count = None
        loss_sum = None
        logits = None
        outputs = None
        encoded = None
        _release_mps_working_set(
            torch_module=torch_module,
            device=device,
        )


__all__ = [
    "DEFAULT_LOG_EVERY",
    "DEFAULT_DISTILL_KL_VOCAB_CHUNK_SIZE",
    "DEFAULT_PRINT_EVERY",
    "DEFAULT_ROUTER_LEARNING_RATE",
    "DEFAULT_ROUTER_LR_ANNEAL_STEPS",
    "DEFAULT_ROUTER_LR_WARMUP_STEPS",
    "DEFAULT_ROUTER_MIN_LEARNING_RATE",
    "DEFAULT_TRAIN_BATCH_SIZE",
    "DEFAULT_TRAIN_DTYPE",
    "DEFAULT_TRAIN_EVAL_EVERY",
    "DEFAULT_TRAIN_EVAL_MAX_BATCHES",
    "DEFAULT_TRAIN_GRADIENT_ACCUMULATION_STEPS",
    "DEFAULT_TRAIN_LEARNING_RATE",
    "DEFAULT_TRAIN_LR_WARMUP_STEPS",
    "DEFAULT_TRAIN_MAX_GRAD_NORM",
    "DEFAULT_TRAIN_MIN_LEARNING_RATE",
    "DEFAULT_TRAIN_STEPS",
    "DEFAULT_TRAIN_TORCH_COMPILE",
    "DEFAULT_WEIGHT_DIFF_EVERY",
    "TokenGroupMetadata",
    "TrainingDryRunReport",
    "TrainingLossWeights",
    "TrainingParameterSelection",
    "build_training_dry_run_report",
    "load_token_group_metadata",
    "prepare_student_for_distillation",
    "train_distilled_model",
]
