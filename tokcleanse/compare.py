"""Model comparison helpers for prompt-by-prompt generation checks."""

from __future__ import annotations

import copy
import gc
import importlib
import json
import logging
import unicodedata
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from ._moe import _load_causal_lm_for_runtime, _load_tokenizer_for_runtime

DEFAULT_COMPARE_PROMPT = "Who are you?"
DEFAULT_GENERATION_BATCH_SIZE = 8
DEFAULT_SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEMANTIC_THRESHOLD = 0.92
DEFAULT_JUDGE_MAX_NEW_TOKENS = 8

CompareDevice = Literal["auto", "cpu", "cuda", "mps"]
CompareDType = Literal["auto", "float32", "float16", "bfloat16"]


@dataclass(frozen=True, slots=True)
class PromptComparison:
    """Comparison result for one prompt."""

    prompt: str
    answer_a: str
    answer_b: str
    match: bool
    normalized_match: bool
    semantic_similarity: float
    semantic_match: bool
    judge_match: bool | None
    close_match: bool


@dataclass(frozen=True, slots=True)
class ComparisonSummary:
    """Aggregate statistics across prompt comparisons."""

    total: int
    matches: int
    normalized_matches: int
    semantic_matches: int
    close_matches: int
    judge_matches: int
    judged_prompts: int
    mean_semantic_similarity: float


@dataclass(frozen=True, slots=True)
class PromptLogitDiff:
    """Logit-difference result for one prompt."""

    prompt: str
    token_count: int
    max_abs_diff: float
    mean_abs_diff: float
    rmse: float
    allclose: bool


@dataclass(frozen=True, slots=True)
class LogitDiffSummary:
    """Aggregate statistics across prompt logit comparisons."""

    total: int
    allclose_prompts: int
    global_max_abs_diff: float
    mean_max_abs_diff: float
    mean_mean_abs_diff: float
    mean_rmse: float


def load_prompts(
    *,
    prompt_texts: list[str] | tuple[str, ...],
    prompt_file: str | Path | None,
) -> tuple[str, ...]:
    """Load prompts from repeated CLI flags and/or a JSONL prompt file."""

    if prompt_file is not None:
        return tuple(_load_prompts_from_jsonl(Path(prompt_file)))

    prompts = [text for text in prompt_texts if text]
    if not prompts:
        return (DEFAULT_COMPARE_PROMPT,)
    return tuple(prompts)


def compare_model_logits(
    model_a: str | Path,
    model_b: str | Path,
    *,
    prompts: list[str] | tuple[str, ...],
    completion: bool = False,
    batch_size: int = DEFAULT_GENERATION_BATCH_SIZE,
    device: CompareDevice = "auto",
    dtype: CompareDType = "auto",
    max_length: int = 1024,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> tuple[PromptLogitDiff, ...]:
    """Compare prompt-conditioned logits between two aligned models."""

    import torch

    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if max_length < 1:
        raise ValueError("--max-length must be at least 1")
    if atol < 0:
        raise ValueError("--atol must be non-negative")
    if rtol < 0:
        raise ValueError("--rtol must be non-negative")

    resolved_device = _resolve_device(torch, device)
    resolved_dtype = _resolve_dtype(torch, dtype)
    prompts_tuple = tuple(prompts)

    tokenizer_a = None
    tokenizer_b = None
    model_a_loaded = None
    model_b_loaded = None
    try:
        tokenizer_a = _load_tokenizer_for_runtime(str(model_a))
        tokenizer_b = _load_tokenizer_for_runtime(str(model_b))
        _prepare_generation_tokenizer(tokenizer_a)
        _prepare_generation_tokenizer(tokenizer_b)
        model_a_loaded = _load_causal_lm_for_runtime(
            model_path=str(model_a),
            dtype=resolved_dtype,
            device=resolved_device,
        )
        model_b_loaded = _load_causal_lm_for_runtime(
            model_path=str(model_b),
            dtype=resolved_dtype,
            device=resolved_device,
        )
        model_a_loaded.eval()
        model_b_loaded.eval()
        model_a_loaded.to(resolved_device)
        model_b_loaded.to(resolved_device)

        results: list[PromptLogitDiff] = []
        num_batches = (len(prompts_tuple) + batch_size - 1) // batch_size
        for start in tqdm(
            range(0, len(prompts_tuple), batch_size),
            total=num_batches,
            desc=f"Comparing logits {_display_name(model_a)} vs {_display_name(model_b)}",
            unit="batch",
        ):
            batch_prompts = prompts_tuple[start : start + batch_size]
            encoded_cpu = _encode_shared_prompt_batch(
                tokenizer_a=tokenizer_a,
                tokenizer_b=tokenizer_b,
                prompts=batch_prompts,
                completion=completion,
                max_length=max_length,
            )
            encoded = _move_to_device(encoded_cpu, resolved_device)
            with torch.inference_mode():
                logits_a = model_a_loaded(**encoded).logits.float().cpu()
                logits_b = model_b_loaded(**encoded).logits.float().cpu()
            if logits_a.shape != logits_b.shape:
                raise ValueError(
                    f"Logit shapes differ between models: {tuple(logits_a.shape)} vs {tuple(logits_b.shape)}"
                )
            attention_mask = encoded_cpu["attention_mask"].cpu()
            results.extend(
                _summarize_batch_logit_diffs(
                    prompts=batch_prompts,
                    logits_a=logits_a,
                    logits_b=logits_b,
                    attention_mask=attention_mask,
                    atol=atol,
                    rtol=rtol,
                    torch_module=torch,
                )
            )
        return tuple(results)
    finally:
        if model_a_loaded is not None:
            del model_a_loaded
        if model_b_loaded is not None:
            del model_b_loaded
        if tokenizer_a is not None:
            del tokenizer_a
        if tokenizer_b is not None:
            del tokenizer_b
        _release_resources(torch_module=torch)


def compare_models(
    model_a: str | Path,
    model_b: str | Path,
    *,
    prompts: list[str] | tuple[str, ...],
    completion: bool = False,
    max_new_tokens: int = 128,
    batch_size: int = DEFAULT_GENERATION_BATCH_SIZE,
    judge_batch_size: int | None = None,
    device: CompareDevice = "auto",
    dtype: CompareDType = "auto",
    semantic_model_name: str | Path = DEFAULT_SEMANTIC_MODEL_NAME,
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    llm_judge: str | Path | None = None,
    judge_all: bool = False,
) -> tuple[PromptComparison, ...]:
    """Run prompts through two models and compare their answers."""

    import torch

    if judge_all and llm_judge is None:
        raise ValueError("--judge-all requires --llm-judge")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if judge_batch_size is not None and judge_batch_size < 1:
        raise ValueError("--judge-batch-size must be at least 1")

    resolved_device = _resolve_device(torch, device)
    resolved_dtype = _resolve_dtype(torch, dtype)
    resolved_judge_batch_size = batch_size if judge_batch_size is None else judge_batch_size
    prompts_tuple = tuple(prompts)
    answers_a = _generate_answers_for_model(
        model_path=model_a,
        prompts=prompts_tuple,
        completion=completion,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=resolved_device,
        dtype=resolved_dtype,
        torch_module=torch,
    )
    answers_b = _generate_answers_for_model(
        model_path=model_b,
        prompts=prompts_tuple,
        completion=completion,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        device=resolved_device,
        dtype=resolved_dtype,
        torch_module=torch,
    )
    semantic_scores = _score_answers_semantically(
        answers_a=answers_a,
        answers_b=answers_b,
        semantic_model_name=semantic_model_name,
        device=resolved_device,
        dtype=resolved_dtype,
        torch_module=torch,
    )

    judge = None
    if llm_judge is not None:
        judge = _LlmJudge(
            judge_model=llm_judge,
            device=resolved_device,
            dtype=resolved_dtype,
            torch_module=torch,
        )

    try:
        normalized_matches = tuple(
            _normalize_text(answer_a) == _normalize_text(answer_b)
            for answer_a, answer_b in zip(answers_a, answers_b, strict=True)
        )
        semantic_matches = tuple(
            semantic_similarity >= semantic_threshold for semantic_similarity in semantic_scores
        )
        judge_matches = [None] * len(prompts_tuple)
        indices_to_judge = [
            index
            for index, (answer_a, normalized_match, semantic_match) in enumerate(
                zip(answers_a, normalized_matches, semantic_matches, strict=True)
            )
            if judge is not None
            and (
                judge_all
                or (
                    answer_a != answers_b[index]
                    and not normalized_match
                    and not semantic_match
                )
            )
        ]
        if judge is not None and indices_to_judge:
            judged_values = judge.judge_batch(
                prompts=tuple(prompts_tuple[index] for index in indices_to_judge),
                answers_a=tuple(answers_a[index] for index in indices_to_judge),
                answers_b=tuple(answers_b[index] for index in indices_to_judge),
                batch_size=resolved_judge_batch_size,
            )
            for index, judged_value in zip(indices_to_judge, judged_values, strict=True):
                judge_matches[index] = judged_value

        results = []
        for index, (prompt, answer_a, answer_b, semantic_similarity) in enumerate(
            tqdm(
                zip(prompts_tuple, answers_a, answers_b, semantic_scores, strict=True),
                total=len(prompts_tuple),
                desc="Scoring outputs",
                unit="prompt",
            )
        ):
            normalized_match = normalized_matches[index]
            semantic_match = semantic_matches[index]
            judge_match = judge_matches[index]
            close_match = answer_a == answer_b or normalized_match or semantic_match or judge_match is True
            results.append(
                PromptComparison(
                    prompt=prompt,
                    answer_a=answer_a,
                    answer_b=answer_b,
                    match=answer_a == answer_b,
                    normalized_match=normalized_match,
                    semantic_similarity=semantic_similarity,
                    semantic_match=semantic_match,
                    judge_match=judge_match,
                    close_match=close_match,
                )
            )
        return tuple(results)
    finally:
        if judge is not None:
            judge.close()


def summarize_comparisons(
    comparisons: list[PromptComparison] | tuple[PromptComparison, ...],
) -> ComparisonSummary:
    """Build aggregate comparison statistics."""

    total = len(comparisons)
    mean_semantic_similarity = 0.0
    if total:
        mean_semantic_similarity = sum(
            comparison.semantic_similarity for comparison in comparisons
        ) / total
    return ComparisonSummary(
        total=total,
        matches=sum(comparison.match for comparison in comparisons),
        normalized_matches=sum(comparison.normalized_match for comparison in comparisons),
        semantic_matches=sum(comparison.semantic_match for comparison in comparisons),
        close_matches=sum(comparison.close_match for comparison in comparisons),
        judge_matches=sum(comparison.judge_match is True for comparison in comparisons),
        judged_prompts=sum(comparison.judge_match is not None for comparison in comparisons),
        mean_semantic_similarity=mean_semantic_similarity,
    )


def serialize_comparisons(
    comparisons: list[PromptComparison] | tuple[PromptComparison, ...],
) -> tuple[str, ...]:
    """Serialize prompt comparisons and a summary to JSONL-style strings."""

    return serialize_prompt_comparisons(comparisons) + (serialize_summary(comparisons),)


def serialize_prompt_comparisons(
    comparisons: list[PromptComparison] | tuple[PromptComparison, ...],
) -> tuple[str, ...]:
    """Serialize prompt comparisons to JSONL-style strings."""

    return tuple(json.dumps(asdict(comparison), ensure_ascii=False) for comparison in comparisons)


def serialize_summary(
    comparisons: list[PromptComparison] | tuple[PromptComparison, ...],
) -> str:
    """Serialize aggregate comparison statistics to one JSON string."""

    return json.dumps({"summary": asdict(summarize_comparisons(comparisons))}, ensure_ascii=False)


def summarize_logit_diffs(
    diffs: list[PromptLogitDiff] | tuple[PromptLogitDiff, ...],
) -> LogitDiffSummary:
    """Build aggregate prompt-logit comparison statistics."""

    total = len(diffs)
    if total == 0:
        return LogitDiffSummary(
            total=0,
            allclose_prompts=0,
            global_max_abs_diff=0.0,
            mean_max_abs_diff=0.0,
            mean_mean_abs_diff=0.0,
            mean_rmse=0.0,
        )
    return LogitDiffSummary(
        total=total,
        allclose_prompts=sum(diff.allclose for diff in diffs),
        global_max_abs_diff=max(diff.max_abs_diff for diff in diffs),
        mean_max_abs_diff=sum(diff.max_abs_diff for diff in diffs) / total,
        mean_mean_abs_diff=sum(diff.mean_abs_diff for diff in diffs) / total,
        mean_rmse=sum(diff.rmse for diff in diffs) / total,
    )


def serialize_logit_diffs(
    diffs: list[PromptLogitDiff] | tuple[PromptLogitDiff, ...],
) -> tuple[str, ...]:
    """Serialize prompt logit diffs and a summary to JSONL-style strings."""

    return serialize_prompt_logit_diffs(diffs) + (serialize_logit_diff_summary(diffs),)


def serialize_prompt_logit_diffs(
    diffs: list[PromptLogitDiff] | tuple[PromptLogitDiff, ...],
) -> tuple[str, ...]:
    """Serialize prompt logit diffs to JSONL-style strings."""

    return tuple(json.dumps(asdict(diff), ensure_ascii=False) for diff in diffs)


def serialize_logit_diff_summary(
    diffs: list[PromptLogitDiff] | tuple[PromptLogitDiff, ...],
) -> str:
    """Serialize aggregate prompt-logit statistics to one JSON string."""

    return json.dumps({"summary": asdict(summarize_logit_diffs(diffs))}, ensure_ascii=False)


def comparison_has_any_mismatch(comparison: PromptComparison) -> bool:
    """Return whether any comparison signal indicates a mismatch."""

    return (
        not comparison.match
        or not comparison.normalized_match
        or not comparison.semantic_match
        or comparison.judge_match is False
    )


def _generate_answers_for_model(
    *,
    model_path: str | Path,
    prompts: tuple[str, ...],
    completion: bool,
    max_new_tokens: int,
    batch_size: int,
    device: str,
    dtype: Any,
    torch_module: Any,
) -> tuple[str, ...]:
    from transformers import AutoTokenizer

    tokenizer = None
    model = None
    try:
        tokenizer = _load_tokenizer_for_runtime(str(model_path))
        _prepare_generation_tokenizer(tokenizer)
        model = _load_causal_lm_for_runtime(
            model_path=str(model_path),
            dtype=dtype,
            device=device,
        )
        model.eval()
        model.to(device)
        answers = []
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        for start in tqdm(
            range(0, len(prompts), batch_size),
            total=num_batches,
            desc=f"Generating {_display_name(model_path)}",
            unit="batch",
        ):
            batch_prompts = prompts[start : start + batch_size]
            answers.extend(
                _generate_answer_batch(
                    tokenizer=tokenizer,
                    model=model,
                    prompts=batch_prompts,
                    completion=completion,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    torch_module=torch_module,
                )
            )
        return tuple(answers)
    finally:
        if model is not None:
            del model
            model = None
        if tokenizer is not None:
            del tokenizer
            tokenizer = None
        _release_resources(torch_module=torch_module)


def _score_answers_semantically(
    *,
    answers_a: tuple[str, ...],
    answers_b: tuple[str, ...],
    semantic_model_name: str | Path,
    device: str,
    dtype: Any,
    torch_module: Any,
) -> tuple[float, ...]:
    scorer = _SemanticScorer(
        semantic_model_name=semantic_model_name,
        device=device,
        dtype=dtype,
        torch_module=torch_module,
    )
    try:
        return scorer.score_pairs(answers_a=answers_a, answers_b=answers_b)
    finally:
        scorer.close()


def _load_prompts_from_jsonl(path: Path) -> list[str]:
    prompts: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            raise ValueError(f"Expected JSON object with string text on line {line_number} of {path}")
        prompts.append(data["text"])
    return prompts


def _encode_shared_prompt_batch(
    *,
    tokenizer_a: Any,
    tokenizer_b: Any,
    prompts: tuple[str, ...],
    completion: bool,
    max_length: int,
) -> dict[str, Any]:
    encoded_a = _encode_prompt_batch(
        tokenizer=tokenizer_a,
        prompts=prompts,
        completion=completion,
        max_length=max_length,
    )
    encoded_b = _encode_prompt_batch(
        tokenizer=tokenizer_b,
        prompts=prompts,
        completion=completion,
        max_length=max_length,
    )
    required_keys = ("input_ids", "attention_mask")
    for key in required_keys:
        if key not in encoded_a or key not in encoded_b:
            raise ValueError(f"Missing {key} while encoding prompts for logit diff")
        tensor_a = encoded_a[key]
        tensor_b = encoded_b[key]
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(
                f"Logit diff requires aligned tokenization, but {key} shapes differ: "
                f"{tuple(tensor_a.shape)} vs {tuple(tensor_b.shape)}"
            )
        if not tensor_a.equal(tensor_b):
            raise ValueError(
                "Logit diff requires aligned tokenization, but encoded prompt ids differ between models"
            )
    return encoded_a


def _summarize_batch_logit_diffs(
    *,
    prompts: tuple[str, ...],
    logits_a: Any,
    logits_b: Any,
    attention_mask: Any,
    atol: float,
    rtol: float,
    torch_module: Any,
) -> tuple[PromptLogitDiff, ...]:
    results: list[PromptLogitDiff] = []
    for index, prompt in enumerate(prompts):
        valid_positions = attention_mask[index].bool()
        token_count = int(valid_positions.sum().item())
        if token_count == 0:
            raise ValueError("Encountered an empty encoded prompt while computing logit diff")
        prompt_logits_a = logits_a[index][valid_positions]
        prompt_logits_b = logits_b[index][valid_positions]
        diff = prompt_logits_a - prompt_logits_b
        abs_diff = diff.abs()
        results.append(
            PromptLogitDiff(
                prompt=prompt,
                token_count=token_count,
                max_abs_diff=float(abs_diff.max().item()),
                mean_abs_diff=float(abs_diff.mean().item()),
                rmse=float(diff.pow(2).mean().sqrt().item()),
                allclose=bool(
                    torch_module.allclose(
                        prompt_logits_a,
                        prompt_logits_b,
                        atol=atol,
                        rtol=rtol,
                    )
                ),
            )
        )
    return tuple(results)


def _generate_answer(
    *,
    tokenizer: Any,
    model: Any,
    prompt: str,
    completion: bool,
    max_new_tokens: int,
    device: str,
    torch_module: Any,
) -> str:
    return _generate_answer_batch(
        tokenizer=tokenizer,
        model=model,
        prompts=(prompt,),
        completion=completion,
        max_new_tokens=max_new_tokens,
        device=device,
        torch_module=torch_module,
    )[0]


def _generate_answer_batch(
    *,
    tokenizer: Any,
    model: Any,
    prompts: tuple[str, ...] | list[str],
    completion: bool,
    max_new_tokens: int,
    device: str,
    torch_module: Any,
) -> tuple[str, ...]:
    encoded = _encode_prompt_batch(
        tokenizer=tokenizer,
        prompts=tuple(prompts),
        completion=completion,
    )
    encoded = _move_to_device(encoded, device)
    input_ids = encoded["input_ids"]
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    with _temporary_generation_config(
        model=model,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
    ):
        with torch_module.inference_mode():
            output_ids = model.generate(**encoded)
    generated_ids = output_ids[:, input_ids.shape[-1] :]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return tuple(decoded)


def _encode_prompt_batch(
    *,
    tokenizer: Any,
    prompts: tuple[str, ...],
    completion: bool,
    max_length: int | None = None,
) -> dict[str, Any]:
    if completion:
        return dict(
            tokenizer(
                list(prompts),
                return_tensors="pt",
                padding=True,
                truncation=max_length is not None,
                max_length=max_length,
            )
        )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not support chat templates; use --completion instead")
    rendered_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in prompts
    ]
    return dict(
        tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        )
    )


def _build_deterministic_generation_config(model: Any) -> Any:
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    for field_name in (
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "typical_p",
        "epsilon_cutoff",
        "eta_cutoff",
    ):
        if hasattr(generation_config, field_name):
            setattr(generation_config, field_name, None)
    return generation_config


@contextmanager
def _temporary_generation_config(
    *,
    model: Any,
    max_new_tokens: int,
    pad_token_id: int | None,
) -> Any:
    original_generation_config = model.generation_config
    generation_config = _build_deterministic_generation_config(model)
    generation_config.max_new_tokens = max_new_tokens
    generation_config.pad_token_id = pad_token_id
    model.generation_config = generation_config
    try:
        yield
    finally:
        model.generation_config = original_generation_config


def _prepare_generation_tokenizer(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"


def _resolve_device(torch_module: Any, device: CompareDevice) -> str:
    if device == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        if _mps_is_available(torch_module):
            return "mps"
        return "cpu"
    if device == "cuda" and not torch_module.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device == "mps" and not _mps_is_available(torch_module):
        raise ValueError("MPS was requested but is not available")
    return device


def _resolve_dtype(torch_module: Any, dtype: CompareDType) -> Any:
    if dtype == "auto":
        return "auto"
    return getattr(torch_module, dtype)


def resolve_compare_device(device: CompareDevice) -> str:
    """Resolve the requested compare device to the concrete runtime device."""

    import torch

    return _resolve_device(torch, device)


def _mps_is_available(torch_module: Any) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    return bool(mps_backend is not None and mps_backend.is_available())


def _move_to_device(batch: Any, device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in dict(batch).items()
    }


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.casefold().split())


def _display_name(model_path: str | Path) -> str:
    path = Path(model_path)
    if path.exists():
        return path.name
    return str(model_path)


def _release_resources(*, torch_module: Any) -> None:
    gc.collect()
    _clear_torch_cache(torch_module)


def _clear_torch_cache(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    mps_module = getattr(torch_module, "mps", None)
    mps_is_available = (
        mps_module is not None
        and hasattr(mps_module, "empty_cache")
        and hasattr(mps_module, "is_available")
        and mps_module.is_available()
    )
    if mps_is_available:
        mps_module.empty_cache()


@contextmanager
def _suppress_transformers_loading_report() -> Any:
    loading_report_module = importlib.import_module("transformers.utils.loading_report")
    modeling_utils_module = importlib.import_module("transformers.modeling_utils")
    original_loading_report = loading_report_module.log_state_dict_report
    original_modeling_utils_loading_report = modeling_utils_module.log_state_dict_report
    logger_names = (
        "transformers.utils.loading_report",
        "transformers.modeling_utils",
        "transformers.integrations.tensor_parallel",
    )
    logger_states = []
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger_states.append((logger, logger.disabled, logger.level))
        logger.disabled = True
        logger.setLevel(logging.ERROR)
    loading_report_module.log_state_dict_report = _noop_log_state_dict_report
    modeling_utils_module.log_state_dict_report = _noop_log_state_dict_report
    try:
        yield
    finally:
        loading_report_module.log_state_dict_report = original_loading_report
        modeling_utils_module.log_state_dict_report = original_modeling_utils_loading_report
        for logger, disabled, level in logger_states:
            logger.disabled = disabled
            logger.setLevel(level)


def _noop_log_state_dict_report(**_: Any) -> None:
    return None


class _SemanticScorer:
    def __init__(
        self,
        *,
        semantic_model_name: str | Path,
        device: str,
        dtype: Any,
        torch_module: Any,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch_module
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
        with _suppress_transformers_loading_report():
            self._model = AutoModel.from_pretrained(semantic_model_name, dtype=dtype)
        self._model.eval()
        self._model.to(device)

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        _release_resources(torch_module=self._torch)

    def score_pairs(
        self,
        *,
        answers_a: tuple[str, ...],
        answers_b: tuple[str, ...],
        batch_size: int = 16,
    ) -> tuple[float, ...]:
        if not answers_a:
            return ()
        embeddings_a = self._encode_texts(texts=answers_a, batch_size=batch_size)
        embeddings_b = self._encode_texts(texts=answers_b, batch_size=batch_size)
        similarities = (embeddings_a * embeddings_b).sum(dim=1)
        return tuple(float(value) for value in similarities.tolist())

    def _encode_texts(self, *, texts: tuple[str, ...], batch_size: int) -> Any:
        chunks = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Semantic scoring", unit="batch"):
            batch = texts[start : start + batch_size]
            encoded = self._tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = _move_to_device(encoded, self._device)
            with self._torch.inference_mode():
                outputs = self._model(**encoded)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            masked_hidden = outputs.last_hidden_state * attention_mask
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(normalized.cpu())
        return self._torch.cat(chunks, dim=0)


class _LlmJudge:
    def __init__(
        self,
        *,
        judge_model: str | Path,
        device: str,
        dtype: Any,
        torch_module: Any,
    ) -> None:
        from transformers import AutoTokenizer

        self._torch = torch_module
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(judge_model)
        self._model = _load_causal_lm_for_runtime(
            model_path=str(judge_model),
            dtype=dtype,
            device=device,
        )
        self._model.eval()
        self._model.to(device)

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        _release_resources(torch_module=self._torch)

    def judge(self, *, prompt: str, answer_a: str, answer_b: str) -> bool:
        return self.judge_batch(
            prompts=(prompt,),
            answers_a=(answer_a,),
            answers_b=(answer_b,),
            batch_size=1,
        )[0]

    def judge_batch(
        self,
        *,
        prompts: tuple[str, ...],
        answers_a: tuple[str, ...],
        answers_b: tuple[str, ...],
        batch_size: int,
    ) -> tuple[bool, ...]:
        judge_prompts = tuple(
            _build_judge_prompt(prompt=prompt, answer_a=answer_a, answer_b=answer_b)
            for prompt, answer_a, answer_b in zip(prompts, answers_a, answers_b, strict=True)
        )
        judged_values = []
        num_batches = (len(judge_prompts) + batch_size - 1) // batch_size
        for start in tqdm(
            range(0, len(judge_prompts), batch_size),
            total=num_batches,
            desc="Judging outputs",
            unit="batch",
        ):
            batch_prompts = judge_prompts[start : start + batch_size]
            responses = _generate_answer_batch(
                tokenizer=self._tokenizer,
                model=self._model,
                prompts=batch_prompts,
                completion=not hasattr(self._tokenizer, "apply_chat_template"),
                max_new_tokens=DEFAULT_JUDGE_MAX_NEW_TOKENS,
                device=self._device,
                torch_module=self._torch,
            )
            judged_values.extend(_parse_judge_response(response) for response in responses)
        return tuple(judged_values)


def _build_judge_prompt(*, prompt: str, answer_a: str, answer_b: str) -> str:
    return (
        "Decide whether two answers mean the same thing for the given user prompt. "
        "Ignore wording differences, but do not ignore factual disagreement, missing key details, "
        "or contradictory claims. Respond with only YES or NO.\n\n"
        f"User prompt:\n{prompt}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}\n"
    )


def _parse_judge_response(response: str) -> bool:
    normalized = _normalize_text(response)
    return normalized.startswith("yes")


__all__ = [
    "ComparisonSummary",
    "CompareDType",
    "CompareDevice",
    "DEFAULT_COMPARE_PROMPT",
    "DEFAULT_SEMANTIC_MODEL_NAME",
    "DEFAULT_SEMANTIC_THRESHOLD",
    "LogitDiffSummary",
    "PromptComparison",
    "PromptLogitDiff",
    "comparison_has_any_mismatch",
    "compare_model_logits",
    "compare_models",
    "load_prompts",
    "resolve_compare_device",
    "serialize_comparisons",
    "serialize_logit_diff_summary",
    "serialize_logit_diffs",
    "serialize_prompt_logit_diffs",
    "serialize_prompt_comparisons",
    "summarize_logit_diffs",
    "serialize_summary",
    "summarize_comparisons",
]
