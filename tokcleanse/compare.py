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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = None
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        _prepare_generation_tokenizer(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
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
) -> dict[str, Any]:
    if completion:
        return dict(tokenizer(list(prompts), return_tensors="pt", padding=True))
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
    return dict(tokenizer(rendered_prompts, return_tensors="pt", padding=True))


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
    if mps_module is not None and hasattr(mps_module, "empty_cache"):
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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch_module
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(judge_model)
        self._model = AutoModelForCausalLM.from_pretrained(judge_model, dtype=dtype)
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
    "PromptComparison",
    "comparison_has_any_mismatch",
    "compare_models",
    "load_prompts",
    "resolve_compare_device",
    "serialize_comparisons",
    "serialize_prompt_comparisons",
    "serialize_summary",
    "summarize_comparisons",
]
