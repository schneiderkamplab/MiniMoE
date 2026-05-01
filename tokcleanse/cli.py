"""Command-line interface for tokcleanse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from .cleaner import DEFAULT_SAVE_ORDER_NAME, available_order_names, describe_order
from .compare import (
    comparison_has_any_mismatch,
    compare_model_logits,
    CompareDType,
    DEFAULT_COMPARE_PROMPT,
    DEFAULT_GENERATION_BATCH_SIZE,
    DEFAULT_SEMANTIC_MODEL_NAME,
    DEFAULT_SEMANTIC_THRESHOLD,
    compare_models,
    load_prompts,
    resolve_compare_device,
    serialize_logit_diff_summary,
    serialize_prompt_logit_diffs,
    serialize_prompt_comparisons,
    serialize_summary,
)
from .downloader import download_model_snapshot
from .loader import load_tokenizer_contents
from .model_surgery import (
    GEMMA4_DEFAULT_EMBEDDING_WEIGHT_NAMES,
    GEMMA4_DEFAULT_LM_HEAD_WEIGHT_NAMES,
)
from .saver import _SanitizeSummary, _save_reordered_tokenizer_with_summary
from .train import (
    TrainingDryRunReport,
    DEFAULT_DISTILL_KL_VOCAB_CHUNK_SIZE,
    DEFAULT_LOG_EVERY,
    DEFAULT_PRINT_EVERY,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TRAIN_DTYPE,
    DEFAULT_TRAIN_EVAL_EVERY,
    DEFAULT_TRAIN_EVAL_MAX_BATCHES,
    DEFAULT_TRAIN_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_ROUTER_LEARNING_RATE,
    DEFAULT_ROUTER_LR_ANNEAL_STEPS,
    DEFAULT_ROUTER_LR_WARMUP_STEPS,
    DEFAULT_ROUTER_MIN_LEARNING_RATE,
    DEFAULT_TRAIN_LEARNING_RATE,
    DEFAULT_TRAIN_MAX_GRAD_NORM,
    DEFAULT_TRAIN_MIN_LEARNING_RATE,
    DEFAULT_TRAIN_STEPS,
    DEFAULT_TRAIN_TORCH_COMPILE,
    DEFAULT_TRAIN_LR_WARMUP_STEPS,
    DEFAULT_WEIGHT_DIFF_EVERY,
    build_training_dry_run_report,
    train_distilled_model,
)
from .upcycle import ExpertInit, upcycle_gemma4_model

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Tokenizer/model sanitizing and comparison utilities.",
)


def _order_help() -> str:
    lines = ["Built-in topological-sort order to use when rewriting merges.", "", "Orders:"]
    lines.append(f"  original: {describe_order('original')}")
    for name in available_order_names():
        suffix = " (default)" if name == DEFAULT_SAVE_ORDER_NAME else ""
        lines.append(f"  {name}{suffix}: {describe_order(name)}")
    return "\n".join(lines)


@app.command("sanitize")
def sanitize_command(
    source: str = typer.Argument(..., help="Local tokenizer directory or Hugging Face repo id."),
    destination: Path = typer.Argument(..., help="Output directory for the reordered tokenizer."),
    order_name: str = typer.Option(
        DEFAULT_SAVE_ORDER_NAME,
        "--order",
        "-o",
        help=_order_help(),
    ),
    seed: int | None = typer.Option(
        None,
        help="Random seed for a random topological sort. Only valid with the original order.",
    ),
    models_dir: Path = typer.Option(
        Path("models"),
        help="Base directory for local Hugging Face snapshots.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace the destination directory if it already exists.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Print the concrete token, merge, and special-token sets in addition to summary counts.",
    ),
    reassign: bool = typer.Option(
        False,
        "--reassign",
        help="Reassign token ids so specials come first, then base tokens, then merged tokens.",
    ),
    special_token_map_file: Path | None = typer.Option(
        None,
        "--special-token-map",
        help=(
            'JSON file with {"keep": [...], "rename": {...}} for special tokens. Omitted '
            "current special tokens are removed before reassignment."
        ),
    ),
    token_map_file: Path | None = typer.Option(
        None,
        "--token-map",
        help=(
            'JSON file with {"delete": [...], "add": [...], "rename": {...}} for non-special '
            "tokens. Deletes cascade through merge-derived tokens."
        ),
    ),
    strip_multimodal: bool = typer.Option(
        False,
        "--strip-multimodal",
        help="Remove Gemma 4 audio/vision tower tensors and rewrite copied metadata into a text-only shape.",
    ),
    embedding_weight_names: list[str] = typer.Option(
        list(GEMMA4_DEFAULT_EMBEDDING_WEIGHT_NAMES),
        "--embedding-weight",
        help="Tensor name to reindex for token embeddings. Repeat to include multiple tensors.",
    ),
    lm_head_weight_names: list[str] = typer.Option(
        list(GEMMA4_DEFAULT_LM_HEAD_WEIGHT_NAMES),
        "--lm-head-weight",
        help="Tensor name to reindex for LM head weights. Repeat to include multiple tensors.",
    ),
) -> None:
    """Save a fully copied tokenizer directory with reordered merges."""

    if special_token_map_file is not None and not reassign:
        raise typer.BadParameter("--special-token-map requires --reassign")
    if token_map_file is not None and not reassign:
        raise typer.BadParameter("--token-map requires --reassign")
    contents = load_tokenizer_contents(source, models_dir=models_dir)
    special_token_literal_map = None
    token_delete_literals: tuple[str, ...] = ()
    token_add_literals: tuple[str, ...] = ()
    token_rename_literals: dict[str, str] = {}
    if special_token_map_file is not None:
        special_token_literal_map = _load_special_token_literal_map_file(special_token_map_file)
    if token_map_file is not None:
        token_delete_literals, token_add_literals, token_rename_literals = _load_token_map_file(token_map_file)
    saved_path, summary = _save_reordered_tokenizer_with_summary(
        contents,
        destination,
        order_name=order_name,
        seed=seed,
        overwrite=overwrite,
        reassign=reassign,
        special_token_literal_map=special_token_literal_map,
        token_delete_literals=token_delete_literals,
        token_add_literals=token_add_literals,
        token_rename_literals=token_rename_literals,
        strip_multimodal=strip_multimodal,
        embedding_weight_names=tuple(embedding_weight_names),
        lm_head_weight_names=tuple(lm_head_weight_names),
    )
    typer.echo(str(saved_path))
    _echo_sanitize_run_details(
        saved_path=saved_path,
        order_name=order_name,
        reassign=reassign,
        strip_multimodal=strip_multimodal,
    )
    typer.echo(
        (
            "Special tokens: "
            f"kept={summary.special_tokens_kept}, "
            f"renamed={summary.special_tokens_renamed}, "
            f"dropped={summary.special_tokens_dropped}"
        ),
        err=True,
    )
    typer.echo(
        (
            "Tokens: "
            f"delete_requested={summary.requested_tokens_deleted}, "
            f"rename_requested={summary.requested_tokens_renamed}, "
            f"deleted_total={summary.tokens_deleted_total}, "
            f"add_requested={summary.requested_tokens_added}, "
            f"existing_ignored={summary.existing_tokens_ignored}, "
            f"requested_added={summary.requested_tokens_added_total}, "
            f"intermediate_added={summary.intermediate_tokens_added}"
        ),
        err=True,
    )
    typer.echo(
        (
            "Merges: "
            f"added={summary.synthetic_merges_added}, "
            f"deleted={summary.original_merges_deleted}"
        ),
        err=True,
    )
    if verbose:
        _echo_verbose_sanitize_summary(summary)


@app.command("download")
def download_command(
    repo_id: str = typer.Argument(..., help="Hugging Face model repo id to download."),
    models_dir: Path = typer.Option(
        Path("models"),
        help="Base directory where the snapshot will be stored as models/HF_ID.",
    ),
    force_download: bool = typer.Option(
        False,
        "--force-download",
        help="Force a fresh download even if files are already cached locally.",
    ),
) -> None:
    """Download a Hugging Face model snapshot into the local models directory."""

    downloaded_path = download_model_snapshot(
        repo_id,
        models_dir=models_dir,
        force_download=force_download,
    )
    typer.echo(str(downloaded_path))


@app.command("compare")
def compare_command(
    model_a: str = typer.Argument(..., help="First local model directory or Hugging Face repo id."),
    model_b: str = typer.Argument(..., help="Second local model directory or Hugging Face repo id."),
    prompt_texts: list[str] = typer.Option(
        None,
        "--prompt",
        help=(
            "Prompt text to compare. Repeat to include multiple prompts unless --prompt-file is used. "
            f"Defaults to {DEFAULT_COMPARE_PROMPT!r} when no prompts are provided."
        ),
    ),
    prompt_file: Path | None = typer.Option(
        None,
        "--prompt-file",
        help='JSONL file with one object per line: {"text": "..."}. When provided, repeated --prompt values are ignored.',
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output-file",
        help="Write per-prompt comparison JSONL to this file instead of printing all prompt rows to stdout.",
    ),
    completion: bool = typer.Option(
        False,
        "--completion/--no-completion",
        help="Use direct completion instead of chat-template driven generation.",
    ),
    max_new_tokens: int = typer.Option(
        128,
        "--max-new-tokens",
        help="Maximum number of new tokens to generate per prompt.",
    ),
    batch_size: int = typer.Option(
        DEFAULT_GENERATION_BATCH_SIZE,
        "--batch-size",
        help="Batch size for model-a/model-b prompt generation.",
    ),
    judge_batch_size: int | None = typer.Option(
        None,
        "--judge-batch-size",
        help="Batch size for judge-model generations. Defaults to --batch-size.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Generation and scoring device: auto, cpu, cuda, or mps.",
    ),
    dtype: CompareDType = typer.Option(
        "auto",
        "--dtype",
        help="Model weight dtype for generation and scoring: auto, float32, float16, or bfloat16.",
    ),
    semantic_model_name: str = typer.Option(
        DEFAULT_SEMANTIC_MODEL_NAME,
        "--semantic-model",
        help="Sentence-embedding model used for semantic similarity scoring.",
    ),
    semantic_threshold: float = typer.Option(
        DEFAULT_SEMANTIC_THRESHOLD,
        "--semantic-threshold",
        help="Cosine-similarity threshold used to count semantic close matches.",
    ),
    llm_judge: str | None = typer.Option(
        None,
        "--llm-judge",
        help="Judge model repo id or local path for unresolved non-exact, non-semantic mismatches.",
    ),
    judge_all: bool = typer.Option(
        False,
        "--judge-all",
        help="Run the judge for every prompt instead of only unresolved mismatches.",
    ),
) -> None:
    """Run prompts through two models, print their answers, and compare them."""

    prompts = load_prompts(prompt_texts=prompt_texts or [], prompt_file=prompt_file)
    try:
        resolved_device = resolve_compare_device(device)
        typer.echo(f"Using device: {resolved_device}", err=True)
        comparisons = compare_models(
            model_a,
            model_b,
            prompts=prompts,
            completion=completion,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            judge_batch_size=judge_batch_size,
            device=resolved_device,
            dtype=dtype,
            semantic_model_name=semantic_model_name,
            semantic_threshold=semantic_threshold,
            llm_judge=llm_judge,
            judge_all=judge_all,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    prompt_lines = serialize_prompt_comparisons(comparisons)
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("".join(f"{line}\n" for line in prompt_lines), encoding="utf-8")
        for comparison, line in zip(comparisons, prompt_lines, strict=True):
            if comparison_has_any_mismatch(comparison):
                typer.echo(line)
    else:
        for line in prompt_lines:
            typer.echo(line)
    typer.echo(serialize_summary(comparisons))


@app.command("logit-diff")
def logit_diff_command(
    model_a: str = typer.Argument(..., help="First local model directory or Hugging Face repo id."),
    model_b: str = typer.Argument(..., help="Second local model directory or Hugging Face repo id."),
    prompt_texts: list[str] = typer.Option(
        None,
        "--prompt",
        help=(
            "Prompt text to compare. Repeat to include multiple prompts unless --prompt-file is used. "
            f"Defaults to {DEFAULT_COMPARE_PROMPT!r} when no prompts are provided."
        ),
    ),
    prompt_file: Path | None = typer.Option(
        None,
        "--prompt-file",
        help='JSONL file with one object per line: {"text": "..."}. When provided, repeated --prompt values are ignored.',
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output-file",
        help="Write per-prompt logit-diff JSONL to this file instead of printing all prompt rows to stdout.",
    ),
    completion: bool = typer.Option(
        False,
        "--completion/--no-completion",
        help="Use direct completion-style tokenization instead of chat-template driven tokenization.",
    ),
    batch_size: int = typer.Option(
        DEFAULT_GENERATION_BATCH_SIZE,
        "--batch-size",
        help="Batch size for prompt encoding and logit comparison.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Forward-pass device: auto, cpu, cuda, or mps.",
    ),
    dtype: CompareDType = typer.Option(
        "auto",
        "--dtype",
        help="Model weight dtype for forward passes: auto, float32, float16, or bfloat16.",
    ),
    max_length: int = typer.Option(
        1024,
        "--max-length",
        help="Maximum tokenized prompt length. Longer prompts are truncated.",
    ),
    atol: float = typer.Option(
        1e-5,
        "--atol",
        help="Absolute tolerance used for the per-prompt allclose check.",
    ),
    rtol: float = typer.Option(
        1e-5,
        "--rtol",
        help="Relative tolerance used for the per-prompt allclose check.",
    ),
) -> None:
    """Compare prompt-conditioned logits between two aligned models."""

    prompts = load_prompts(prompt_texts=prompt_texts or [], prompt_file=prompt_file)
    try:
        resolved_device = resolve_compare_device(device)
        typer.echo(f"Using device: {resolved_device}", err=True)
        diffs = compare_model_logits(
            model_a,
            model_b,
            prompts=prompts,
            completion=completion,
            batch_size=batch_size,
            device=resolved_device,
            dtype=dtype,
            max_length=max_length,
            atol=atol,
            rtol=rtol,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    prompt_lines = serialize_prompt_logit_diffs(diffs)
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("".join(f"{line}\n" for line in prompt_lines), encoding="utf-8")
    else:
        for line in prompt_lines:
            typer.echo(line)
    typer.echo(serialize_logit_diff_summary(diffs))


@app.command("upcycle")
def upcycle_command(
    source: str = typer.Argument(..., help="Local Gemma 4 model directory or Hugging Face repo id."),
    destination: Path = typer.Argument(..., help="Output directory for the upcycled model."),
    models_dir: Path = typer.Option(
        Path("models"),
        help="Base directory for local Hugging Face snapshots.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace the destination directory if it already exists.",
    ),
    num_experts: int = typer.Option(
        2,
        "--num-experts",
        help="Number of MoE experts to create per decoder layer.",
    ),
    top_k_experts: int = typer.Option(
        2,
        "--top-k-experts",
        help="Number of experts to route each token to.",
    ),
    expert_init: ExpertInit = typer.Option(
        "copy",
        "--expert-init",
        help="Expert initialization mode: copy or zero.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Print MoE configuration and rewritten artifact details after upcycling.",
    ),
) -> None:
    """Copy a Gemma 4 model directory and upcycle its dense MLPs into MoE-ready tensors."""

    try:
        upcycled_path = upcycle_gemma4_model(
            source,
            destination,
            models_dir=models_dir,
            overwrite=overwrite,
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            expert_init=expert_init,
        )
    except (FileExistsError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(upcycled_path))
    _echo_upcycle_summary(
        upcycled_path=upcycled_path,
        source=source,
        expert_init=expert_init,
        verbose=verbose,
    )


@app.command("train")
def train_command(
    student_model: str = typer.Argument(..., help="Student model directory or Hugging Face repo id."),
    teacher_model: str = typer.Argument(..., help="Teacher model directory or Hugging Face repo id."),
    output_dir: Path = typer.Argument(..., help="Directory where the trained student checkpoint will be written."),
    ind_files: list[Path] = typer.Option(
        [],
        "--ind-file",
        help='In-distribution corpus file. Repeat as needed. `.jsonl`, `.jsonl.gz`, and `.jinx` expect `{"text": "..."}` samples; other files are treated as one example per non-empty line.',
    ),
    ood_files: list[Path] = typer.Option(
        [],
        "--ood-file",
        help='Out-of-distribution corpus file. Repeat as needed. `.jsonl`, `.jsonl.gz`, and `.jinx` expect `{"text": "..."}` samples; other files are treated as one example per non-empty line.',
    ),
    eval_files: list[Path] = typer.Option(
        [],
        "--eval-file",
        help='Evaluation corpus file. Repeat as needed. `.jsonl`, `.jsonl.gz`, and `.jinx` expect `{"text": "..."}` samples; other files are treated as one example per non-empty line.',
    ),
    steps: int = typer.Option(
        DEFAULT_TRAIN_STEPS,
        "--steps",
        help="Optimizer steps to run. Negative values mean epochs: -1 is 1 epoch, -2 is 2 epochs, and so on.",
    ),
    batch_size: int = typer.Option(
        DEFAULT_TRAIN_BATCH_SIZE,
        "--batch-size",
        help="Per-language microbatch size.",
    ),
    resume_from: Path | None = typer.Option(
        None,
        "--resume-from",
        help="Resume training from a checkpoint directory containing saved weights and training_state.pt.",
    ),
    eval_batch_size: int | None = typer.Option(
        None,
        "--eval-batch-size",
        help="Evaluation batch size. Defaults to --batch-size.",
    ),
    eval_max_batches: int = typer.Option(
        DEFAULT_TRAIN_EVAL_MAX_BATCHES,
        "--eval-max-batches",
        help="Maximum number of evaluation batches to process per eval pass. Use 0 for the full eval set.",
    ),
    weight_diff_every: int = typer.Option(
        DEFAULT_WEIGHT_DIFF_EVERY,
        "--weight-diff-every",
        help="Compute router/expert parameter-difference metrics every N optimizer steps. Use 0 to disable.",
    ),
    pad_to_max_length: bool | None = typer.Option(
        None,
        "--pad-to-max-length/--no-pad-to-max-length",
        help="Force fixed max-length padding on or off. By default, tokcleanse uses fixed padding on MPS and dynamic padding elsewhere.",
    ),
    gradient_accumulation_steps: int = typer.Option(
        DEFAULT_TRAIN_GRADIENT_ACCUMULATION_STEPS,
        "--gradient-accumulation",
        help="Number of microbatches to accumulate before each optimizer step.",
    ),
    gradient_checkpointing: bool = typer.Option(
        True,
        "--gradient-checkpointing/--no-gradient-checkpointing",
        help="Enable activation checkpointing on the student model to reduce memory usage.",
    ),
    torch_compile: bool | None = typer.Option(
        DEFAULT_TRAIN_TORCH_COMPILE,
        "--torch-compile/--no-torch-compile",
        help="Enable torch.compile on the student model. By default this is on for CUDA and off for MPS/CPU.",
    ),
    distill_ind: bool = typer.Option(
        True,
        "--distill-ind/--no-distill-ind",
        help="Enable or disable teacher distillation on in-distribution batches.",
    ),
    distill_ood: bool = typer.Option(
        True,
        "--distill-ood/--no-distill-ood",
        help="Enable or disable teacher distillation on out-of-distribution batches.",
    ),
    distill_original_tokens_only: bool = typer.Option(
        False,
        "--distill-original-tokens-only/--no-distill-original-tokens-only",
        help="Restrict teacher KL to the original vocabulary recorded in tokenizer_token_groups.json.",
    ),
    distill_every: int = typer.Option(
        1,
        "--distill-every",
        help="Run teacher distillation every N optimizer steps on enabled streams.",
    ),
    max_length: int = typer.Option(
        1024,
        "--max-length",
        help="Maximum sequence length for tokenization and training.",
    ),
    learning_rate: float = typer.Option(
        DEFAULT_TRAIN_LEARNING_RATE,
        "--learning-rate",
        help="AdamW learning rate for non-router trainable parameters.",
    ),
    min_learning_rate: float = typer.Option(
        DEFAULT_TRAIN_MIN_LEARNING_RATE,
        "--min-learning-rate",
        help="Minimum non-router learning rate after cosine decay.",
    ),
    max_grad_norm: float = typer.Option(
        DEFAULT_TRAIN_MAX_GRAD_NORM,
        "--max-grad-norm",
        help="Clip main and router gradient norms separately to this value before each optimizer step. Use 0 to disable.",
    ),
    router_learning_rate: float = typer.Option(
        DEFAULT_ROUTER_LEARNING_RATE,
        "--router-learning-rate",
        help="AdamW learning rate for router parameters.",
    ),
    router_lr_warmup_steps: int = typer.Option(
        DEFAULT_ROUTER_LR_WARMUP_STEPS,
        "--router-lr-warmup-steps",
        help="Number of optimizer steps for linear router LR warmup before router cosine annealing.",
    ),
    router_lr_anneal_steps: int = typer.Option(
        DEFAULT_ROUTER_LR_ANNEAL_STEPS,
        "--router-lr-anneal-steps",
        help="Number of optimizer steps after router warmup for cosine annealing the router learning rate to --router-min-learning-rate.",
    ),
    router_min_learning_rate: float = typer.Option(
        DEFAULT_ROUTER_MIN_LEARNING_RATE,
        "--router-min-learning-rate",
        help="Minimum router learning rate after cosine annealing.",
    ),
    weight_decay: float = typer.Option(
        0.0,
        "--weight-decay",
        help="AdamW weight decay.",
    ),
    distill_kl_vocab_chunk_size: int = typer.Option(
        DEFAULT_DISTILL_KL_VOCAB_CHUNK_SIZE,
        "--distill-kl-vocab-chunk-size",
        help="Teacher-KL vocabulary chunk size. Use 0 to disable chunking and compute full-vocab KL in one shot for speed at higher memory cost.",
    ),
    checkpoint_every: int | None = typer.Option(
        None,
        "--checkpoint-every",
        help="Write a checkpoint every N optimizer steps. Defaults to the same value as --eval-every; set 0 to disable checkpointing.",
    ),
    checkpoint_dir: Path = typer.Option(
        Path("checkpoints"),
        "--checkpoint-dir",
        help="Directory where periodic `stepN` checkpoints and a `final` checkpoint will be written.",
    ),
    eval_every_steps: int = typer.Option(
        DEFAULT_TRAIN_EVAL_EVERY,
        "--eval-every",
        help="Run evaluation on --eval-file inputs every N optimizer steps. Set 0 to disable.",
    ),
    log_every: int = typer.Option(
        DEFAULT_LOG_EVERY,
        "--log-every",
        help="Write one JSONL metrics record every N optimizer steps.",
    ),
    print_every: int = typer.Option(
        DEFAULT_PRINT_EVERY,
        "--print-every",
        help="Print one compact training-metrics line to stdout every N optimizer steps.",
    ),
    ind_batches_per_cycle: int = typer.Option(
        1,
        "--ind-batches-per-cycle",
        help="How many in-distribution batches to draw before the schedule advances.",
    ),
    ood_batches_per_cycle: int = typer.Option(
        1,
        "--ood-batches-per-cycle",
        help="How many out-of-distribution batches to draw before the schedule advances.",
    ),
    ind_lm_weight: float = typer.Option(
        0.05,
        "--ind-lm-weight",
        help="Language-model loss weight for in-distribution batches.",
    ),
    ind_distill_weight: float = typer.Option(
        1.0,
        "--ind-distill-weight",
        help="Teacher-distillation KL weight for in-distribution batches.",
    ),
    ind_route_weight: float = typer.Option(
        0.0,
        "--ind-route-weight",
        help="Router loss weight for in-distribution batches. Positive values push IND tokens toward expert 0.",
    ),
    ind_route_logit_bias: float = typer.Option(
        0.0,
        "--ind-route-logit-bias",
        help="Expert-1 router-logit bias for in-distribution batches. Negative values push IND toward expert 0.",
    ),
    ood_lm_weight: float = typer.Option(
        1.0,
        "--ood-lm-weight",
        help="Language-model loss weight for out-of-distribution batches.",
    ),
    ood_distill_weight: float = typer.Option(
        0.25,
        "--ood-distill-weight",
        help="Teacher-distillation KL weight for out-of-distribution batches.",
    ),
    ood_route_weight: float = typer.Option(
        0.0,
        "--ood-route-weight",
        help="Router loss weight for out-of-distribution batches. Positive values push OOD tokens toward expert 1.",
    ),
    ood_route_logit_bias: float = typer.Option(
        0.0,
        "--ood-route-logit-bias",
        help="Expert-1 router-logit bias for out-of-distribution batches. Positive values push OOD toward expert 1.",
    ),
    route_logit_bias_anneal_steps: int = typer.Option(
        0,
        "--route-logit-bias-anneal-steps",
        help="Cosine-anneal route logit biases to zero over this many optimizer steps. Use 0 to keep biases constant.",
    ),
    route_logit_bias_anneal_offset_steps: int = typer.Option(
        0,
        "--route-logit-bias-anneal-offset-steps",
        help="Keep route logit biases at full strength for this many optimizer steps before bias annealing can advance.",
    ),
    route_logit_bias_anneal_loss_threshold: float = typer.Option(
        5e-2,
        "--route-logit-bias-anneal-loss-threshold",
        help="Advance route-logit-bias annealing only when the optimizer-step router loss is at or below this threshold.",
    ),
    train_shared: bool = typer.Option(
        False,
        "--train-shared/--freeze-shared",
        help="Train or freeze shared non-router, non-expert backbone parameters.",
    ),
    train_expert_0: bool = typer.Option(
        False,
        "--train-expert-0/--freeze-expert-0",
        help="Train or freeze expert 0 weights.",
    ),
    train_expert_1: bool = typer.Option(
        True,
        "--train-expert-1/--freeze-expert-1",
        help="Train or freeze expert 1 weights.",
    ),
    train_embedding_lm_head: bool = typer.Option(
        True,
        "--train-embedding-lmhead/--freeze-embedding-lmhead",
        help="Train or freeze embedding and LM-head weights. When enabled, only added token rows train.",
    ),
    train_full_embedding_lm_head: bool = typer.Option(
        False,
        "--train-full-embedding-lmhead/--train-added-embedding-lmhead",
        help="Train all embedding and LM-head rows instead of only added token rows.",
    ),
    lr_warmup_steps: int = typer.Option(
        DEFAULT_TRAIN_LR_WARMUP_STEPS,
        "--lr-warmup-steps",
        help="Number of optimizer steps for linear warmup before cosine decay to --min-learning-rate over the resolved total steps.",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Training device: auto, cpu, cuda, or mps.",
    ),
    dtype: CompareDType = typer.Option(
        DEFAULT_TRAIN_DTYPE,
        "--dtype",
        help="Training/model dtype: auto, float32, float16, or bfloat16.",
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Random seed used for corpus cycling.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Replace the output directory if it already exists.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print which parameters, optimizer groups, and token rows would train, then exit.",
    ),
) -> None:
    """Train an upcycled student model against a frozen teacher with aligned tokenizers."""

    try:
        resolved_device = resolve_compare_device(device)
        resolved_checkpoint_every = eval_every_steps if checkpoint_every is None else checkpoint_every
        typer.echo(f"Using device: {resolved_device}", err=True)
        if dry_run:
            report = build_training_dry_run_report(
                student_model,
                resume_from=resume_from,
                ind_files=tuple(ind_files),
                ood_files=tuple(ood_files),
                device=resolved_device,
                dtype=dtype,
                steps=steps,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                eval_max_batches=eval_max_batches,
                weight_diff_every=weight_diff_every,
                pad_to_max_length=pad_to_max_length,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                torch_compile=torch_compile,
                distill_ind=distill_ind,
                distill_ood=distill_ood,
                distill_original_tokens_only=distill_original_tokens_only,
                distill_every=distill_every,
                lr_warmup_steps=lr_warmup_steps,
                learning_rate=learning_rate,
                min_learning_rate=min_learning_rate,
                max_grad_norm=max_grad_norm,
                router_learning_rate=router_learning_rate,
                router_lr_warmup_steps=router_lr_warmup_steps,
                router_lr_anneal_steps=router_lr_anneal_steps,
                router_min_learning_rate=router_min_learning_rate,
                weight_decay=weight_decay,
                ind_route_weight=ind_route_weight,
                ind_route_logit_bias=ind_route_logit_bias,
                ood_route_weight=ood_route_weight,
                ood_route_logit_bias=ood_route_logit_bias,
                route_logit_bias_anneal_steps=route_logit_bias_anneal_steps,
                route_logit_bias_anneal_offset_steps=route_logit_bias_anneal_offset_steps,
                route_logit_bias_anneal_loss_threshold=route_logit_bias_anneal_loss_threshold,
                train_shared=train_shared,
                train_expert_0=train_expert_0,
                train_expert_1=train_expert_1,
                train_embedding_lm_head=train_embedding_lm_head,
                train_full_embedding_lm_head=train_full_embedding_lm_head,
                distill_kl_vocab_chunk_size=distill_kl_vocab_chunk_size,
                checkpoint_every=resolved_checkpoint_every,
                checkpoint_dir=checkpoint_dir,
                ind_batches_per_cycle=ind_batches_per_cycle,
                ood_batches_per_cycle=ood_batches_per_cycle,
            )
            typer.echo(_serialize_training_dry_run_report(report))
            return
        trained_path = train_distilled_model(
            student_model,
            teacher_model,
            output_dir,
            resume_from=resume_from,
            ind_files=tuple(ind_files),
            ood_files=tuple(ood_files),
            eval_files=tuple(eval_files),
            device=resolved_device,
            dtype=dtype,
            steps=steps,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            eval_max_batches=eval_max_batches,
            weight_diff_every=weight_diff_every,
            pad_to_max_length=pad_to_max_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            torch_compile=torch_compile,
            distill_ind=distill_ind,
            distill_ood=distill_ood,
            distill_original_tokens_only=distill_original_tokens_only,
            distill_every=distill_every,
            lr_warmup_steps=lr_warmup_steps,
            max_length=max_length,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            max_grad_norm=max_grad_norm,
            router_learning_rate=router_learning_rate,
            router_lr_warmup_steps=router_lr_warmup_steps,
            router_lr_anneal_steps=router_lr_anneal_steps,
            router_min_learning_rate=router_min_learning_rate,
            weight_decay=weight_decay,
            distill_kl_vocab_chunk_size=distill_kl_vocab_chunk_size,
            checkpoint_every=resolved_checkpoint_every,
            checkpoint_dir=checkpoint_dir,
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
            train_shared=train_shared,
            train_expert_0=train_expert_0,
            train_expert_1=train_expert_1,
            train_embedding_lm_head=train_embedding_lm_head,
            train_full_embedding_lm_head=train_full_embedding_lm_head,
            seed=seed,
            overwrite=overwrite,
        )
    except (FileExistsError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(trained_path))


def main() -> None:
    """Run the tokcleanse CLI."""

    app(prog_name="tokcleanse")


def _load_special_token_literal_map_file(path: Path) -> dict[str, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON in {path}") from exc
    if not isinstance(data, dict):
        raise typer.BadParameter(f"{path} must contain a JSON object")

    keep = data.get("keep", [])
    rename = data.get("rename", {})
    if not isinstance(keep, list) or not all(isinstance(item, str) for item in keep):
        raise typer.BadParameter(f"{path} keep must be a list of strings")
    if not isinstance(rename, dict):
        raise typer.BadParameter(f"{path} rename must be a JSON object")

    mapping: dict[str, str | None] = {literal: None for literal in keep}
    for key, value in rename.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise typer.BadParameter(f"{path} rename must map strings to strings")
        if key in mapping:
            raise typer.BadParameter(f"{path} cannot list the same token in both keep and rename: {key}")
        mapping[key] = value
    return mapping


def _load_token_map_file(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"Invalid JSON in {path}") from exc
    if not isinstance(data, dict):
        raise typer.BadParameter(f"{path} must contain a JSON object")

    delete = data.get("delete", [])
    add = data.get("add", [])
    rename = data.get("rename", {})
    if not isinstance(delete, list) or not all(isinstance(item, str) for item in delete):
        raise typer.BadParameter(f"{path} delete must be a list of strings")
    if not isinstance(add, list) or not all(isinstance(item, str) for item in add):
        raise typer.BadParameter(f"{path} add must be a list of strings")
    if not isinstance(rename, dict):
        raise typer.BadParameter(f"{path} rename must be a JSON object")

    if len(set(delete)) != len(delete):
        raise typer.BadParameter(f"{path} delete cannot contain duplicate tokens")
    if len(set(add)) != len(add):
        raise typer.BadParameter(f"{path} add cannot contain duplicate tokens")
    rename_pairs: dict[str, str] = {}
    for key, value in rename.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise typer.BadParameter(f"{path} rename must map strings to strings")
        rename_pairs[key] = value
    overlap = sorted(set(delete) & set(add))
    if overlap:
        raise typer.BadParameter(
            f"{path} cannot list the same token in both delete and add: {', '.join(overlap)}"
        )
    return tuple(delete), tuple(add), rename_pairs


def _echo_verbose_sanitize_summary(summary: _SanitizeSummary) -> None:
    _echo_json_detail("Requested token deletes", summary.requested_token_deletes)
    _echo_json_detail("Requested token renames", dict(summary.requested_token_renames))
    _echo_json_detail("Deleted tokens", summary.deleted_tokens)
    _echo_json_detail("Requested token adds", summary.requested_token_adds)
    _echo_json_detail("Existing add tokens ignored", summary.existing_ignored_tokens)
    _echo_json_detail("Requested added tokens", summary.requested_added_tokens)
    _echo_json_detail("Intermediate added tokens", summary.intermediate_added_tokens)
    _echo_json_detail("Added merges", summary.synthetic_merge_pairs)
    _echo_json_detail("Deleted merges", summary.deleted_merge_pairs)
    _echo_json_detail("Kept special tokens", summary.special_tokens_kept_literals)
    _echo_json_detail("Renamed special tokens", dict(summary.special_tokens_renamed_pairs))
    _echo_json_detail("Dropped special tokens", summary.special_tokens_dropped_literals)


def _echo_json_detail(label: str, value: object) -> None:
    typer.echo(f"{label}: {json.dumps(value, ensure_ascii=False)}", err=True)


def _serialize_training_dry_run_report(report: TrainingDryRunReport) -> str:
    return json.dumps(
        {
            "dry_run": {
                "resolved_device": report.resolved_device,
                "dtype": report.dtype,
                "resume_from": report.resume_from,
                "resume_step": report.resume_step,
                "requested_steps": report.requested_steps,
                "resolved_steps": report.resolved_steps,
                "resolved_epochs": report.resolved_epochs,
                "batch_size": report.batch_size,
                "eval_batch_size": report.eval_batch_size,
                "eval_max_batches": report.eval_max_batches,
                "weight_diff_every": report.weight_diff_every,
                "pad_to_max_length": report.pad_to_max_length,
                "torch_compile": report.torch_compile,
                "distill_ind": report.distill_ind,
                "distill_ood": report.distill_ood,
                "distill_original_tokens_only": report.distill_original_tokens_only,
                "distill_every": report.distill_every,
                "lr_warmup_steps": report.lr_warmup_steps,
                "learning_rate_schedule": "linear_warmup_cosine_decay",
                "gradient_accumulation": report.gradient_accumulation_steps,
                "gradient_checkpointing": report.gradient_checkpointing,
                "learning_rate": report.learning_rate,
                "min_learning_rate": report.min_learning_rate,
                "max_grad_norm": report.max_grad_norm,
                "router_learning_rate": report.router_learning_rate,
                "router_lr_warmup_steps": report.router_lr_warmup_steps,
                "router_lr_anneal_steps": report.router_lr_anneal_steps,
                "router_min_learning_rate": report.router_min_learning_rate,
                "weight_decay": report.weight_decay,
                "ind_route_weight": report.ind_route_weight,
                "ind_route_logit_bias": report.ind_route_logit_bias,
                "ood_route_weight": report.ood_route_weight,
                "ood_route_logit_bias": report.ood_route_logit_bias,
                "route_logit_bias_anneal_steps": report.route_logit_bias_anneal_steps,
                "route_logit_bias_anneal_offset_steps": report.route_logit_bias_anneal_offset_steps,
                "route_logit_bias_anneal_loss_threshold": report.route_logit_bias_anneal_loss_threshold,
                "train_shared": report.train_shared,
                "train_expert_0": report.train_expert_0,
                "train_expert_1": report.train_expert_1,
                "train_embedding_lm_head": report.train_embedding_lm_head,
                "train_full_embedding_lm_head": report.train_full_embedding_lm_head,
                "distill_kl_vocab_chunk_size": report.distill_kl_vocab_chunk_size,
                "checkpoint_every": report.checkpoint_every,
                "checkpoint_dir": report.checkpoint_dir,
                "masked_row_parameter_names": list(report.masked_row_parameter_names),
                "added_token_ids": list(report.added_token_ids),
                "requested_added_token_ids": list(report.requested_added_token_ids),
                "intermediate_added_token_ids": list(report.intermediate_added_token_ids),
                "optimizer_groups": list(report.optimizer_groups),
                "trainable_parameter_names": list(report.trainable_parameter_names),
            }
        },
        ensure_ascii=False,
    )


def _echo_sanitize_run_details(
    *,
    saved_path: Path,
    order_name: str,
    reassign: bool,
    strip_multimodal: bool,
) -> None:
    tokenizer_data = _load_json_object(saved_path / "tokenizer.json")
    vocab = tokenizer_data.get("model", {}).get("vocab", {})
    merges = tokenizer_data.get("model", {}).get("merges", [])
    typer.echo(
        (
            "Sanitize run: "
            f"order={order_name}, "
            f"reassign={_format_bool(reassign)}, "
            f"strip_multimodal={_format_bool(strip_multimodal)}"
        ),
        err=True,
    )
    if isinstance(vocab, dict) and isinstance(merges, list):
        typer.echo(
            f"Tokenizer: vocab_size={len(vocab)}, merge_count={len(merges)}",
            err=True,
        )
    tokenizer_mapping_path = saved_path / "tokenizer_mapping.json"
    token_group_path = saved_path / "tokenizer_token_groups.json"
    initializer_path = saved_path / "tokenizer_added_token_initializers.json"
    typer.echo(
        (
            "Artifacts: "
            f"mapping={_format_bool(tokenizer_mapping_path.exists())}, "
            f"token_groups={_format_bool(token_group_path.exists())}, "
            f"added_initializers={_format_bool(initializer_path.exists())}"
        ),
        err=True,
    )


def _echo_upcycle_summary(
    *,
    upcycled_path: Path,
    source: str,
    expert_init: ExpertInit,
    verbose: bool,
) -> None:
    config = _load_json_object(upcycled_path / "config.json")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        return
    typer.echo(
        (
            "MoE config: "
            f"experts={text_config.get('num_experts')}, "
            f"top_k={text_config.get('top_k_experts')}, "
            f"expert_intermediate_size={text_config.get('expert_intermediate_size')}, "
            f"init={expert_init}"
        ),
        err=True,
    )
    if not verbose:
        return
    checkpoint_path = _find_written_checkpoint(upcycled_path)
    plan_path = upcycled_path / "model_upcycle_plan.yaml"
    typer.echo(f"Upcycle source: {source}", err=True)
    typer.echo(
        (
            "Artifacts: "
            f"plan={plan_path.name if plan_path.exists() else 'missing'}, "
            f"checkpoint={checkpoint_path.name if checkpoint_path is not None else 'missing'}"
        ),
        err=True,
    )


def _find_written_checkpoint(path: Path) -> Path | None:
    for candidate_name in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"):
        candidate = path / candidate_name
        if candidate.exists():
            return candidate
    return None


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


__all__ = [
    "app",
    "compare_command",
    "download_command",
    "logit_diff_command",
    "main",
    "sanitize_command",
    "upcycle_command",
]


if __name__ == "__main__":
    main()
