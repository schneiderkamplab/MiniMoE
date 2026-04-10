"""Command-line interface for tokcleanse."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .cleaner import DEFAULT_SAVE_ORDER_NAME, available_order_names, describe_order
from .compare import (
    comparison_has_any_mismatch,
    CompareDType,
    DEFAULT_COMPARE_PROMPT,
    DEFAULT_GENERATION_BATCH_SIZE,
    DEFAULT_SEMANTIC_MODEL_NAME,
    DEFAULT_SEMANTIC_THRESHOLD,
    compare_models,
    load_prompts,
    resolve_compare_device,
    serialize_prompt_comparisons,
    serialize_summary,
)
from .downloader import download_model_snapshot
from .loader import load_tokenizer_contents
from .model_surgery import (
    GEMMA4_DEFAULT_EMBEDDING_WEIGHT_NAMES,
    GEMMA4_DEFAULT_LM_HEAD_WEIGHT_NAMES,
)
from .saver import save_reordered_tokenizer

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
    reassign: bool = typer.Option(
        False,
        "--reassign",
        help="Reassign token ids so specials come first, then base tokens, then merged tokens.",
    ),
    special_token_map_file: Path | None = typer.Option(
        None,
        "--special-token-map-file",
        help=(
            "JSON file mapping old special-token literals to null (keep the literal) or a new "
            "literal string. Special tokens omitted from the file are removed before reassignment."
        ),
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
        raise typer.BadParameter("--special-token-map-file requires --reassign")
    contents = load_tokenizer_contents(source, models_dir=models_dir)
    special_token_literal_map = None
    if special_token_map_file is not None:
        special_token_literal_map = _load_special_token_literal_map_file(special_token_map_file)
    saved_path = save_reordered_tokenizer(
        contents,
        destination,
        order_name=order_name,
        seed=seed,
        overwrite=overwrite,
        reassign=reassign,
        special_token_literal_map=special_token_literal_map,
        embedding_weight_names=tuple(embedding_weight_names),
        lm_head_weight_names=tuple(lm_head_weight_names),
    )
    typer.echo(str(saved_path))


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

    mapping: dict[str, str | None] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise typer.BadParameter(f"{path} must map string literals to null or strings")
        if value is not None and not isinstance(value, str):
            raise typer.BadParameter(f"{path} must map string literals to null or strings")
        mapping[key] = value
    return mapping


__all__ = ["app", "compare_command", "download_command", "main", "sanitize_command"]


if __name__ == "__main__":
    main()
