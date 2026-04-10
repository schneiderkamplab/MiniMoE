"""Checkpoint rewriting helpers for token-id reassignment."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

GEMMA4_DEFAULT_EMBEDDING_WEIGHT_NAMES = (
    "model.language_model.embed_tokens.weight",
    "model.language_model.embed_tokens_per_layer.weight",
)
GEMMA4_DEFAULT_LM_HEAD_WEIGHT_NAMES: tuple[str, ...] = ()
_MODEL_PLAN_FILENAME = "model_reassignment_plan.yaml"


def rewrite_reassigned_model(
    source_dir: Path,
    destination_dir: Path,
    *,
    embedding_weight_names: Sequence[str],
    lm_head_weight_names: Sequence[str],
) -> Path | None:
    """Rewrite copied checkpoint weights to match tokenizer id reassignment."""

    tensor_names = (*embedding_weight_names, *lm_head_weight_names)
    if not tensor_names:
        return None

    checkpoint_input_path = _discover_checkpoint_path(source_dir)
    if checkpoint_input_path is None:
        return None

    mapping_path = destination_dir / "tokenizer_mapping.json"
    if not mapping_path.exists():
        raise ValueError(f"Missing tokenizer mapping file: {mapping_path}")
    final_vocab_size = _load_final_vocab_size(destination_dir, mapping_path)

    output_checkpoint_path = _destination_checkpoint_path(destination_dir, checkpoint_input_path)
    plan_path = destination_dir / _MODEL_PLAN_FILENAME
    plan_path.write_text(
        _render_brainsurgery_plan(
            checkpoint_input_path=checkpoint_input_path,
            output_checkpoint_path=output_checkpoint_path,
            mapping_path=mapping_path,
            final_vocab_size=final_vocab_size,
            embedding_weight_names=embedding_weight_names,
            lm_head_weight_names=lm_head_weight_names,
        ),
        encoding="utf-8",
    )
    _run_brainsurgery_plan(plan_path)
    return output_checkpoint_path


def _discover_checkpoint_path(source_dir: Path) -> Path | None:
    if (source_dir / "model.safetensors").exists():
        return source_dir / "model.safetensors"
    if (source_dir / "model.safetensors.index.json").exists():
        return source_dir

    shard_paths = sorted(source_dir.glob("*.safetensors"))
    if shard_paths:
        return source_dir if len(shard_paths) > 1 else shard_paths[0]
    return None


def _destination_checkpoint_path(destination_dir: Path, checkpoint_input_path: Path) -> Path:
    if checkpoint_input_path.is_dir():
        return destination_dir
    return destination_dir / checkpoint_input_path.name


def _render_brainsurgery_plan(
    *,
    checkpoint_input_path: Path,
    output_checkpoint_path: Path,
    mapping_path: Path,
    final_vocab_size: int,
    embedding_weight_names: Sequence[str],
    lm_head_weight_names: Sequence[str],
) -> str:
    transforms = [
        *_render_reindex_transforms(
            tensor_names=embedding_weight_names,
            mapping_path=mapping_path,
            output_size=final_vocab_size,
        ),
        *_render_reindex_transforms(
            tensor_names=lm_head_weight_names,
            mapping_path=mapping_path,
            output_size=final_vocab_size,
        ),
    ]
    lines = [
        "inputs:",
        f"  - {json.dumps(f'model::{checkpoint_input_path.resolve()}')}",
        "transforms:",
        *transforms,
        "  - save:",
        f"      path: {json.dumps(str(output_checkpoint_path.resolve()))}",
        '      alias: "model"',
    ]
    return "\n".join(lines) + "\n"


def _render_reindex_transforms(
    *,
    tensor_names: Sequence[str],
    mapping_path: Path,
    output_size: int,
) -> list[str]:
    lines: list[str] = []
    for tensor_name in tensor_names:
        lines.extend(
            [
                "  - reindex_token_ids:",
                f"      target: {json.dumps(f'model::{tensor_name}')}",
                f"      mapping: {json.dumps(str(mapping_path.resolve()))}",
                f"      output_size: {output_size}",
                "      keep_unmapped: false",
            ]
        )
    return lines


def _load_final_vocab_size(destination_dir: Path, mapping_path: Path) -> int:
    tokenizer_json_path = destination_dir / "tokenizer.json"
    if tokenizer_json_path.exists():
        tokenizer_data = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
        model = tokenizer_data.get("model")
        if isinstance(model, dict):
            vocab = model.get("vocab")
            if isinstance(vocab, dict):
                return len(vocab)

    mapping_data = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping_data, dict):
        raise ValueError(f"Tokenizer mapping must be a JSON object: {mapping_path}")
    return len(mapping_data)


def _run_brainsurgery_plan(plan_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "tokcleanse._brainsurgery_runner",
        "--no-summarize",
        str(plan_path),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return

    output = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part.strip()
    )
    raise RuntimeError(f"brainsurgery plan failed for {plan_path}:\n{output}")


__all__ = [
    "GEMMA4_DEFAULT_EMBEDDING_WEIGHT_NAMES",
    "GEMMA4_DEFAULT_LM_HEAD_WEIGHT_NAMES",
    "rewrite_reassigned_model",
]
