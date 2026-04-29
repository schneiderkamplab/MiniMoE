"""Gemma 4 dense-to-MoE upcycling helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from .downloader import download_model_snapshot
from .model_surgery import _render_delete_transforms, _render_move_transforms
from .model_surgery import (
    _destination_checkpoint_path,
    _discover_checkpoint_path,
    _run_brainsurgery_plan,
)
from .odin_moe_tokenizer import tokenizer_class_name_for_checkpoint

ExpertInit = Literal["copy", "zero"]

_UPCYCLE_PLAN_FILENAME = "model_upcycle_plan.yaml"


def upcycle_gemma4_model(
    source: str | Path,
    destination: Path,
    *,
    models_dir: Path = Path("models"),
    overwrite: bool = False,
    num_experts: int = 2,
    top_k_experts: int = 2,
    expert_init: ExpertInit = "copy",
) -> Path:
    """Copy a Gemma 4 checkpoint directory and add MoE expert tensors."""

    if num_experts < 1:
        raise ValueError("num_experts must be at least 1")
    if top_k_experts < 1:
        raise ValueError("top_k_experts must be at least 1")
    if top_k_experts > num_experts:
        raise ValueError("top_k_experts cannot exceed num_experts")
    if expert_init not in {"copy", "zero"}:
        raise ValueError("expert_init must be 'copy' or 'zero'")

    source_dir = _resolve_model_directory(source, models_dir=models_dir)
    destination_path = Path(destination).expanduser()
    if destination_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {destination_path}")
        shutil.rmtree(destination_path)
    shutil.copytree(source_dir, destination_path)

    config_path = destination_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"Missing config.json in {source_dir}")
    expert_intermediate_size, strip_multimodal = _rewrite_upcycled_config(
        config_path,
        num_experts=num_experts,
        top_k_experts=top_k_experts,
        expert_init=expert_init,
    )
    _rewrite_upcycled_tokenizer_config(
        destination_path / "tokenizer_config.json",
    )
    processor_config_path = destination_path / "processor_config.json"
    if processor_config_path.exists():
        processor_config_path.unlink()

    checkpoint_input_path = _discover_checkpoint_path(source_dir)
    if checkpoint_input_path is None:
        raise ValueError(f"Could not find a checkpoint in {source_dir}")
    output_checkpoint_path = _destination_checkpoint_path(destination_path, checkpoint_input_path)
    plan_path = destination_path / _UPCYCLE_PLAN_FILENAME
    plan_path.write_text(
        _render_upcycle_brainsurgery_plan(
            checkpoint_input_path=checkpoint_input_path,
            output_checkpoint_path=output_checkpoint_path,
            num_experts=num_experts,
            expert_intermediate_size=expert_intermediate_size,
            expert_init=expert_init,
            strip_multimodal=strip_multimodal,
        ),
        encoding="utf-8",
    )
    _run_brainsurgery_plan(plan_path)
    return destination_path.resolve()


def _resolve_model_directory(source: str | Path, *, models_dir: Path) -> Path:
    source_path = Path(source).expanduser()
    if source_path.exists():
        return source_path.resolve()

    candidate = models_dir / str(source)
    if candidate.exists():
        return candidate.resolve()

    downloaded_path = download_model_snapshot(str(source), models_dir=models_dir)
    return downloaded_path.resolve()


def _rewrite_upcycled_config(
    path: Path,
    *,
    num_experts: int,
    top_k_experts: int,
    expert_init: ExpertInit,
) -> tuple[int, bool]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a JSON object")

    text_config = config.get("text_config")
    is_conditional_config = isinstance(text_config, dict)
    if isinstance(text_config, dict):
        target = text_config
    elif config.get("model_type") == "gemma4_text":
        target = config
    else:
        raise ValueError(f"{path} is not a Gemma 4 text or conditional config")

    intermediate_size = target.get("intermediate_size")
    if not isinstance(intermediate_size, int) or intermediate_size <= 0:
        raise ValueError(f"{path} is missing a positive intermediate_size")
    expert_intermediate_size = intermediate_size
    if target.get("use_double_wide_mlp") is True:
        num_kv_shared_layers = target.get("num_kv_shared_layers")
        if isinstance(num_kv_shared_layers, int) and num_kv_shared_layers > 0:
            expert_intermediate_size *= 2

    normalized = dict(target)
    normalized.pop("enable_moe_block", None)
    normalized["num_experts"] = num_experts
    normalized["top_k_experts"] = top_k_experts
    normalized["moe_intermediate_size"] = expert_intermediate_size
    normalized["expert_intermediate_size"] = expert_intermediate_size
    normalized["model_type"] = "OdinMoE"
    normalized["architectures"] = ["OdinMOEForCausalLM"]
    if "_name_or_path" in config and "_name_or_path" not in normalized:
        normalized["_name_or_path"] = config["_name_or_path"]

    path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return expert_intermediate_size, is_conditional_config


def _rewrite_upcycled_tokenizer_config(
    path: Path,
) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    for key in ("audio_token", "image_token", "video_token", "processor_class"):
        data.pop(key, None)
    data["tokenizer_class"] = tokenizer_class_name_for_checkpoint(checkpoint_dir=path.parent)
    data["fix_mistral_regex"] = True
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _render_upcycle_brainsurgery_plan(
    *,
    checkpoint_input_path: Path,
    output_checkpoint_path: Path,
    num_experts: int,
    expert_intermediate_size: int,
    expert_init: ExpertInit,
    strip_multimodal: bool,
) -> str:
    lines = [
        "inputs:",
        f"  - {json.dumps(f'model::{checkpoint_input_path.resolve()}')}",
        "transforms:",
        "  - upcycle_gemma4_dense_to_moe:",
        '      alias: "model"',
        f"      num_experts: {num_experts}",
        f"      expert_intermediate_size: {expert_intermediate_size}",
        f"      init: {json.dumps(expert_init)}",
        *_render_move_transforms(strip_multimodal=strip_multimodal),
        *_render_delete_transforms(strip_multimodal=strip_multimodal),
        "  - save:",
        f"      path: {json.dumps(str(output_checkpoint_path.resolve()))}",
        '      alias: "model"',
    ]
    return "\n".join(lines) + "\n"


__all__ = ["ExpertInit", "upcycle_gemma4_model"]
