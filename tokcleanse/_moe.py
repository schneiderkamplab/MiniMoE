"""Helpers for loading and configuring MoE-capable causal language models."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _configure_moe_implementation_for_device(*, model: Any, device: str) -> tuple[str, ...]:
    """Switch MoE implementation to an MPS-safe backend when needed."""

    if device != "mps" or not hasattr(model, "config"):
        return ()

    changed_configs: list[str] = []
    for config_name, config in _iter_relevant_configs(model.config):
        if getattr(config, "_experts_implementation", None) != "grouped_mm":
            continue
        setattr(config, "_experts_implementation", "eager")
        changed_configs.append(config_name)
    return tuple(changed_configs)


def _load_causal_lm_for_runtime(
    *,
    model_path: str,
    dtype: Any,
    device: str,
) -> Any:
    """Load a causal LM, selecting the OdinMOE implementation when requested by config."""

    from transformers import AutoConfig, AutoModelForCausalLM

    from .odin_moe_config import OdinMoETextConfig

    raw_config = _load_model_config_json(model_path)
    if raw_config is not None and raw_config.get("model_type") == "OdinMoE":
        from .odin_moe import OdinMOEForCausalLM

        config = OdinMoETextConfig.from_dict(raw_config)
        model = OdinMOEForCausalLM.from_pretrained(
            model_path,
            config=config,
            dtype=dtype,
        )
    else:
        if raw_config is None:
            AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    _configure_moe_implementation_for_device(model=model, device=device)
    return model


def _load_tokenizer_for_runtime(model_path: str) -> Any:
    """Load a tokenizer, using OdinMoE tokenizer classes for local OdinMoE checkpoints."""

    from transformers import AutoTokenizer

    from .odin_moe_tokenizer import (
        OdinMoETokenizer,
        OdinMoETokenizerFast,
        register_odin_moe_tokenizer_classes,
    )

    raw_config = _load_model_config_json(model_path)
    if raw_config is not None and raw_config.get("model_type") == "OdinMoE":
        register_odin_moe_tokenizer_classes()
        tokenizer_config = _load_tokenizer_config_json(model_path)
        prefer_fast = (Path(model_path).expanduser() / "tokenizer.json").exists()
        if isinstance(tokenizer_config, dict):
            tokenizer_class = tokenizer_config.get("tokenizer_class")
            if isinstance(tokenizer_class, str) and "Fast" in tokenizer_class:
                prefer_fast = True
        if prefer_fast:
            try:
                return OdinMoETokenizerFast.from_pretrained(model_path, fix_mistral_regex=True)
            except Exception:
                pass
        return OdinMoETokenizer.from_pretrained(model_path, fix_mistral_regex=True)
    return AutoTokenizer.from_pretrained(model_path)


def _iter_relevant_configs(config: Any) -> Iterator[tuple[str, Any]]:
    yield "config", config

    seen_ids = {id(config)}
    sub_config_names: list[str] = []
    raw_sub_configs = getattr(config, "sub_configs", None)
    if isinstance(raw_sub_configs, dict):
        sub_config_names.extend(str(name) for name in raw_sub_configs)
    sub_config_names.extend(name for name in ("text_config", "language_config") if name not in sub_config_names)
    for name in sub_config_names:
        child = getattr(config, name, None)
        if child is None or id(child) in seen_ids:
            continue
        seen_ids.add(id(child))
        yield name, child


def _load_model_config_json(model_path: str) -> dict[str, Any] | None:
    config_path = Path(model_path).expanduser() / "config.json"
    if not config_path.exists():
        return None
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return data


def _load_tokenizer_config_json(model_path: str) -> dict[str, Any] | None:
    config_path = Path(model_path).expanduser() / "tokenizer_config.json"
    if not config_path.exists():
        return None
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return data


__all__ = []
