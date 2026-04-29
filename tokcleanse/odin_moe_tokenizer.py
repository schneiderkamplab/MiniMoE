"""Tokenizer classes for OdinMoE checkpoints."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoTokenizer, GemmaTokenizer, GemmaTokenizerFast

from .odin_moe_config import OdinMoETextConfig


class OdinMoETokenizer(GemmaTokenizer):
    """Gemma tokenizer with OdinMoE identity."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("fix_mistral_regex", True)
        super().__init__(*args, **kwargs)


class OdinMoETokenizerFast(GemmaTokenizerFast):
    """Fast Gemma tokenizer with OdinMoE identity."""

    slow_tokenizer_class = OdinMoETokenizer

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("fix_mistral_regex", True)
        super().__init__(*args, **kwargs)


_REGISTERED_WITH_TRANSFORMERS = False


def register_odin_moe_tokenizer_classes() -> None:
    """Register OdinMoE config/tokenizer classes with Transformers auto loaders."""

    global _REGISTERED_WITH_TRANSFORMERS
    if _REGISTERED_WITH_TRANSFORMERS:
        return
    AutoConfig.register("OdinMoE", OdinMoETextConfig, exist_ok=True)
    AutoTokenizer.register(
        OdinMoETextConfig,
        slow_tokenizer_class=OdinMoETokenizer,
        fast_tokenizer_class=OdinMoETokenizerFast,
        exist_ok=True,
    )
    _REGISTERED_WITH_TRANSFORMERS = True


def tokenizer_class_name_for_checkpoint(*, checkpoint_dir: str | Any) -> str:
    """Return the tokenizer class name that should be written for a checkpoint."""

    from pathlib import Path

    path = Path(checkpoint_dir).expanduser()
    if (path / "tokenizer.json").exists():
        return "OdinMoETokenizerFast"
    return "OdinMoETokenizer"


__all__ = [
    "OdinMoETokenizer",
    "OdinMoETokenizerFast",
    "register_odin_moe_tokenizer_classes",
    "tokenizer_class_name_for_checkpoint",
]
