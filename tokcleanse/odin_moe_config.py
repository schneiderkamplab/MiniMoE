"""Configuration class for text-only OdinMoE models."""

from __future__ import annotations

from typing import Any

from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig


class OdinMoETextConfig(Gemma4TextConfig):
    """Gemma 4 text config with OdinMoE model identity."""

    model_type = "OdinMoE"

    def __post_init__(self, **kwargs: Any) -> None:
        super().__post_init__(**kwargs)
        if hasattr(self, "enable_moe_block"):
            delattr(self, "enable_moe_block")
        self.model_type = type(self).model_type

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.pop("enable_moe_block", None)
        payload["model_type"] = type(self).model_type
        return payload


__all__ = ["OdinMoETextConfig"]
