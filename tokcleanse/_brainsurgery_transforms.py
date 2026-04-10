"""Project-local brainsurgery transforms."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from brainsurgery.core import (
    StateDictProvider,
    TransformError,
    UnarySpec,
    UnaryTransform,
    must_model,
    register_transform,
    require_nonempty_string,
)


class ReindexTokenIdsTransformError(TransformError):
    """Raised when token-id reindexing cannot be applied."""


@dataclass(frozen=True)
class ReindexTokenIdsSpec(UnarySpec):
    """Specification for token-id based tensor reindexing."""

    mapping_path: Path
    dim: int | None
    keep_unmapped: bool
    output_size: int | None


class ReindexTokenIdsTransform(UnaryTransform[ReindexTokenIdsSpec]):
    """Reindex tensor rows or columns using a tokenizer id mapping."""

    name = "reindex_token_ids"
    error_type = ReindexTokenIdsTransformError
    spec_type = ReindexTokenIdsSpec
    allowed_keys = {"target", "mapping", "dim", "keep_unmapped", "output_size"}
    required_keys = {"target", "mapping"}
    help_text = (
        "Reindexes a token-dependent tensor using tokenizer_mapping.json.\n"
        "\n"
        "The mapping file must map original token ids to new token ids. The transform\n"
        "builds the inverse order for mapped ids. By default, unmapped tensor rows or\n"
        "columns are left at the tail in their original order, but you can drop them by\n"
        "setting keep_unmapped: false. When output_size is provided, the transform writes\n"
        "mapped ids into a zero-initialized output axis of that size, which supports added\n"
        "tokens that have no source checkpoint row.\n"
        "\n"
        "Examples:\n"
        "  reindex_token_ids: { target: model::model.embed_tokens.weight, mapping: /tmp/tokenizer_mapping.json }\n"
        "  reindex_token_ids: { target: model::lm_head.weight, mapping: /tmp/tokenizer_mapping.json, dim: -1 }\n"
        "  reindex_token_ids: { target: model::model.embed_tokens.weight, mapping: /tmp/tokenizer_mapping.json, keep_unmapped: false }"
    )

    def build_spec(self, target_ref: Any, payload: dict[str, Any]) -> ReindexTokenIdsSpec:
        mapping_path = Path(require_nonempty_string(payload, op_name=self.name, key="mapping"))
        raw_dim = payload.get("dim")
        if raw_dim is None or raw_dim == "auto":
            dim: int | None = None
        elif isinstance(raw_dim, int):
            dim = raw_dim
        else:
            raise ReindexTokenIdsTransformError(
                "reindex_token_ids.dim must be an integer or 'auto'"
            )
        raw_keep_unmapped = payload.get("keep_unmapped", True)
        if not isinstance(raw_keep_unmapped, bool):
            raise ReindexTokenIdsTransformError(
                "reindex_token_ids.keep_unmapped must be a boolean"
            )
        raw_output_size = payload.get("output_size")
        if raw_output_size is None:
            output_size: int | None = None
        elif isinstance(raw_output_size, int) and raw_output_size >= 0:
            output_size = raw_output_size
        else:
            raise ReindexTokenIdsTransformError(
                "reindex_token_ids.output_size must be a non-negative integer"
            )
        return ReindexTokenIdsSpec(
            target_ref=target_ref,
            mapping_path=mapping_path,
            dim=dim,
            keep_unmapped=raw_keep_unmapped,
            output_size=output_size,
        )

    def apply_to_target(
        self,
        spec: ReindexTokenIdsSpec,
        name: str,
        provider: StateDictProvider,
    ) -> None:
        model_alias = must_model(spec.target_ref)
        state_dict = provider.get_state_dict(model_alias)
        tensor = state_dict[name]
        mapping = _load_tokenizer_mapping(spec.mapping_path)
        dim = _resolve_reindex_dim(spec.dim, tensor, mapping, name)
        if spec.output_size is None:
            index = _build_reindex_tensor(
                mapping,
                size=int(tensor.shape[dim]),
                device=tensor.device,
                keep_unmapped=spec.keep_unmapped,
            )
            state_dict[name] = tensor.index_select(dim, index).clone()
            return
        if spec.keep_unmapped:
            raise ReindexTokenIdsTransformError(
                "reindex_token_ids.output_size requires keep_unmapped: false"
            )
        state_dict[name] = _remap_tensor_with_output_size(
            tensor,
            mapping,
            dim=dim,
            output_size=spec.output_size,
        )


def _resolve_reindex_dim(
    configured_dim: int | None,
    tensor: torch.Tensor,
    mapping: dict[int, int],
    tensor_name: str,
) -> int:
    if configured_dim is not None:
        dim = configured_dim if configured_dim >= 0 else configured_dim + tensor.dim()
        if dim < 0 or dim >= tensor.dim():
            raise ReindexTokenIdsTransformError(
                f"reindex_token_ids.dim {configured_dim} out of range for {tensor_name}"
            )
        return dim

    max_index = max(mapping.keys(), default=-1)
    candidate_dims = [
        dim
        for dim in {0, tensor.dim() - 1}
        if int(tensor.shape[dim]) > max_index
    ]
    if len(candidate_dims) == 1:
        return candidate_dims[0]
    if not candidate_dims:
        raise ReindexTokenIdsTransformError(
            f"Could not infer reindex dimension for {tensor_name} with shape {tuple(tensor.shape)}"
        )
    raise ReindexTokenIdsTransformError(
        f"Ambiguous reindex dimension for {tensor_name}; please pass dim explicitly"
    )


def _build_reindex_tensor(
    mapping: dict[int, int],
    *,
    size: int,
    device: torch.device,
    keep_unmapped: bool,
) -> torch.Tensor:
    mapped_size = len(mapping)
    if mapped_size > size:
        raise ReindexTokenIdsTransformError(
            f"Mapping covers {mapped_size} token ids, but tensor axis has size {size}"
        )

    old_ids_by_new: list[int | None] = [None] * mapped_size
    for old_id, new_id in mapping.items():
        if old_id < 0 or new_id < 0:
            raise ReindexTokenIdsTransformError("Tokenizer mappings must be non-negative")
        if old_id >= size:
            raise ReindexTokenIdsTransformError(
                f"Mapping references original id {old_id}, but tensor axis has size {size}"
            )
        if new_id >= mapped_size:
            raise ReindexTokenIdsTransformError(
                f"Mapping references new id {new_id}, but only {mapped_size} mapped ids exist"
            )
        if old_ids_by_new[new_id] is not None:
            raise ReindexTokenIdsTransformError(f"Duplicate target id in mapping: {new_id}")
        old_ids_by_new[new_id] = old_id

    if any(old_id is None for old_id in old_ids_by_new):
        raise ReindexTokenIdsTransformError("Tokenizer mapping must cover 0..k-1 without gaps")

    mapped_old_ids = set(mapping)
    full_index = [old_id for old_id in old_ids_by_new if old_id is not None]
    if keep_unmapped:
        full_index.extend(old_id for old_id in range(size) if old_id not in mapped_old_ids)
    expected_length = size if keep_unmapped else mapped_size
    if len(full_index) != expected_length:
        raise ReindexTokenIdsTransformError(
            "Constructed reindex order length "
            f"{len(full_index)} does not match expected length {expected_length}"
        )
    return torch.tensor(full_index, dtype=torch.long, device=device)


def _remap_tensor_with_output_size(
    tensor: torch.Tensor,
    mapping: dict[int, int],
    *,
    dim: int,
    output_size: int,
) -> torch.Tensor:
    old_ids = sorted(mapping)
    if not old_ids:
        shape = list(tensor.shape)
        shape[dim] = output_size
        return tensor.new_zeros(shape)

    max_old_id = max(old_ids)
    max_new_id = max(mapping.values())
    axis_size = int(tensor.shape[dim])
    if max_old_id >= axis_size:
        raise ReindexTokenIdsTransformError(
            f"Mapping references original id {max_old_id}, but tensor axis has size {axis_size}"
        )
    if max_new_id >= output_size:
        raise ReindexTokenIdsTransformError(
            f"Mapping references new id {max_new_id}, but output axis has size {output_size}"
        )

    old_index = torch.tensor(old_ids, dtype=torch.long, device=tensor.device)
    new_index = torch.tensor([mapping[old_id] for old_id in old_ids], dtype=torch.long, device=tensor.device)
    remapped = tensor.new_zeros([*tensor.shape[:dim], output_size, *tensor.shape[dim + 1 :]])
    source = tensor.index_select(dim, old_index)
    remapped.index_copy_(dim, new_index, source)
    return remapped


@lru_cache(maxsize=None)
def _load_tokenizer_mapping(mapping_path: Path) -> dict[int, int]:
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ReindexTokenIdsTransformError(
            f"Tokenizer mapping must be a JSON object: {mapping_path}"
        )

    mapping: dict[int, int] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key.isdigit() or not isinstance(value, int):
            raise ReindexTokenIdsTransformError(
                f"Tokenizer mapping must map stringified ints to ints: {mapping_path}"
            )
        mapping[int(key)] = value
    return mapping


register_transform(ReindexTokenIdsTransform())
