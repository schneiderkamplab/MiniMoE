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
    TransformResult,
    TransformError,
    TypedTransform,
    UnarySpec,
    UnaryTransform,
    must_model,
    register_transform,
    require_nonempty_string,
    ensure_mapping_payload,
    validate_payload_keys,
)


class ReindexTokenIdsTransformError(TransformError):
    """Raised when token-id reindexing cannot be applied."""


class UpcycleGemma4DenseToMoeTransformError(TransformError):
    """Raised when Gemma 4 dense-to-MoE upcycling cannot be applied."""


@dataclass(frozen=True)
class ReindexTokenIdsSpec(UnarySpec):
    """Specification for token-id based tensor reindexing."""

    mapping_path: Path
    initializers_path: Path | None
    dim: int | None
    keep_unmapped: bool
    output_size: int | None


@dataclass(frozen=True)
class UpcycleGemma4DenseToMoeSpec:
    """Specification for Gemma 4 dense-to-MoE tensor creation."""

    alias: str
    num_experts: int
    expert_intermediate_size: int
    init: str

    def collect_models(self) -> set[str]:
        return {self.alias}


class ReindexTokenIdsTransform(UnaryTransform[ReindexTokenIdsSpec]):
    """Reindex tensor rows or columns using a tokenizer id mapping."""

    name = "reindex_token_ids"
    error_type = ReindexTokenIdsTransformError
    spec_type = ReindexTokenIdsSpec
    allowed_keys = {"target", "mapping", "initializers", "dim", "keep_unmapped", "output_size"}
    required_keys = {"target", "mapping"}
    help_text = (
        "Reindexes a token-dependent tensor using tokenizer_mapping.json.\n"
        "\n"
        "The mapping file must map original token ids to new token ids. The transform\n"
        "builds the inverse order for mapped ids. By default, unmapped tensor rows or\n"
        "columns are left at the tail in their original order, but you can drop them by\n"
        "setting keep_unmapped: false. When output_size is provided, the transform writes\n"
        "mapped ids into a zero-initialized output axis of that size. Added tokens can\n"
        "optionally be initialized from existing source ids by passing an initializers\n"
        "JSON file whose values list the source ids to average.\n"
        "\n"
        "Examples:\n"
        "  reindex_token_ids: { target: model::model.embed_tokens.weight, mapping: /tmp/tokenizer_mapping.json }\n"
        "  reindex_token_ids: { target: model::lm_head.weight, mapping: /tmp/tokenizer_mapping.json, dim: -1 }\n"
        "  reindex_token_ids: { target: model::model.embed_tokens.weight, mapping: /tmp/tokenizer_mapping.json, initializers: /tmp/tokenizer_added_token_initializers.json, keep_unmapped: false, output_size: 12345 }\n"
        "  reindex_token_ids: { target: model::model.embed_tokens.weight, mapping: /tmp/tokenizer_mapping.json, keep_unmapped: false }"
    )

    def build_spec(self, target_ref: Any, payload: dict[str, Any]) -> ReindexTokenIdsSpec:
        mapping_path = Path(require_nonempty_string(payload, op_name=self.name, key="mapping"))
        raw_initializers = payload.get("initializers")
        if raw_initializers is None:
            initializers_path: Path | None = None
        elif isinstance(raw_initializers, str) and raw_initializers:
            initializers_path = Path(raw_initializers)
        else:
            raise ReindexTokenIdsTransformError(
                "reindex_token_ids.initializers must be a non-empty string path"
            )
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
            initializers_path=initializers_path,
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
        initializers = (
            _load_tokenizer_initializers(spec.initializers_path)
            if spec.initializers_path is not None
            else {}
        )
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
            initializers=initializers,
            dim=dim,
            output_size=spec.output_size,
        )


class UpcycleGemma4DenseToMoeTransform(TypedTransform[UpcycleGemma4DenseToMoeSpec]):
    """Create Gemma 4 MoE tensors from dense MLP tensors."""

    name = "upcycle_gemma4_dense_to_moe"
    error_type = UpcycleGemma4DenseToMoeTransformError
    spec_type = UpcycleGemma4DenseToMoeSpec
    completion_requires_payload = False
    allowed_keys = {"alias", "num_experts", "expert_intermediate_size", "init"}
    help_text = (
        "Creates Gemma 4 router/expert tensors for each dense MLP gate projection in an alias.\n"
        "\n"
        "The transform scans an alias for Gemma 4 dense gate_proj weights such as\n"
        "model.layers.N.mlp.gate_proj.weight or model.language_model.layers.N.mlp.gate_proj.weight.\n"
        "For each matched layer it reads gate/up/down dense MLP weights,\n"
        "creates experts.gate_up_proj and experts.down_proj, and initializes the router\n"
        "for a pure routed-expert replacement of the dense MLP.\n"
        "\n"
        "Examples:\n"
        '  upcycle_gemma4_dense_to_moe: { alias: model, num_experts: 2, expert_intermediate_size: 12288 }\n'
        '  upcycle_gemma4_dense_to_moe: { alias: model, num_experts: 2, expert_intermediate_size: 12288, init: "copy" }\n'
        '  upcycle_gemma4_dense_to_moe: { alias: model, num_experts: 2, expert_intermediate_size: 12288, init: "zero" }'
    )

    def compile(self, payload: Any, default_model: str | None) -> UpcycleGemma4DenseToMoeSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
        )
        alias = payload.get("alias")
        if alias is None:
            if default_model is None:
                raise UpcycleGemma4DenseToMoeTransformError(
                    "upcycle_gemma4_dense_to_moe.alias is required when no default model is available"
                )
            resolved_alias = default_model
        elif isinstance(alias, str) and alias:
            resolved_alias = alias
        else:
            raise UpcycleGemma4DenseToMoeTransformError(
                "upcycle_gemma4_dense_to_moe.alias must be a non-empty string"
            )
        raw_num_experts = payload.get("num_experts")
        if not isinstance(raw_num_experts, int) or raw_num_experts < 1:
            raise UpcycleGemma4DenseToMoeTransformError(
                "upcycle_gemma4_dense_to_moe.num_experts must be a positive integer"
            )
        raw_expert_intermediate_size = payload.get("expert_intermediate_size")
        if not isinstance(raw_expert_intermediate_size, int) or raw_expert_intermediate_size < 1:
            raise UpcycleGemma4DenseToMoeTransformError(
                "upcycle_gemma4_dense_to_moe.expert_intermediate_size must be a positive integer"
            )
        raw_init = payload.get("init", "copy")
        if raw_init not in {"copy", "zero"}:
            raise UpcycleGemma4DenseToMoeTransformError(
                "upcycle_gemma4_dense_to_moe.init must be 'copy' or 'zero'"
            )
        return UpcycleGemma4DenseToMoeSpec(
            alias=resolved_alias,
            num_experts=raw_num_experts,
            expert_intermediate_size=raw_expert_intermediate_size,
            init=raw_init,
        )

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        state_dict = provider.get_state_dict(typed.alias)
        matched_names = sorted(
            name
            for name in state_dict
            if (
                name.endswith(".mlp.gate_proj.weight")
                and (
                    name.startswith("model.layers.")
                    or name.startswith("model.language_model.layers.")
                )
            )
        )
        if not matched_names:
            raise UpcycleGemma4DenseToMoeTransformError(
                f"upcycle_gemma4_dense_to_moe found no dense Gemma 4 MLP gate tensors in alias {typed.alias!r}"
            )
        for name in matched_names:
            self._apply_one_layer(typed, name, state_dict)
        return TransformResult(name=self.name, count=len(matched_names))

    def _infer_output_model(self, spec: object) -> str:
        return self.require_spec(spec).alias

    def _apply_one_layer(
        self,
        spec: UpcycleGemma4DenseToMoeSpec,
        name: str,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        suffix = "mlp.gate_proj.weight"
        if not name.endswith(suffix):
            raise UpcycleGemma4DenseToMoeTransformError(
                f"Unsupported dense MLP tensor for upcycling: {name}"
            )
        prefix = name[: -len(suffix)]
        gate_name = name
        up_name = f"{prefix}mlp.up_proj.weight"
        down_name = f"{prefix}mlp.down_proj.weight"
        required_names = (up_name, down_name)
        missing = [tensor_name for tensor_name in required_names if tensor_name not in state_dict]
        if missing:
            raise UpcycleGemma4DenseToMoeTransformError(
                "Missing sibling feed-forward tensors required for upcycling: "
                + ", ".join(missing)
            )

        gate = state_dict[gate_name]
        up = state_dict[up_name]
        down = state_dict[down_name]
        if gate.dim() != 2 or up.dim() != 2 or down.dim() != 2:
            raise UpcycleGemma4DenseToMoeTransformError(
                f"Dense MLP tensors must be rank-2 for {prefix.rstrip('.')}"
            )
        if gate.shape != up.shape:
            raise UpcycleGemma4DenseToMoeTransformError(
                f"gate_proj and up_proj shapes differ for {prefix.rstrip('.')}: "
                f"{tuple(gate.shape)} vs {tuple(up.shape)}"
            )
        hidden_dim = int(gate.shape[1])
        source_intermediate_dim = int(gate.shape[0])
        target_intermediate_dim = spec.expert_intermediate_size
        if tuple(down.shape) != (hidden_dim, source_intermediate_dim):
            raise UpcycleGemma4DenseToMoeTransformError(
                f"Unexpected down_proj shape for {prefix.rstrip('.')}: got {tuple(down.shape)}, "
                f"expected {(hidden_dim, source_intermediate_dim)}"
            )
        if source_intermediate_dim > target_intermediate_dim:
            raise UpcycleGemma4DenseToMoeTransformError(
                f"Dense MLP width {source_intermediate_dim} exceeds requested expert_intermediate_size "
                f"{target_intermediate_dim} for {prefix.rstrip('.')}"
            )

        gate_up_name = f"{prefix}experts.gate_up_proj"
        experts_down_name = f"{prefix}experts.down_proj"
        router_proj_name = f"{prefix}router.proj.weight"
        router_scale_name = f"{prefix}router.scale"
        per_expert_scale_name = f"{prefix}router.per_expert_scale"
        created_names = (
            gate_up_name,
            experts_down_name,
            router_proj_name,
            router_scale_name,
            per_expert_scale_name,
        )
        collisions = [tensor_name for tensor_name in created_names if tensor_name in state_dict]
        if collisions:
            raise UpcycleGemma4DenseToMoeTransformError(
                "Refusing to overwrite existing MoE tensors: " + ", ".join(collisions)
            )

        if spec.init == "copy":
            gate_padded = _pad_rows(gate, target_intermediate_dim)
            up_padded = _pad_rows(up, target_intermediate_dim)
            down_padded = _pad_columns(down, target_intermediate_dim)
            gate_up_base = torch.cat((gate_padded, up_padded), dim=0)
            gate_up = gate_up_base.unsqueeze(0).repeat(spec.num_experts, 1, 1).clone()
            experts_down = down_padded.unsqueeze(0).repeat(spec.num_experts, 1, 1).clone()
        else:
            gate_up = gate.new_zeros((spec.num_experts, 2 * target_intermediate_dim, hidden_dim))
            experts_down = down.new_zeros((spec.num_experts, hidden_dim, target_intermediate_dim))

        state_dict[gate_up_name] = gate_up
        state_dict[experts_down_name] = experts_down
        state_dict[router_proj_name] = gate.new_zeros((spec.num_experts, hidden_dim))
        state_dict[router_scale_name] = gate.new_ones((hidden_dim,))
        state_dict[per_expert_scale_name] = gate.new_ones((spec.num_experts,))
        del state_dict[gate_name]
        del state_dict[up_name]
        del state_dict[down_name]


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
    initializers: dict[int, tuple[int, ...]],
    dim: int,
    output_size: int,
) -> torch.Tensor:
    old_ids = sorted(mapping)
    if not old_ids:
        shape = list(tensor.shape)
        shape[dim] = output_size
        remapped = tensor.new_zeros(shape)
        _apply_added_token_initializers(
            remapped,
            tensor=tensor,
            initializers=initializers,
            dim=dim,
            output_size=output_size,
        )
        return remapped

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
    _apply_added_token_initializers(
        remapped,
        tensor=tensor,
        initializers=initializers,
        dim=dim,
        output_size=output_size,
    )
    return remapped


def _apply_added_token_initializers(
    remapped: torch.Tensor,
    *,
    tensor: torch.Tensor,
    initializers: dict[int, tuple[int, ...]],
    dim: int,
    output_size: int,
) -> None:
    if not initializers:
        return

    axis_size = int(tensor.shape[dim])
    for new_id, old_ids in sorted(initializers.items()):
        if new_id < 0 or new_id >= output_size:
            raise ReindexTokenIdsTransformError(
                f"Initializer references new id {new_id}, but output axis has size {output_size}"
            )
        if not old_ids:
            raise ReindexTokenIdsTransformError(
                f"Initializer for new id {new_id} must reference at least one source id"
            )
        if any(old_id < 0 or old_id >= axis_size for old_id in old_ids):
            raise ReindexTokenIdsTransformError(
                f"Initializer for new id {new_id} references a source id outside axis size {axis_size}"
            )
        old_index = torch.tensor(old_ids, dtype=torch.long, device=tensor.device)
        source = tensor.index_select(dim, old_index).mean(dim=dim, keepdim=True)
        new_index = torch.tensor([new_id], dtype=torch.long, device=tensor.device)
        remapped.index_copy_(dim, new_index, source)


def _pad_rows(tensor: torch.Tensor, target_rows: int) -> torch.Tensor:
    current_rows = int(tensor.shape[0])
    if current_rows == target_rows:
        return tensor.clone()
    if current_rows > target_rows:
        raise UpcycleGemma4DenseToMoeTransformError(
            f"Cannot pad rows from {current_rows} down to smaller target {target_rows}"
        )
    padded = tensor.new_zeros((target_rows, *tensor.shape[1:]))
    padded[:current_rows] = tensor
    return padded


def _pad_columns(tensor: torch.Tensor, target_columns: int) -> torch.Tensor:
    current_columns = int(tensor.shape[1])
    if current_columns == target_columns:
        return tensor.clone()
    if current_columns > target_columns:
        raise UpcycleGemma4DenseToMoeTransformError(
            f"Cannot pad columns from {current_columns} down to smaller target {target_columns}"
        )
    padded = tensor.new_zeros((tensor.shape[0], target_columns))
    padded[:, :current_columns] = tensor
    return padded


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


@lru_cache(maxsize=None)
def _load_tokenizer_initializers(initializers_path: Path) -> dict[int, tuple[int, ...]]:
    data = json.loads(initializers_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ReindexTokenIdsTransformError(
            f"Tokenizer initializers must be a JSON object: {initializers_path}"
        )

    initializers: dict[int, tuple[int, ...]] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key.isdigit():
            raise ReindexTokenIdsTransformError(
                "Tokenizer initializers must use stringified integer keys: "
                f"{initializers_path}"
            )
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, int) for item in value)
        ):
            raise ReindexTokenIdsTransformError(
                "Tokenizer initializers must map each key to a non-empty list of ints: "
                f"{initializers_path}"
            )
        initializers[int(key)] = tuple(value)
    return initializers


register_transform(ReindexTokenIdsTransform())
register_transform(UpcycleGemma4DenseToMoeTransform())
