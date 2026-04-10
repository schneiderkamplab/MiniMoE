"""Saving helpers for reordered tokenizer directories."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ._jinja import _rewrite_template
from .cleaner import DEFAULT_SAVE_ORDER_NAME, topological_sort_rules
from .loader import MergeRule, TokenizerContents, _load_special_tokens
from .model_surgery import (
    GEMMA4_DEFAULT_EMBEDDING_WEIGHT_NAMES,
    GEMMA4_DEFAULT_LM_HEAD_WEIGHT_NAMES,
    rewrite_reassigned_model,
)


@dataclass(frozen=True, slots=True)
class _ReassignmentPlan:
    token_order: tuple[str, ...]
    mapping: dict[int, int]


@dataclass(frozen=True, slots=True)
class _SpecialTokenRewritePlan:
    replacements: dict[str, str]
    dropped_literals: frozenset[str]
    dropped_ids: frozenset[int]


def save_reordered_tokenizer(
    contents: TokenizerContents,
    destination: str | Path,
    *,
    order_name: str = DEFAULT_SAVE_ORDER_NAME,
    seed: int | None = None,
    overwrite: bool = False,
    reassign: bool = False,
    special_token_literal_map: dict[str, str | None] | None = None,
    embedding_weight_names: tuple[str, ...] = GEMMA4_DEFAULT_EMBEDDING_WEIGHT_NAMES,
    lm_head_weight_names: tuple[str, ...] = GEMMA4_DEFAULT_LM_HEAD_WEIGHT_NAMES,
) -> Path:
    """Copy a tokenizer directory and rewrite its merge order.

    When ``reassign`` is true, special tokens are packed first, then non-merged
    tokens, then merged tokens in the requested topological order. The rewritten
    directory also stores ``tokenizer_mapping.json`` plus copies of the original
    ``tokenizer.json`` and ``tokenizer_config.json`` when present.
    """

    destination_path = Path(destination).expanduser()
    source_path = contents.resolved_dir

    if destination_path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {destination_path}")
        shutil.rmtree(destination_path)

    shutil.copytree(source_path, destination_path)
    special_token_rewrite_plan = None
    if reassign:
        _write_original_tokenizer_copies(destination_path)
    if special_token_literal_map is not None:
        if not reassign:
            raise ValueError("Special-token remapping requires reassign=True")
        special_token_rewrite_plan = _rewrite_special_tokens(
            destination=destination_path,
            contents=contents,
            special_token_literal_map=special_token_literal_map,
        )
    ordered_rules = topological_sort_rules(contents, order_name=order_name, seed=seed)
    merges = cast(list[tuple[str, str]], [(rule.left, rule.right) for rule in ordered_rules])
    _write_reordered_merges(destination_path, merges)
    if reassign:
        plan = _build_reassignment_plan(contents, ordered_rules, destination_path)
        _write_reassigned_tokenizer(destination_path, plan)
        if special_token_rewrite_plan is not None:
            _rewrite_chat_template_files(destination_path, special_token_rewrite_plan)
        _write_tokenizer_mapping(destination_path, plan.mapping)
        rewrite_reassigned_model(
            source_path,
            destination_path,
            embedding_weight_names=embedding_weight_names,
            lm_head_weight_names=lm_head_weight_names,
        )
    return destination_path.resolve()


def _write_original_tokenizer_copies(destination: Path) -> None:
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        path = destination / filename
        if not path.exists():
            continue
        original_copy_path = destination / f"original_{filename}"
        shutil.copy2(path, original_copy_path)


def _write_reordered_merges(
    destination: Path,
    merges: list[tuple[str, str]],
) -> None:
    wrote_any = False
    tokenizer_json_path = destination / "tokenizer.json"
    if tokenizer_json_path.exists():
        _write_tokenizer_json_merges(tokenizer_json_path, merges)
        wrote_any = True

    merges_txt_path = destination / "merges.txt"
    if merges_txt_path.exists():
        _write_merges_txt(merges_txt_path, merges)
        wrote_any = True

    if not wrote_any:
        raise ValueError(f"Could not find tokenizer.json or merges.txt in {destination}")


def _build_reassignment_plan(
    contents: TokenizerContents,
    ordered_rules: list[MergeRule],
    destination: Path,
) -> _ReassignmentPlan:
    tokenizer_json_path = destination / "tokenizer.json"
    if not tokenizer_json_path.exists():
        raise ValueError("Cannot reassign token ids without tokenizer.json")

    tokenizer_data = _load_json_object(tokenizer_json_path)
    token_to_index = _load_token_to_index(tokenizer_data)
    special_tokens = _collect_special_tokens(destination, tokenizer_data)
    special_token_set = set(special_tokens)
    merged_tokens = tuple(
        dict.fromkeys(rule.merged for rule in ordered_rules if rule.merged in token_to_index)
    )
    merged_token_set = set(merged_tokens)
    non_merged_tokens = tuple(
        token
        for token, _ in sorted(token_to_index.items(), key=lambda item: item[1])
        if token not in special_token_set and token not in merged_token_set
    )

    token_order = (*special_tokens, *non_merged_tokens, *merged_tokens)
    if len(token_order) != len(token_to_index):
        raise ValueError("Token reassignment did not partition the tokenizer vocabulary")

    token_to_new_index = {token: index for index, token in enumerate(token_order)}
    if len(token_to_new_index) != len(token_order):
        raise ValueError("Token reassignment encountered duplicate token literals")

    mapping = {
        original_index: token_to_new_index[token]
        for token, original_index in token_to_index.items()
    }
    return _ReassignmentPlan(token_order=tuple(token_order), mapping=mapping)


def _collect_special_tokens(
    destination: Path,
    tokenizer_data: dict[str, Any],
) -> tuple[str, ...]:
    special_candidates: list[str] = []
    added_tokens = tokenizer_data.get("added_tokens")
    if isinstance(added_tokens, list):
        for token_data in added_tokens:
            if not isinstance(token_data, dict) or token_data.get("special") is not True:
                continue
            content = token_data.get("content")
            if isinstance(content, str):
                special_candidates.append(content)
    special_candidates.extend(_load_special_tokens(destination).values())

    token_to_index = _load_token_to_index(tokenizer_data)

    seen: set[str] = set()
    ordered_specials: list[str] = []
    for token in sorted(
        special_candidates,
        key=lambda candidate: _token_id(token_to_index, candidate),
    ):
        if token in seen:
            continue
        seen.add(token)
        ordered_specials.append(token)
    return tuple(ordered_specials)


def _token_id(token_to_index: dict[str, int], token: str) -> int:
    token_id = token_to_index.get(token)
    if token_id is None:
        raise ValueError(f"Missing token id for {token!r}")
    return token_id


def _write_reassigned_tokenizer(destination: Path, plan: _ReassignmentPlan) -> None:
    tokenizer_json_path = destination / "tokenizer.json"
    tokenizer_data = _load_json_object(tokenizer_json_path)
    model = tokenizer_data.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{tokenizer_json_path} is missing a JSON object model")

    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(f"{tokenizer_json_path} is missing a JSON object vocab")
    model["vocab"] = {token: plan.mapping[int(vocab[token])] for token in plan.token_order}

    added_tokens = tokenizer_data.get("added_tokens")
    if isinstance(added_tokens, list):
        _rewrite_added_tokens(added_tokens, plan.mapping)

    post_processor = tokenizer_data.get("post_processor")
    if isinstance(post_processor, dict):
        special_tokens = post_processor.get("special_tokens")
        post_processor["special_tokens"] = _rewrite_special_token_structure(
            special_tokens,
            plan.mapping,
        )

    tokenizer_json_path.write_text(
        json.dumps(tokenizer_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _rewrite_json_config_file(destination / "tokenizer_config.json", plan.mapping)
    _rewrite_json_config_file(destination / "config.json", plan.mapping, vocab_size=len(plan.token_order))
    _rewrite_json_config_file(destination / "generation_config.json", plan.mapping)
    _rewrite_added_tokens_json(destination, plan.mapping)


def _rewrite_added_tokens(
    added_tokens: list[Any],
    mapping: dict[int, int],
) -> None:
    for token_data in added_tokens:
        if not isinstance(token_data, dict):
            continue
        token_id = token_data.get("id")
        if isinstance(token_id, int):
            token_data["id"] = _remap_id(mapping, token_id)
    added_tokens.sort(key=_added_token_sort_key)


def _added_token_sort_key(token_data: Any) -> tuple[int, str]:
    if not isinstance(token_data, dict):
        return (1 << 30, "")
    token_id = token_data.get("id")
    content = token_data.get("content")
    return (
        token_id if isinstance(token_id, int) else 1 << 30,
        content if isinstance(content, str) else "",
    )


def _rewrite_special_token_structure(
    value: Any,
    mapping: dict[int, int],
) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_special_token_structure_item(key, item, mapping)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_special_token_structure(item, mapping) for item in value]
    return value


def _rewrite_special_token_structure_item(
    key: str,
    value: Any,
    mapping: dict[int, int],
) -> Any:
    if key == "id" and isinstance(value, int):
        return _remap_id(mapping, value)
    if key == "ids" and isinstance(value, list):
        return [_remap_id(mapping, item) if isinstance(item, int) else item for item in value]
    return _rewrite_special_token_structure(value, mapping)


def _rewrite_json_config_file(
    path: Path,
    mapping: dict[int, int],
    vocab_size: int | None = None,
) -> None:
    if not path.exists():
        return

    data = _load_json_object(path)
    rewritten = _rewrite_config_value(data, mapping, vocab_size=vocab_size)
    path.write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _rewrite_config_value(value: Any, mapping: dict[int, int], *, vocab_size: int | None) -> Any:
    if isinstance(value, dict):
        rewritten: dict[str, Any] = {}
        for key, item in value.items():
            if key == "added_tokens_decoder" and isinstance(item, dict):
                rewritten[key] = _rewrite_added_tokens_decoder(item, mapping)
                continue
            if key == "added_tokens_encoder" and isinstance(item, dict):
                rewritten[key] = _rewrite_added_tokens_encoder(item, mapping)
                continue
            if key.endswith("_token_id") or key == "decoder_start_token_id":
                rewritten[key] = _rewrite_token_id_value(item, mapping)
                continue
            if (
                key in {"vocab_size", "vocab_size_per_layer_input"}
                and isinstance(item, int)
                and vocab_size is not None
            ):
                rewritten[key] = vocab_size
                continue
            rewritten[key] = _rewrite_config_value(item, mapping, vocab_size=vocab_size)
        return rewritten
    if isinstance(value, list):
        return [_rewrite_config_value(item, mapping, vocab_size=vocab_size) for item in value]
    return value


def _rewrite_added_tokens_decoder(
    decoder: dict[str, Any],
    mapping: dict[int, int],
) -> dict[str, Any]:
    rewritten_items: list[tuple[int, str, Any]] = []
    for key, value in decoder.items():
        original_id = int(key)
        new_id = _remap_id(mapping, original_id)
        rewritten_value = _rewrite_special_token_structure(value, mapping)
        rewritten_items.append((new_id, str(new_id), rewritten_value))
    rewritten_items.sort(key=lambda item: item[0])
    return {key: value for _, key, value in rewritten_items}


def _rewrite_added_tokens_encoder(
    encoder: dict[str, Any],
    mapping: dict[int, int],
) -> dict[str, Any]:
    return {
        token: _remap_id(mapping, token_id) if isinstance(token_id, int) else token_id
        for token, token_id in encoder.items()
    }


def _rewrite_token_id_value(value: Any, mapping: dict[int, int]) -> Any:
    if isinstance(value, int):
        return _remap_id(mapping, value)
    if isinstance(value, list):
        return [_remap_id(mapping, item) if isinstance(item, int) else item for item in value]
    return value


def _rewrite_added_tokens_json(destination: Path, mapping: dict[int, int]) -> None:
    added_tokens_path = destination / "added_tokens.json"
    if not added_tokens_path.exists():
        return

    added_tokens_data = _load_json_object(added_tokens_path)
    rewritten = _rewrite_added_tokens_encoder(added_tokens_data, mapping)
    added_tokens_path.write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_tokenizer_mapping(destination: Path, mapping: dict[int, int]) -> None:
    mapping_path = destination / "tokenizer_mapping.json"
    serialized_mapping = {
        str(original_id): new_id
        for original_id, new_id in sorted(mapping.items())
    }
    mapping_path.write_text(
        json.dumps(serialized_mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _remap_id(mapping: dict[int, int], token_id: int) -> int:
    remapped = mapping.get(token_id)
    if remapped is None:
        raise ValueError(f"Missing remapped token id for original id {token_id}")
    return remapped


def _rewrite_special_tokens(
    *,
    destination: Path,
    contents: TokenizerContents,
    special_token_literal_map: dict[str, str | None],
) -> _SpecialTokenRewritePlan:
    tokenizer_json_path = destination / "tokenizer.json"
    if not tokenizer_json_path.exists():
        raise ValueError("Special-token remapping requires tokenizer.json")

    tokenizer_data = _load_json_object(tokenizer_json_path)
    plan = _build_special_token_rewrite_plan(
        contents=contents,
        tokenizer_data=tokenizer_data,
        special_token_literal_map=special_token_literal_map,
    )
    _rewrite_tokenizer_json_special_tokens(tokenizer_json_path, tokenizer_data, plan)
    for filename in (
        "special_tokens_map.json",
        "tokenizer_config.json",
        "config.json",
        "generation_config.json",
    ):
        _rewrite_special_token_config_file(destination / filename, plan)
    _rewrite_special_token_added_tokens_file(destination / "added_tokens.json", plan)
    _rewrite_chat_template_files(destination, plan)
    return plan


def _build_special_token_rewrite_plan(
    *,
    contents: TokenizerContents,
    tokenizer_data: dict[str, Any],
    special_token_literal_map: dict[str, str | None],
) -> _SpecialTokenRewritePlan:
    current_specials = set(_collect_original_special_token_literals(contents, tokenizer_data))
    unknown_literals = sorted(set(special_token_literal_map) - current_specials)
    if unknown_literals:
        raise ValueError(
            f"Special-token map references non-special literals: {', '.join(unknown_literals)}"
        )

    token_to_index = _load_token_to_index(tokenizer_data)
    replacements: dict[str, str] = {}
    dropped_literals: set[str] = set()
    for literal in current_specials:
        if literal not in special_token_literal_map:
            dropped_literals.add(literal)
            continue
        new_literal = special_token_literal_map[literal]
        replacements[literal] = literal if new_literal is None else new_literal

    _validate_special_token_replacements(
        token_to_index=token_to_index,
        replacements=replacements,
        dropped_literals=dropped_literals,
    )
    dropped_ids = frozenset(token_to_index[literal] for literal in dropped_literals)
    return _SpecialTokenRewritePlan(
        replacements=replacements,
        dropped_literals=frozenset(dropped_literals),
        dropped_ids=dropped_ids,
    )


def _collect_original_special_token_literals(
    contents: TokenizerContents,
    tokenizer_data: dict[str, Any],
) -> tuple[str, ...]:
    special_candidates = list(contents.special_tokens.values())
    added_tokens = tokenizer_data.get("added_tokens")
    if isinstance(added_tokens, list):
        for token_data in added_tokens:
            if not isinstance(token_data, dict) or token_data.get("special") is not True:
                continue
            content = token_data.get("content")
            if isinstance(content, str):
                special_candidates.append(content)
    seen: set[str] = set()
    ordered: list[str] = []
    token_to_index = _load_token_to_index(tokenizer_data)
    for literal in sorted(set(special_candidates), key=lambda token: _token_id(token_to_index, token)):
        if literal in seen:
            continue
        seen.add(literal)
        ordered.append(literal)
    return tuple(ordered)


def _validate_special_token_replacements(
    *,
    token_to_index: dict[str, int],
    replacements: dict[str, str],
    dropped_literals: set[str],
) -> None:
    renamed_targets = [new_literal for old_literal, new_literal in replacements.items() if new_literal != old_literal]
    if len(set(renamed_targets)) != len(renamed_targets):
        raise ValueError("Special-token remapping cannot rename multiple tokens to the same literal")

    for old_literal, new_literal in replacements.items():
        if new_literal == old_literal:
            continue
        if new_literal not in token_to_index:
            continue
        if new_literal in dropped_literals:
            continue
        if new_literal in replacements and replacements[new_literal] != new_literal:
            continue
        raise ValueError(f"Special-token remapping would collide with existing token {new_literal!r}")


def _rewrite_tokenizer_json_special_tokens(
    path: Path,
    tokenizer_data: dict[str, Any],
    plan: _SpecialTokenRewritePlan,
) -> None:
    model = tokenizer_data.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{path} is missing a JSON object model")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(f"{path} is missing a JSON object vocab")

    rewritten_vocab: dict[str, int] = {}
    for token, token_id in sorted(vocab.items(), key=lambda item: int(item[1])):
        if token in plan.dropped_literals:
            continue
        rewritten_vocab[_rewrite_literal(token, plan)] = int(token_id)
    model["vocab"] = rewritten_vocab

    added_tokens = tokenizer_data.get("added_tokens")
    if isinstance(added_tokens, list):
        tokenizer_data["added_tokens"] = _rewrite_tokenizer_added_tokens(added_tokens, plan)

    post_processor = tokenizer_data.get("post_processor")
    if isinstance(post_processor, dict):
        rewritten_post_processor = _rewrite_post_processor(post_processor, plan)
        tokenizer_data["post_processor"] = rewritten_post_processor

    path.write_text(
        json.dumps(tokenizer_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _rewrite_tokenizer_added_tokens(
    added_tokens: list[Any],
    plan: _SpecialTokenRewritePlan,
) -> list[Any]:
    rewritten: list[Any] = []
    for token_data in added_tokens:
        if not isinstance(token_data, dict):
            rewritten.append(token_data)
            continue
        content = token_data.get("content")
        is_special = token_data.get("special") is True
        if not is_special or not isinstance(content, str) or content not in plan.dropped_literals | set(plan.replacements):
            rewritten.append(token_data)
            continue
        if content in plan.dropped_literals:
            continue
        rewritten_token_data = dict(token_data)
        rewritten_token_data["content"] = _rewrite_literal(content, plan)
        rewritten.append(rewritten_token_data)
    return rewritten


def _rewrite_post_processor(value: Any, plan: _SpecialTokenRewritePlan, *, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        rewritten: dict[str, Any] = {}
        for key, item in value.items():
            if parent_key == "special_tokens":
                if key in plan.dropped_literals:
                    continue
                rewritten_key = _rewrite_literal(key, plan)
            else:
                rewritten_key = key
            rewritten_item = _rewrite_post_processor(item, plan, parent_key=rewritten_key)
            if rewritten_item is _DROP_VALUE:
                continue
            rewritten[rewritten_key] = rewritten_item
        return rewritten
    if isinstance(value, list):
        rewritten_items = [
            _rewrite_post_processor(item, plan, parent_key=parent_key)
            for item in value
        ]
        return [item for item in rewritten_items if item is not _DROP_VALUE]
    if isinstance(value, str) and parent_key in {"content", "token"}:
        if value in plan.dropped_literals:
            return _DROP_VALUE
        return _rewrite_literal(value, plan)
    if isinstance(value, str) and parent_key == "tokens":
        if value in plan.dropped_literals:
            return _DROP_VALUE
        return _rewrite_literal(value, plan)
    return value


def _rewrite_special_token_config_file(path: Path, plan: _SpecialTokenRewritePlan) -> None:
    if not path.exists():
        return
    data = _load_json_object(path)
    rewritten = _rewrite_special_token_config_value(data, plan)
    path.write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _rewrite_special_token_config_value(value: Any, plan: _SpecialTokenRewritePlan) -> Any:
    if isinstance(value, dict):
        rewritten: dict[str, Any] = {}
        for key, item in value.items():
            if key == "content" and isinstance(item, str):
                if item in plan.dropped_literals:
                    return _DROP_VALUE
                rewritten[key] = _rewrite_literal(item, plan)
                continue
            if key == "tokens" and isinstance(item, list):
                rewritten_tokens = [
                    _rewrite_literal(token, plan)
                    for token in item
                    if not isinstance(token, str) or token not in plan.dropped_literals
                ]
                rewritten[key] = rewritten_tokens
                continue
            if key.endswith("_token"):
                rewritten_token = _rewrite_special_token_descriptor(item, plan)
                if rewritten_token is _DROP_VALUE:
                    continue
                rewritten[key] = rewritten_token
                continue
            if key.endswith("_token_id") or key == "decoder_start_token_id":
                rewritten_token_id = _rewrite_special_token_id_value(item, plan)
                if rewritten_token_id is _DROP_VALUE:
                    continue
                rewritten[key] = rewritten_token_id
                continue
            if key == "extra_special_tokens":
                rewritten_extra = _rewrite_extra_special_tokens(item, plan)
                if rewritten_extra in (_DROP_VALUE, {}, []):
                    continue
                rewritten[key] = rewritten_extra
                continue
            if key == "added_tokens_decoder" and isinstance(item, dict):
                rewritten_decoder = _rewrite_special_added_tokens_decoder(item, plan)
                if rewritten_decoder:
                    rewritten[key] = rewritten_decoder
                continue
            if key == "added_tokens_encoder" and isinstance(item, dict):
                rewritten_encoder = _rewrite_special_added_tokens_encoder(item, plan)
                if rewritten_encoder:
                    rewritten[key] = rewritten_encoder
                continue
            rewritten[key] = _rewrite_special_token_config_value(item, plan)
        return rewritten
    if isinstance(value, list):
        return [_rewrite_special_token_config_value(item, plan) for item in value]
    return value


def _rewrite_special_token_descriptor(value: Any, plan: _SpecialTokenRewritePlan) -> Any:
    literal = _special_token_literal_from_descriptor(value)
    if literal is None:
        return value
    if literal in plan.dropped_literals:
        return _DROP_VALUE
    rewritten_literal = _rewrite_literal(literal, plan)
    if isinstance(value, dict):
        rewritten_value = dict(value)
        if "content" in rewritten_value:
            rewritten_value["content"] = rewritten_literal
        return rewritten_value
    return rewritten_literal


def _special_token_literal_from_descriptor(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


def _rewrite_special_token_id_value(value: Any, plan: _SpecialTokenRewritePlan) -> Any:
    if isinstance(value, int):
        return _DROP_VALUE if value in plan.dropped_ids else value
    if isinstance(value, list):
        rewritten = [item for item in value if not isinstance(item, int) or item not in plan.dropped_ids]
        return rewritten if rewritten else _DROP_VALUE
    return value


def _rewrite_extra_special_tokens(value: Any, plan: _SpecialTokenRewritePlan) -> Any:
    if isinstance(value, dict):
        rewritten: dict[str, Any] = {}
        for key, item in value.items():
            rewritten_item = _rewrite_special_token_descriptor(item, plan)
            if rewritten_item is _DROP_VALUE:
                continue
            rewritten[key] = rewritten_item
        return rewritten
    if isinstance(value, list):
        rewritten_list = []
        for item in value:
            rewritten_item = _rewrite_special_token_descriptor(item, plan)
            if rewritten_item is _DROP_VALUE:
                continue
            rewritten_list.append(rewritten_item)
        return rewritten_list
    return value


def _rewrite_special_added_tokens_decoder(
    decoder: dict[str, Any],
    plan: _SpecialTokenRewritePlan,
) -> dict[str, Any]:
    rewritten: dict[str, Any] = {}
    for key, value in decoder.items():
        if key.isdigit() and int(key) in plan.dropped_ids:
            continue
        rewritten_value = _rewrite_special_token_config_value(value, plan)
        if rewritten_value is _DROP_VALUE:
            continue
        rewritten[key] = rewritten_value
    return rewritten


def _rewrite_special_added_tokens_encoder(
    encoder: dict[str, Any],
    plan: _SpecialTokenRewritePlan,
) -> dict[str, Any]:
    rewritten: dict[str, Any] = {}
    for token, token_id in encoder.items():
        if token in plan.dropped_literals:
            continue
        if isinstance(token_id, int) and token_id in plan.dropped_ids:
            continue
        rewritten[_rewrite_literal(token, plan)] = token_id
    return rewritten


def _rewrite_special_token_added_tokens_file(path: Path, plan: _SpecialTokenRewritePlan) -> None:
    if not path.exists():
        return
    data = _load_json_object(path)
    rewritten = _rewrite_special_added_tokens_encoder(data, plan)
    path.write_text(
        json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _rewrite_chat_template_files(destination: Path, plan: _SpecialTokenRewritePlan) -> None:
    tokenizer_config_path = destination / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        data = _load_json_object(tokenizer_config_path)
        chat_template = data.get("chat_template")
        if isinstance(chat_template, str):
            data["chat_template"] = _rewrite_chat_template_text(chat_template, plan)
            tokenizer_config_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    chat_template_path = destination / "chat_template.jinja"
    if chat_template_path.exists():
        chat_template = chat_template_path.read_text(encoding="utf-8")
        chat_template_path.write_text(
            _rewrite_chat_template_text(chat_template, plan),
            encoding="utf-8",
        )


def _rewrite_chat_template_text(text: str, plan: _SpecialTokenRewritePlan) -> str:
    return _rewrite_template(
        text,
        replacements=plan.replacements,
        dropped_literals=plan.dropped_literals,
    )


def _load_token_to_index(tokenizer_data: dict[str, Any]) -> dict[str, int]:
    model = tokenizer_data.get("model")
    if not isinstance(model, dict):
        raise ValueError("tokenizer.json is missing a JSON object model")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError("tokenizer.json is missing a JSON object vocab")
    return {token: int(token_id) for token, token_id in vocab.items()}


def _rewrite_literal(literal: str, plan: _SpecialTokenRewritePlan) -> str:
    return plan.replacements.get(literal, literal)


_DROP_VALUE = object()


def _write_tokenizer_json_merges(
    tokenizer_json_path: Path,
    merges: list[tuple[str, str]],
) -> None:
    tokenizer_data = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
    if not isinstance(tokenizer_data, dict):
        raise ValueError(f"{tokenizer_json_path} does not contain a JSON object")
    model = tokenizer_data.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{tokenizer_json_path} is missing a JSON object model")
    model["merges"] = [list(merge) for merge in merges]
    tokenizer_json_path.write_text(
        json.dumps(tokenizer_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_merges_txt(
    merges_txt_path: Path,
    merges: list[tuple[str, str]],
) -> None:
    lines = merges_txt_path.read_text(encoding="utf-8").splitlines()
    header_lines: list[str] = []
    for line in lines:
        if not line.startswith("#"):
            break
        header_lines.append(line)

    merge_lines = [f"{left} {right}" for left, right in merges]
    output_lines = [*header_lines, *merge_lines]
    merges_txt_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


__all__ = ["save_reordered_tokenizer"]
