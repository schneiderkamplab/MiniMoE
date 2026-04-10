"""Tokenizer loading helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download

_TOKENIZER_ALLOW_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "sentencepiece.bpe.model",
)
_NO_RULE = -1
_TRANSITIVE_REDUCTION_BATCH_SIZE = 4096


@dataclass(frozen=True, slots=True)
class MergeRule:
    """One merge rule in the tokenizer's original order."""

    index: int
    left: str
    right: str
    merged: str


@dataclass(frozen=True, slots=True)
class MergeRuleGraph:
    """A conservative precedence DAG over merge rules."""

    rules: tuple[MergeRule, ...]
    predecessors: dict[int, tuple[int, ...]]
    successors: dict[int, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class TokenizerContents:
    """Tokenizer contents split into special tokens, vocab mappings, and a merge graph."""

    source: str
    resolved_dir: Path
    special_tokens: dict[str, str]
    token_to_index: dict[str, int]
    index_to_token: dict[int, str]
    original_merges: tuple[tuple[str, str], ...]
    graph: MergeRuleGraph


def load_tokenizer_contents(
    source: str | Path,
    *,
    models_dir: str | Path = "models",
) -> TokenizerContents:
    """Load tokenizer metadata from a local directory or Hugging Face repo id.

    If ``source`` names both a local directory and a Hugging Face repo id, the local
    directory is preferred. When only a repo id is available, tokenizer files are
    downloaded into ``models_dir / source``.
    """

    resolved_dir = _resolve_tokenizer_dir(source=source, models_dir=models_dir)
    special_tokens = _load_special_tokens(resolved_dir)
    token_to_index, merges = _load_vocab_and_merges(resolved_dir)
    index_to_token = _invert_vocab(token_to_index)
    graph = _load_merge_rule_graph(merges)
    return TokenizerContents(
        source=str(source),
        resolved_dir=resolved_dir,
        special_tokens=special_tokens,
        token_to_index=token_to_index,
        index_to_token=index_to_token,
        original_merges=tuple(merges),
        graph=graph,
    )


def _resolve_tokenizer_dir(source: str | Path, models_dir: str | Path) -> Path:
    source_path = Path(source).expanduser()
    if source_path.is_dir():
        return source_path.resolve()

    if source_path.is_file():
        return source_path.parent.resolve()

    local_candidate = Path(models_dir).expanduser() / str(source)
    if local_candidate.is_dir():
        return local_candidate.resolve()

    snapshot_path = snapshot_download(
        repo_id=str(source),
        local_dir=str(local_candidate),
        allow_patterns=list(_TOKENIZER_ALLOW_PATTERNS),
    )
    return Path(snapshot_path).resolve()


def _load_special_tokens(tokenizer_dir: Path) -> dict[str, str]:
    special_tokens: dict[str, str] = {}
    for filename in ("special_tokens_map.json", "tokenizer_config.json"):
        path = tokenizer_dir / filename
        if not path.exists():
            continue
        data = _load_json(path)
        special_tokens.update(_extract_special_tokens(data))
    return special_tokens


def _extract_special_tokens(data: dict[str, Any]) -> dict[str, str]:
    extracted: dict[str, str] = {}

    for key, value in data.items():
        if key.endswith("_token"):
            literal = _special_token_literal(value)
            if literal is not None:
                extracted[key] = literal

    extra_special_tokens = data.get("extra_special_tokens")
    if isinstance(extra_special_tokens, dict):
        for key, value in extra_special_tokens.items():
            literal = _special_token_literal(value)
            if literal is not None:
                extracted[key] = literal
    elif isinstance(extra_special_tokens, list):
        for index, value in enumerate(extra_special_tokens):
            literal = _special_token_literal(value)
            if literal is not None:
                extracted[f"extra_special_tokens[{index}]"] = literal

    return extracted


def _special_token_literal(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
    return None


def _load_merge_rule_graph(merges: list[tuple[str, str]]) -> MergeRuleGraph:
    rules = tuple(
        MergeRule(index=index, left=left, right=right, merged=left + right)
        for index, (left, right) in enumerate(merges)
    )
    touching_rules: dict[str, list[int]] = defaultdict(list)
    for rule in rules:
        touching_rules[rule.left].append(rule.index)
        if rule.right != rule.left:
            touching_rules[rule.right].append(rule.index)

    predecessors: dict[int, set[int]] = defaultdict(set)
    successors: dict[int, set[int]] = defaultdict(set)
    for indices in touching_rules.values():
        for previous, current in pairwise(indices):
            predecessors[current].add(previous)
            successors[previous].add(current)

    _reduce_transitive_edges(predecessors=predecessors, successors=successors, rule_count=len(rules))

    return MergeRuleGraph(
        rules=rules,
        predecessors={
            rule.index: tuple(sorted(predecessors.get(rule.index, ())))
            for rule in rules
        },
        successors={
            rule.index: tuple(sorted(successors.get(rule.index, ())))
            for rule in rules
        },
    )


def _reduce_transitive_edges(
    *,
    predecessors: dict[int, set[int]],
    successors: dict[int, set[int]],
    rule_count: int,
) -> None:
    predecessor_pairs = _build_pair_array(adjacency=predecessors, rule_count=rule_count)
    successor_pairs = _build_pair_array(adjacency=successors, rule_count=rule_count)
    reduction_queries = _build_reduction_queries(predecessor_pairs=predecessor_pairs)
    if reduction_queries is None:
        return

    query_nodes, query_lowers, query_uppers = reduction_queries
    redundant_nodes = _find_redundant_query_nodes(
        rule_count=rule_count,
        successor_pairs=successor_pairs,
        query_nodes=query_nodes,
        query_lowers=query_lowers,
        query_uppers=query_uppers,
    )

    for rule_index in redundant_nodes:
        lower_predecessor = int(predecessor_pairs[rule_index, 0])
        current_predecessors = predecessors.get(rule_index)
        if current_predecessors is not None:
            current_predecessors.discard(lower_predecessor)

        successor_set = successors.get(lower_predecessor)
        if successor_set is not None:
            successor_set.discard(int(rule_index))


def _build_pair_array(
    *,
    adjacency: dict[int, set[int]],
    rule_count: int,
) -> np.ndarray[Any, np.dtype[np.int32]]:
    pair_array = np.full((rule_count, 2), _NO_RULE, dtype=np.int32)
    for rule_index, neighbors in adjacency.items():
        sorted_neighbors = sorted(neighbors)
        if len(sorted_neighbors) > 2:
            raise ValueError("Merge graph helper expects indegree and outdegree to stay within 2")
        for slot, neighbor in enumerate(sorted_neighbors):
            pair_array[rule_index, slot] = neighbor
    return pair_array


def _build_reduction_queries(
    *,
    predecessor_pairs: np.ndarray[Any, np.dtype[np.int32]],
) -> tuple[np.ndarray[Any, np.dtype[np.int32]], ...] | None:
    query_nodes = np.flatnonzero(predecessor_pairs[:, 1] != _NO_RULE).astype(np.int32, copy=False)
    if query_nodes.size == 0:
        return None

    query_lowers = predecessor_pairs[query_nodes, 0].astype(np.int32, copy=False)
    query_uppers = predecessor_pairs[query_nodes, 1].astype(np.int32, copy=False)
    return query_nodes, query_lowers, query_uppers


def _find_redundant_query_nodes(
    *,
    rule_count: int,
    successor_pairs: np.ndarray[Any, np.dtype[np.int32]],
    query_nodes: np.ndarray[Any, np.dtype[np.int32]],
    query_lowers: np.ndarray[Any, np.dtype[np.int32]],
    query_uppers: np.ndarray[Any, np.dtype[np.int32]],
) -> np.ndarray[Any, np.dtype[np.int32]]:
    unique_uppers, upper_inverse = np.unique(query_uppers, return_inverse=True)
    query_order = np.argsort(upper_inverse, kind="stable")
    first_successors = successor_pairs[:, 0].tolist()
    second_successors = successor_pairs[:, 1].tolist()
    ordered_nodes = query_nodes[query_order].tolist()
    ordered_lowers = query_lowers[query_order].tolist()
    ordered_groups = upper_inverse[query_order].tolist()
    group_offsets = np.searchsorted(
        upper_inverse[query_order],
        np.arange(unique_uppers.size + 1, dtype=np.int32),
    ).tolist()
    redundant_nodes: list[int] = []

    for batch_start in range(0, unique_uppers.size, _TRANSITIVE_REDUCTION_BATCH_SIZE):
        batch_end = min(batch_start + _TRANSITIVE_REDUCTION_BATCH_SIZE, unique_uppers.size)
        batch_uppers = unique_uppers[batch_start:batch_end].tolist()
        reachable_masks = [0] * rule_count
        for local_bit, upper in enumerate(batch_uppers):
            reachable_masks[int(upper)] = 1 << local_bit

        for rule_index in range(rule_count - 1, -1, -1):
            mask = reachable_masks[rule_index]
            first_successor = first_successors[rule_index]
            if first_successor != _NO_RULE:
                mask |= reachable_masks[first_successor]
            second_successor = second_successors[rule_index]
            if second_successor != _NO_RULE:
                mask |= reachable_masks[second_successor]
            reachable_masks[rule_index] = mask

        query_start = group_offsets[batch_start]
        query_end = group_offsets[batch_end]
        for ordered_index in range(query_start, query_end):
            local_bit = ordered_groups[ordered_index] - batch_start
            lower_predecessor = ordered_lowers[ordered_index]
            if (reachable_masks[lower_predecessor] >> local_bit) & 1:
                redundant_nodes.append(ordered_nodes[ordered_index])

    return np.array(redundant_nodes, dtype=np.int32)


def _load_vocab_and_merges(tokenizer_dir: Path) -> tuple[dict[str, int], list[tuple[str, str]]]:
    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    if tokenizer_json_path.exists():
        tokenizer_data = _load_json(tokenizer_json_path)
        model = tokenizer_data.get("model", {})
        vocab = model.get("vocab")
        if not isinstance(vocab, dict):
            raise ValueError(f"{tokenizer_json_path} does not contain a dictionary vocab")
        return _normalize_vocab(vocab), _normalize_merges(model.get("merges", []))

    vocab_path = tokenizer_dir / "vocab.json"
    if not vocab_path.exists():
        raise ValueError(
            f"Could not find tokenizer.json or vocab.json in {tokenizer_dir}"
        )

    vocab_data = _load_json(vocab_path)
    merges_path = tokenizer_dir / "merges.txt"
    merges = _load_merges_txt(merges_path) if merges_path.exists() else []
    return _normalize_vocab(vocab_data), merges


def _normalize_vocab(vocab: dict[str, Any]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for literal, token_id in vocab.items():
        if not isinstance(literal, str):
            raise ValueError("Tokenizer vocab contains a non-string token literal")
        if not isinstance(token_id, int):
            raise ValueError(f"Tokenizer vocab entry {literal!r} does not map to an integer id")
        normalized[literal] = token_id
    return normalized


def _invert_vocab(vocab: dict[str, int]) -> dict[int, str]:
    inverted: dict[int, str] = {}
    for literal, token_id in vocab.items():
        if token_id in inverted:
            raise ValueError(f"Tokenizer vocab reuses token id {token_id}")
        inverted[token_id] = literal
    return inverted


def _normalize_merges(raw_merges: Any) -> list[tuple[str, str]]:
    if not isinstance(raw_merges, list):
        return []

    merges: list[tuple[str, str]] = []
    for merge in raw_merges:
        if isinstance(merge, str):
            parts = merge.split(" ", maxsplit=1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
            continue
        if isinstance(merge, list) and len(merge) == 2 and all(isinstance(part, str) for part in merge):
            merges.append((merge[0], merge[1]))
    return merges


def _load_merges_txt(merges_path: Path) -> list[tuple[str, str]]:
    merges: list[tuple[str, str]] = []
    for raw_line in merges_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(" ", maxsplit=1)
        if len(parts) == 2:
            merges.append((parts[0], parts[1]))
    return merges


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


__all__ = [
    "MergeRule",
    "MergeRuleGraph",
    "TokenizerContents",
    "load_tokenizer_contents",
]
