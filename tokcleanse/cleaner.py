"""Graph-to-merge cleaning helpers."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any, Literal, TypeAlias

from .loader import MergeRule, TokenizerContents

CleanMode: TypeAlias = Literal["id", "literal"]
MergeOrder: TypeAlias = Callable[[MergeRule], Any]
OrderName: TypeAlias = Literal[
    "original",
    "good",
    "length_then_lexicographic",
    "lexicographic_then_length",
    "descending_length_then_lexicographic",
    "operands_then_output",
    "whitespace_priority_then_length",
]

DEFAULT_ORDER_NAME: OrderName = "original"
DEFAULT_SAVE_ORDER_NAME: OrderName = "good"


def available_order_names() -> tuple[str, ...]:
    """Return the built-in named orderings."""

    return tuple(_ORDER_FUNCTIONS)


def describe_order(name: str) -> str:
    """Return a one-line description for a built-in ordering."""

    description = _ORDER_DESCRIPTIONS.get(name)
    if description is None:
        raise ValueError(f"Unknown order name: {name}")
    return description


def clean(
    contents: TokenizerContents,
    *,
    mode: CleanMode = "id",
    order: MergeOrder | None = None,
    order_name: str = DEFAULT_ORDER_NAME,
    seed: int | None = None,
) -> list[tuple[int, int, int]] | list[tuple[str, str]]:
    """Return merge operations in a chosen topological order.

    With the default ``order_name="original"``, this yields the tokenizer's
    original merge order. Set ``seed`` for a deterministic random topological
    order, pass ``order_name`` for a built-in deterministic heuristic, or pass
    ``order`` to prioritize currently-available merge rules with a custom key.
    """

    rules = topological_sort_rules(
        contents,
        order=order,
        order_name=order_name,
        seed=seed,
    )
    if mode == "literal":
        return [(rule.left, rule.right) for rule in rules]
    if mode != "id":
        raise ValueError(f"Unsupported clean mode: {mode}")

    return [
        (
            _token_id(contents, rule.merged),
            _token_id(contents, rule.left),
            _token_id(contents, rule.right),
        )
        for rule in rules
    ]


def topological_sort_rules(
    contents: TokenizerContents,
    *,
    order: MergeOrder | None = None,
    order_name: str = DEFAULT_ORDER_NAME,
    seed: int | None = None,
) -> list[MergeRule]:
    """Return a topological sort of the merge-rule graph."""

    resolved_order = _resolve_order(order=order, order_name=order_name, seed=seed)

    rules_by_index = {rule.index: rule for rule in contents.graph.rules}
    remaining_predecessors = {
        rule.index: len(contents.graph.predecessors[rule.index])
        for rule in contents.graph.rules
    }
    available = [
        rule
        for rule in contents.graph.rules
        if remaining_predecessors[rule.index] == 0
    ]
    sorted_rules: list[MergeRule] = []
    rng = random.Random(seed) if seed is not None else None

    while available:
        next_rule = _pick_next_rule(available, order=resolved_order, rng=rng)
        sorted_rules.append(next_rule)

        for successor in contents.graph.successors[next_rule.index]:
            remaining_predecessors[successor] -= 1
            if remaining_predecessors[successor] == 0:
                available.append(rules_by_index[successor])

    if len(sorted_rules) != len(contents.graph.rules):
        raise ValueError("Merge graph contains a cycle")

    return sorted_rules


def _pick_next_rule(
    available: list[MergeRule],
    *,
    order: MergeOrder | None,
    rng: random.Random | None,
) -> MergeRule:
    if rng is not None:
        return available.pop(rng.randrange(len(available)))

    if order is None:
        best_index = min(range(len(available)), key=lambda index: available[index].index)
        return available.pop(best_index)

    best_index = min(
        range(len(available)),
        key=lambda index: (order(available[index]), available[index].index),
    )
    return available.pop(best_index)


def _token_id(contents: TokenizerContents, literal: str) -> int:
    token_id = contents.token_to_index.get(literal)
    if token_id is None:
        raise ValueError(f"Missing token id for {literal!r}")
    return token_id


def _resolve_order(
    *,
    order: MergeOrder | None,
    order_name: str,
    seed: int | None,
) -> MergeOrder | None:
    if order is not None and seed is not None:
        raise ValueError("Pass either `order` or `seed`, not both")
    if seed is not None and order_name != "original":
        raise ValueError("Pass `seed` only with `order_name=\"original\"`")
    if order is not None and order_name != "original":
        raise ValueError("Pass either `order` or a non-original `order_name`, not both")
    if order is not None:
        return order
    if order_name == "original":
        return None
    resolved_order = _ORDER_FUNCTIONS.get(order_name)
    if resolved_order is None:
        raise ValueError(f"Unknown order name: {order_name}")
    return resolved_order


def _order_by_length_then_lexicographic(rule: MergeRule) -> tuple[int, str, str, str, int]:
    return (len(rule.merged), rule.merged, rule.left, rule.right, rule.index)


def _order_by_lexicographic_then_length(rule: MergeRule) -> tuple[str, int, str, str, int]:
    return (rule.merged, len(rule.merged), rule.left, rule.right, rule.index)


def _order_by_descending_length_then_lexicographic(rule: MergeRule) -> tuple[int, str, str, str, int]:
    return (-len(rule.merged), rule.merged, rule.left, rule.right, rule.index)


def _order_by_operands_then_output(rule: MergeRule) -> tuple[str, str, str, int]:
    return (rule.left, rule.right, rule.merged, rule.index)


def _order_by_whitespace_priority_then_length(rule: MergeRule) -> tuple[int, int, str, str, str, int]:
    return (
        _whitespace_priority(rule),
        -len(rule.merged),
        rule.merged,
        rule.left,
        rule.right,
        rule.index,
    )


def _whitespace_priority(rule: MergeRule) -> int:
    if rule.merged and rule.merged[0] in {"\n", "\t", " "}:
        return 0
    if rule.merged.startswith("▁"):
        return 1
    return 2


_ORDER_FUNCTIONS: dict[str, MergeOrder] = {
    "good": _order_by_length_then_lexicographic,
    "length_then_lexicographic": _order_by_length_then_lexicographic,
    "lexicographic_then_length": _order_by_lexicographic_then_length,
    "descending_length_then_lexicographic": _order_by_descending_length_then_lexicographic,
    "operands_then_output": _order_by_operands_then_output,
    "whitespace_priority_then_length": _order_by_whitespace_priority_then_length,
}

_ORDER_DESCRIPTIONS: dict[str, str] = {
    "original": "Preserve the tokenizer's original merge order.",
    "good": "Prefer shorter merged outputs first, then break ties lexicographically.",
    "length_then_lexicographic": "Prefer shorter merged outputs first, then break ties lexicographically.",
    "lexicographic_then_length": "Prefer lexicographically smaller merged outputs, then shorter ones.",
    "descending_length_then_lexicographic": "Prefer longer merged outputs first, then break ties lexicographically.",
    "operands_then_output": "Prefer merges by left operand, then right operand, then merged output.",
    "whitespace_priority_then_length": "Prioritize whitespace-like merges, then longer outputs, then lexicographic ties.",
}


__all__ = [
    "DEFAULT_ORDER_NAME",
    "DEFAULT_SAVE_ORDER_NAME",
    "available_order_names",
    "clean",
    "describe_order",
    "topological_sort_rules",
]
