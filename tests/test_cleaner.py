from tokcleanse import TokenizerContents, clean
from tokcleanse.loader import _reduce_transitive_edges


def test_clean_mode_id_matches_original_merges(tokenizer_contents: TokenizerContents) -> None:
    contents = tokenizer_contents
    expected = [
        (
            contents.token_to_index[left + right],
            contents.token_to_index[left],
            contents.token_to_index[right],
        )
        for left, right in contents.original_merges
    ]

    assert clean(contents, mode="id") == expected


def test_original_rule_order_is_a_topological_sort(tokenizer_contents: TokenizerContents) -> None:
    contents = tokenizer_contents
    seen: set[int] = set()
    for rule in contents.graph.rules:
        assert all(predecessor in seen for predecessor in contents.graph.predecessors[rule.index])
        seen.add(rule.index)


def test_reduce_transitive_edges_removes_redundant_edge() -> None:
    predecessors = {
        1: {0},
        2: {0, 1},
    }
    successors = {
        0: {1, 2},
        1: {2},
    }

    _reduce_transitive_edges(predecessors=predecessors, successors=successors, rule_count=3)

    assert successors[0] == {1}
    assert predecessors[2] == {1}


def test_reduce_transitive_edges_removes_multihop_redundant_edge() -> None:
    predecessors = {
        1: {0},
        2: {1},
        4: {0, 2},
    }
    successors = {
        0: {1, 4},
        1: {2},
        2: {4},
    }

    _reduce_transitive_edges(predecessors=predecessors, successors=successors, rule_count=5)

    assert successors[0] == {1}
    assert predecessors[4] == {2}
