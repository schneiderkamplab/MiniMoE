from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment

from tokcleanse import (
    TokenizerContents,
    clean,
    load_tokenizer_contents,
    save_reordered_tokenizer,
    topological_sort_rules,
)
from tokcleanse.saver import _SpecialTokenRewritePlan, _rewrite_chat_template_files


def test_good_order_reorders_sample_tokenizer(sample_tokenizer_contents: TokenizerContents) -> None:
    ordered_rules = topological_sort_rules(sample_tokenizer_contents, order_name="good")

    assert [rule.index for rule in ordered_rules] == [1, 0, 2, 3]
    assert clean(sample_tokenizer_contents, mode="literal", order_name="good") == [
        ("a", "b"),
        ("x", "y"),
        ("ab", "c"),
        ("xy", "z"),
    ]


def test_save_reordered_tokenizer_copies_directory_and_rewrites_merges(
    sample_tokenizer_contents: TokenizerContents,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "saved-tokenizer"

    saved_path = save_reordered_tokenizer(
        sample_tokenizer_contents,
        destination,
        order_name="good",
    )

    assert saved_path == destination.resolve()
    assert (destination / "tokenizer_config.json").exists()

    tokenizer_data = json.loads((destination / "tokenizer.json").read_text(encoding="utf-8"))
    assert tokenizer_data["model"]["merges"] == [
        ["a", "b"],
        ["x", "y"],
        ["ab", "c"],
        ["xy", "z"],
    ]

    merges_txt_lines = (destination / "merges.txt").read_text(encoding="utf-8").splitlines()
    assert merges_txt_lines == [
        "#version: 0.2",
        "a b",
        "x y",
        "ab c",
        "xy z",
    ]


def test_save_reordered_tokenizer_with_reassign_writes_mapping_and_backups(
    sample_tokenizer_contents: TokenizerContents,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "saved-tokenizer-reassigned"

    save_reordered_tokenizer(
        sample_tokenizer_contents,
        destination,
        order_name="good",
        reassign=True,
    )

    tokenizer_data = json.loads((destination / "tokenizer.json").read_text(encoding="utf-8"))
    assert tokenizer_data["model"]["vocab"] == {
        "<pad>": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "x": 4,
        "y": 5,
        "z": 6,
        "ab": 7,
        "xy": 8,
        "abc": 9,
        "xyz": 10,
    }

    mapping_data = json.loads((destination / "tokenizer_mapping.json").read_text(encoding="utf-8"))
    assert mapping_data == {
        "0": 0,
        "1": 4,
        "2": 5,
        "3": 1,
        "4": 2,
        "5": 3,
        "6": 6,
        "7": 8,
        "8": 7,
        "9": 9,
        "10": 10,
    }

    original_tokenizer_data = json.loads(
        (destination / "original_tokenizer.json").read_text(encoding="utf-8")
    )
    assert original_tokenizer_data["model"]["vocab"]["xy"] == 7
    assert original_tokenizer_data["model"]["vocab"]["ab"] == 8

    original_tokenizer_config = json.loads(
        (destination / "original_tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert original_tokenizer_config["pad_token"] == "<pad>"


def test_save_reordered_tokenizer_with_reassign_updates_added_token_ids(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-tokenizer"
    source.mkdir()
    (source / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {
                        "id": 5,
                        "content": "<pad>",
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    },
                    {
                        "id": 7,
                        "content": "<bos>",
                        "single_word": False,
                        "lstrip": False,
                        "rstrip": False,
                        "normalized": False,
                        "special": True,
                    },
                ],
                "post_processor": {
                    "type": "TemplateProcessing",
                    "special_tokens": {
                        "<bos>": {
                            "id": 7,
                            "ids": [7],
                            "tokens": ["<bos>"],
                        }
                    },
                },
                "model": {
                    "type": "BPE",
                    "vocab": {
                        "a": 0,
                        "b": 1,
                        "ab": 2,
                        "x": 3,
                        "y": 4,
                        "<pad>": 5,
                        "<bos>": 7,
                    },
                    "merges": [["a", "b"]],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "pad_token_id": 5,
                "bos_token_id": 7,
                "added_tokens_decoder": {
                    "5": {"content": "<pad>", "id": 5, "special": True},
                    "7": {"content": "<bos>", "id": 7, "special": True},
                },
                "added_tokens_encoder": {
                    "<pad>": 5,
                    "<bos>": 7,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "added_tokens.json").write_text(
        json.dumps({"<pad>": 5, "<bos>": 7}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    destination = tmp_path / "rewritten-tokenizer"
    contents = load_tokenizer_contents(source)

    save_reordered_tokenizer(
        contents,
        destination,
        reassign=True,
    )

    tokenizer_data = json.loads((destination / "tokenizer.json").read_text(encoding="utf-8"))
    assert tokenizer_data["model"]["vocab"] == {
        "<pad>": 0,
        "<bos>": 1,
        "a": 2,
        "b": 3,
        "x": 4,
        "y": 5,
        "ab": 6,
    }
    assert tokenizer_data["added_tokens"] == [
        {
            "id": 0,
            "content": "<pad>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
        {
            "id": 1,
            "content": "<bos>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        },
    ]
    assert tokenizer_data["post_processor"]["special_tokens"] == {
        "<bos>": {"id": 1, "ids": [1], "tokens": ["<bos>"]}
    }

    tokenizer_config = json.loads((destination / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert tokenizer_config["pad_token_id"] == 0
    assert tokenizer_config["bos_token_id"] == 1
    assert tokenizer_config["added_tokens_decoder"] == {
        "0": {"content": "<pad>", "id": 0, "special": True},
        "1": {"content": "<bos>", "id": 1, "special": True},
    }
    assert tokenizer_config["added_tokens_encoder"] == {
        "<pad>": 0,
        "<bos>": 1,
    }

    added_tokens = json.loads((destination / "added_tokens.json").read_text(encoding="utf-8"))
    assert added_tokens == {"<pad>": 0, "<bos>": 1}


def test_save_reordered_tokenizer_with_special_token_map_drops_and_renames_specials(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-tokenizer"
    source.mkdir()
    (source / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 5, "content": "<pad>", "special": True},
                    {"id": 6, "content": "<bos>", "special": True},
                    {"id": 7, "content": "<eos>", "special": True},
                ],
                "post_processor": {
                    "type": "TemplateProcessing",
                    "special_tokens": {
                        "<bos>": {"id": 6, "ids": [6], "tokens": ["<bos>"]},
                        "<eos>": {"id": 7, "ids": [7], "tokens": ["<eos>"]},
                    },
                },
                "model": {
                    "type": "BPE",
                    "vocab": {
                        "a": 0,
                        "b": 1,
                        "ab": 2,
                        "x": 3,
                        "y": 4,
                        "<pad>": 5,
                        "<bos>": 6,
                        "<eos>": 7,
                    },
                    "merges": [["a", "b"]],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token_id": 5,
                "bos_token_id": 6,
                "eos_token_id": 7,
                "added_tokens_decoder": {
                    "5": {"content": "<pad>", "id": 5, "special": True},
                    "6": {"content": "<bos>", "id": 6, "special": True},
                    "7": {"content": "<eos>", "id": 7, "special": True},
                },
                "added_tokens_encoder": {
                    "<pad>": 5,
                    "<bos>": 6,
                    "<eos>": 7,
                },
                "chat_template": "<|turn>user\nHello<turn|>\n<|tool_call>call:x{}<tool_call|><|image|>",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "vocab_size": 8,
                "bos_token_id": 6,
                "eos_token_id": [7],
                "pad_token_id": 5,
                "text_config": {"vocab_size": 8, "vocab_size_per_layer_input": 8},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "added_tokens.json").write_text(
        json.dumps({"<pad>": 5, "<bos>": 6, "<eos>": 7}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    destination = tmp_path / "rewritten-tokenizer"
    contents = load_tokenizer_contents(source)

    save_reordered_tokenizer(
        contents,
        destination,
        reassign=True,
        special_token_literal_map={
            "<bos>": None,
            "<eos>": "<end>",
        },
    )

    tokenizer_data = json.loads((destination / "tokenizer.json").read_text(encoding="utf-8"))
    assert tokenizer_data["model"]["vocab"] == {
        "<bos>": 0,
        "<end>": 1,
        "a": 2,
        "b": 3,
        "x": 4,
        "y": 5,
        "ab": 6,
    }
    assert tokenizer_data["added_tokens"] == [
        {"id": 0, "content": "<bos>", "special": True},
        {"id": 1, "content": "<end>", "special": True},
    ]
    assert tokenizer_data["post_processor"]["special_tokens"] == {
        "<bos>": {"id": 0, "ids": [0], "tokens": ["<bos>"]},
        "<end>": {"id": 1, "ids": [1], "tokens": ["<end>"]},
    }

    tokenizer_config = json.loads((destination / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert "pad_token" not in tokenizer_config
    assert "pad_token_id" not in tokenizer_config
    assert tokenizer_config["bos_token"] == "<bos>"
    assert tokenizer_config["eos_token"] == "<end>"
    assert tokenizer_config["bos_token_id"] == 0
    assert tokenizer_config["eos_token_id"] == 1
    assert tokenizer_config["added_tokens_decoder"] == {
        "0": {"content": "<bos>", "id": 0, "special": True},
        "1": {"content": "<end>", "id": 1, "special": True},
    }
    assert tokenizer_config["added_tokens_encoder"] == {
        "<bos>": 0,
        "<end>": 1,
    }

    config_data = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    assert config_data["vocab_size"] == 7
    assert config_data["text_config"]["vocab_size"] == 7
    assert config_data["text_config"]["vocab_size_per_layer_input"] == 7
    assert "pad_token_id" not in config_data
    assert config_data["bos_token_id"] == 0
    assert config_data["eos_token_id"] == [1]

    added_tokens = json.loads((destination / "added_tokens.json").read_text(encoding="utf-8"))
    assert added_tokens == {"<bos>": 0, "<end>": 1}

    mapping_data = json.loads((destination / "tokenizer_mapping.json").read_text(encoding="utf-8"))
    assert mapping_data == {
        "0": 2,
        "1": 3,
        "2": 6,
        "3": 4,
        "4": 5,
        "6": 0,
        "7": 1,
    }


def test_rewrite_chat_template_files_renames_and_drops_literals(tmp_path: Path) -> None:
    destination = tmp_path / "tokenizer"
    destination.mkdir()
    (destination / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": "<|turn>user\nHello<turn|>\n<|tool_call>call:x{}<tool_call|><|image|>",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / "chat_template.jinja").write_text(
        "{{ '<|turn>user\\n' }}{{ '<|tool_call>call:x{}<tool_call|>' }}{{ '<|image|>' }}{{ '<|channel>thought\\n' }}{{ '<channel|>' }}{{ '<|\"|>' }}",
        encoding="utf-8",
    )
    plan = _SpecialTokenRewritePlan(
        replacements={
            "<|tool_call>": "<call>",
            "<tool_call|>": "</call>",
            "<|channel>": "<section>",
            "<channel|>": "</section>",
            '<|\"|>': "<escape>",
            "<|turn>": "<turn>",
            "<turn|>": "</turn>",
        },
        dropped_literals=frozenset({"<|image|>"}),
        dropped_ids=frozenset(),
    )

    _rewrite_chat_template_files(destination, plan)

    tokenizer_config = json.loads((destination / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert tokenizer_config["chat_template"] == "<turn>user\nHello</turn>\n<call>call:x{}</call>"
    chat_template = (destination / "chat_template.jinja").read_text(encoding="utf-8")
    assert "<|image|>" not in chat_template
    assert "<|tool_call>" not in chat_template
    assert "<tool_call|>" not in chat_template
    assert "<|channel>" not in chat_template
    assert "<channel|>" not in chat_template
    assert '<|\"|>' not in chat_template
    assert "item['type'] == 'image'" not in chat_template
    assert "<call>" in chat_template
    assert "</call>" in chat_template
    assert "<section>thought\\n" in chat_template
    assert "</section>" in chat_template
    assert "<escape>" in chat_template


def test_rewrite_chat_template_files_prunes_multimodal_branches_via_ast(tmp_path: Path) -> None:
    destination = tmp_path / "tokenizer"
    destination.mkdir()
    template = (
        "{% for item in items %}"
        "{% if item['type'] == 'text' %}{{ item['text'] }}"
        "{% elif item['type']\n== 'image' %}{{ '<|image|>' }}"
        "{% elif item['type'] == 'audio' %}{{ '<|audio|>' }}"
        "{% endif %}"
        "{% endfor %}"
    )
    (destination / "chat_template.jinja").write_text(template, encoding="utf-8")
    plan = _SpecialTokenRewritePlan(
        replacements={},
        dropped_literals=frozenset({"<|image|>", "<|audio|>"}),
        dropped_ids=frozenset(),
    )

    _rewrite_chat_template_files(destination, plan)

    chat_template = (destination / "chat_template.jinja").read_text(encoding="utf-8")
    assert "item['type'] == 'image'" not in chat_template
    assert "item['type'] == 'audio'" not in chat_template
    assert "<|image|>" not in chat_template
    assert "<|audio|>" not in chat_template

    rendered = Environment().from_string(chat_template).render(
        items=[
            {"type": "text", "text": "hello"},
            {"type": "image"},
            {"type": "audio"},
        ]
    )
    assert rendered == "hello"


def test_rewrite_chat_template_files_preserves_condexpr_without_else(tmp_path: Path) -> None:
    destination = tmp_path / "tokenizer"
    destination.mkdir()
    (destination / "chat_template.jinja").write_text(
        "{% for item in items %}{{ ',' if not loop.last }}{{ item }}{% endfor %}",
        encoding="utf-8",
    )
    plan = _SpecialTokenRewritePlan(
        replacements={},
        dropped_literals=frozenset(),
        dropped_ids=frozenset(),
    )

    _rewrite_chat_template_files(destination, plan)

    chat_template = (destination / "chat_template.jinja").read_text(encoding="utf-8")
    rendered = Environment().from_string(chat_template).render(items=["a", "b", "c"])
    assert rendered == ",a,bc"


def test_rewrite_chat_template_files_preserves_literal_brace_boundaries(tmp_path: Path) -> None:
    destination = tmp_path / "tokenizer"
    destination.mkdir()
    (destination / "chat_template.jinja").write_text(
        "{{ value }}:{% if flag %}x{% endif %}{{ '{' }}{{ value }}",
        encoding="utf-8",
    )
    plan = _SpecialTokenRewritePlan(
        replacements={},
        dropped_literals=frozenset(),
        dropped_ids=frozenset(),
    )

    _rewrite_chat_template_files(destination, plan)

    chat_template = (destination / "chat_template.jinja").read_text(encoding="utf-8")
    rendered = Environment().from_string(chat_template).render(value="k", flag=True)
    assert rendered == "k:x{k"


def test_save_reordered_tokenizer_with_token_map_deletes_cascade_and_adds_tokens(
    sample_tokenizer_contents: TokenizerContents,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "saved-tokenizer-token-map"

    save_reordered_tokenizer(
        sample_tokenizer_contents,
        destination,
        order_name="good",
        reassign=True,
        token_delete_literals=("ab",),
        token_add_literals=("ayz",),
    )

    tokenizer_data = json.loads((destination / "tokenizer.json").read_text(encoding="utf-8"))
    assert tokenizer_data["model"]["vocab"] == {
        "<pad>": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "x": 4,
        "y": 5,
        "z": 6,
        "yz": 7,
        "ayz": 8,
        "xy": 9,
        "xyz": 10,
    }
    assert "added_tokens" not in tokenizer_data

    merges_txt_lines = (destination / "merges.txt").read_text(encoding="utf-8").splitlines()
    assert merges_txt_lines == [
        "#version: 0.2",
        "y z",
        "a yz",
        "x y",
        "xy z",
    ]

    mapping_data = json.loads((destination / "tokenizer_mapping.json").read_text(encoding="utf-8"))
    assert mapping_data == {
        "0": 0,
        "1": 4,
        "2": 5,
        "3": 1,
        "4": 2,
        "5": 3,
        "6": 6,
        "7": 9,
        "10": 10,
    }

    tokenizer_config = json.loads((destination / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert "added_tokens_encoder" not in tokenizer_config
    assert "added_tokens_decoder" not in tokenizer_config


def test_save_reordered_tokenizer_with_token_map_renames_tokens_and_merges(
    sample_tokenizer_contents: TokenizerContents,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "saved-tokenizer-token-rename"

    save_reordered_tokenizer(
        sample_tokenizer_contents,
        destination,
        order_name="good",
        reassign=True,
        token_rename_literals={"ab": "AB"},
    )

    tokenizer_data = json.loads((destination / "tokenizer.json").read_text(encoding="utf-8"))
    assert "ab" not in tokenizer_data["model"]["vocab"]
    assert tokenizer_data["model"]["vocab"]["AB"] == 7
    assert tokenizer_data["model"]["vocab"]["a"] == 1
    assert tokenizer_data["model"]["vocab"]["x"] == 4
    assert tokenizer_data["model"]["merges"] == [
        ["a", "b"],
        ["x", "y"],
        ["AB", "c"],
        ["xy", "z"],
    ]

    merges_txt_lines = (destination / "merges.txt").read_text(encoding="utf-8").splitlines()
    assert merges_txt_lines == [
        "#version: 0.2",
        "a b",
        "x y",
        "AB c",
        "xy z",
    ]
