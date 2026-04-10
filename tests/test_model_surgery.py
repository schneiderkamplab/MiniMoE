from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from tokcleanse.model_surgery import (
    _render_brainsurgery_plan,
    rewrite_reassigned_model,
)


def test_render_brainsurgery_plan_includes_requested_tensors(tmp_path: Path) -> None:
    checkpoint_input_path = tmp_path / "source" / "model.safetensors"
    output_checkpoint_path = tmp_path / "destination" / "model.safetensors"
    mapping_path = tmp_path / "destination" / "tokenizer_mapping.json"

    rendered = _render_brainsurgery_plan(
        checkpoint_input_path=checkpoint_input_path,
        output_checkpoint_path=output_checkpoint_path,
        mapping_path=mapping_path,
        embedding_weight_names=("model.embed.weight",),
        lm_head_weight_names=("lm_head.weight",),
    )

    assert "reindex_token_ids" in rendered
    assert "model::model.embed.weight" in rendered
    assert "model::lm_head.weight" in rendered
    assert "keep_unmapped: false" in rendered
    assert str(mapping_path) in rendered
    assert str(output_checkpoint_path) in rendered


def test_rewrite_reassigned_model_returns_none_without_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (destination / "tokenizer_mapping.json").write_text('{"0": 0}\n', encoding="utf-8")

    rewritten = rewrite_reassigned_model(
        source,
        destination,
        embedding_weight_names=("model.embed.weight",),
        lm_head_weight_names=(),
    )

    assert rewritten is None


def test_rewrite_reassigned_model_reindexes_checkpoint(tmp_path: Path) -> None:
    if "/envs/tokclean/" not in Path(sys.executable).as_posix():
        pytest.skip("brainsurgery checkpoint rewrite is validated in the tokclean conda env")

    brainsurgery = pytest.importorskip("brainsurgery")
    del brainsurgery
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()

    embedding = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    lm_head = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    safetensors_torch.save_file(
        {
            "model.embed.weight": embedding,
            "lm_head.weight": lm_head,
        },
        str(source / "model.safetensors"),
    )
    shutil.copy2(source / "model.safetensors", destination / "model.safetensors")
    (destination / "tokenizer_mapping.json").write_text(
        json.dumps({"0": 1, "1": 0, "2": 2}, indent=2) + "\n",
        encoding="utf-8",
    )

    rewritten_path = rewrite_reassigned_model(
        source,
        destination,
        embedding_weight_names=("model.embed.weight",),
        lm_head_weight_names=("lm_head.weight",),
    )

    assert rewritten_path == (destination / "model.safetensors")
    rewritten = safetensors_torch.load_file(str(rewritten_path))
    assert torch.equal(
        rewritten["model.embed.weight"],
        embedding[torch.tensor([1, 0, 2])],
    )
    assert torch.equal(
        rewritten["lm_head.weight"],
        lm_head[:, torch.tensor([1, 0, 2])],
    )
