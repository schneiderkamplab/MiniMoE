from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tokcleanse.upcycle import (
    _render_upcycle_brainsurgery_plan,
    _rewrite_upcycled_config,
    _rewrite_upcycled_tokenizer_config,
    upcycle_gemma4_model,
)


def test_render_upcycle_brainsurgery_plan_includes_transform(tmp_path: Path) -> None:
    checkpoint_input_path = tmp_path / "source" / "model.safetensors"
    output_checkpoint_path = tmp_path / "destination" / "model.safetensors"

    rendered = _render_upcycle_brainsurgery_plan(
        checkpoint_input_path=checkpoint_input_path,
        output_checkpoint_path=output_checkpoint_path,
        num_experts=3,
        expert_intermediate_size=6,
        expert_init="copy",
    )

    assert "upcycle_gemma4_dense_to_moe" in rendered
    assert 'alias: "model"' in rendered
    assert "num_experts: 3" in rendered
    assert "expert_intermediate_size: 6" in rendered
    assert 'init: "copy"' in rendered
    assert str(output_checkpoint_path) in rendered


def test_rewrite_upcycled_config_updates_text_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "text_config": {
                    "model_type": "gemma4_text",
                    "enable_moe_block": False,
                    "intermediate_size": 6144,
                    "num_experts": None,
                    "top_k_experts": None,
                    "moe_intermediate_size": None,
                    "expert_intermediate_size": None,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    expert_intermediate_size, strip_multimodal = _rewrite_upcycled_config(
        config_path,
        num_experts=2,
        top_k_experts=1,
        expert_init="copy",
    )

    rewritten = json.loads(config_path.read_text(encoding="utf-8"))
    assert expert_intermediate_size == 6144
    assert strip_multimodal is True
    assert "text_config" not in rewritten
    assert "enable_moe_block" not in rewritten
    assert rewritten["num_experts"] == 2
    assert rewritten["top_k_experts"] == 1
    assert rewritten["moe_intermediate_size"] == 6144
    assert rewritten["expert_intermediate_size"] == 6144


def test_rewrite_upcycled_config_uses_double_wide_expert_size(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "text_config": {
                    "model_type": "gemma4_text",
                    "enable_moe_block": False,
                    "intermediate_size": 6144,
                    "num_experts": None,
                    "top_k_experts": None,
                    "moe_intermediate_size": None,
                    "expert_intermediate_size": None,
                    "use_double_wide_mlp": True,
                    "num_kv_shared_layers": 20,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    expert_intermediate_size, strip_multimodal = _rewrite_upcycled_config(
        config_path,
        num_experts=1,
        top_k_experts=1,
        expert_init="copy",
    )

    rewritten = json.loads(config_path.read_text(encoding="utf-8"))
    assert expert_intermediate_size == 12288
    assert strip_multimodal is True
    assert "text_config" not in rewritten
    assert rewritten["moe_intermediate_size"] == 12288
    assert rewritten["expert_intermediate_size"] == 12288


def test_rewrite_upcycled_config_marks_odin_moe_model_type(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "text_config": {
                    "model_type": "gemma4_text",
                    "enable_moe_block": False,
                    "intermediate_size": 6144,
                    "num_experts": None,
                    "top_k_experts": None,
                    "moe_intermediate_size": None,
                    "expert_intermediate_size": None,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _rewrite_upcycled_config(
        config_path,
        num_experts=1,
        top_k_experts=1,
        expert_init="copy",
    )

    rewritten = json.loads(config_path.read_text(encoding="utf-8"))
    assert rewritten["model_type"] == "OdinMoE"
    assert rewritten["architectures"] == ["OdinMOEForCausalLM"]
    assert "text_config" not in rewritten


def test_rewrite_upcycled_tokenizer_config_marks_odin_moe_tokenizer(tmp_path: Path) -> None:
    tokenizer_config_path = tmp_path / "tokenizer_config.json"
    tokenizer_config_path.write_text(
        json.dumps(
            {
                "tokenizer_class": "GemmaTokenizer",
                "processor_class": "Gemma4Processor",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    _rewrite_upcycled_tokenizer_config(tokenizer_config_path)

    rewritten = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    assert rewritten["tokenizer_class"] == "OdinMoETokenizerFast"
    assert rewritten["fix_mistral_regex"] is True
    assert "processor_class" not in rewritten


def test_upcycle_gemma4_model_creates_moe_tensors(tmp_path: Path) -> None:
    if "/envs/tokclean/" not in Path(sys.executable).as_posix():
        pytest.skip("brainsurgery checkpoint rewrite is validated in the tokclean conda env")

    pytest.importorskip("brainsurgery")
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_text",
                "enable_moe_block": False,
                "intermediate_size": 3,
                "num_experts": None,
                "top_k_experts": None,
                "moe_intermediate_size": None,
                "expert_intermediate_size": None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    gate = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    up = torch.arange(10, 16, dtype=torch.float32).reshape(3, 2)
    down = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    pre_ff_norm = torch.ones(2, dtype=torch.float32)
    post_ff_norm = torch.ones(2, dtype=torch.float32)
    safetensors_torch.save_file(
        {
            "model.layers.0.mlp.gate_proj.weight": gate,
            "model.layers.0.mlp.up_proj.weight": up,
            "model.layers.0.mlp.down_proj.weight": down,
            "model.layers.0.pre_feedforward_layernorm.weight": pre_ff_norm,
            "model.layers.0.post_feedforward_layernorm.weight": post_ff_norm,
        },
        str(source / "model.safetensors"),
    )

    upcycled_path = upcycle_gemma4_model(
        source,
        destination,
        models_dir=tmp_path,
        overwrite=False,
        num_experts=2,
        top_k_experts=1,
        expert_init="copy",
    )

    assert upcycled_path == destination.resolve()
    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "OdinMoE"
    assert "enable_moe_block" not in config
    assert "text_config" not in config
    assert "audio_config" not in config
    assert "vision_config" not in config
    assert config["num_experts"] == 2
    assert config["top_k_experts"] == 1

    rewritten = safetensors_torch.load_file(str(destination / "model.safetensors"))
    assert not any(".mlp." in name for name in rewritten)
    assert not any(name.startswith("model.language_model.") for name in rewritten)
    assert not any(name.startswith("model.audio_tower.") for name in rewritten)
    assert not any(name.startswith("model.vision_tower.") for name in rewritten)
    assert torch.equal(
        rewritten["model.layers.0.experts.gate_up_proj"],
        torch.stack((torch.cat((gate, up), dim=0), torch.cat((gate, up), dim=0))),
    )
    assert torch.equal(
        rewritten["model.layers.0.experts.down_proj"],
        torch.stack((down, down)),
    )
    assert torch.equal(
        rewritten["model.layers.0.router.proj.weight"],
        torch.zeros((2, 2), dtype=torch.float32),
    )
    assert torch.equal(
        rewritten["model.layers.0.router.scale"],
        torch.ones(2, dtype=torch.float32),
    )
    assert torch.equal(
        rewritten["model.layers.0.router.per_expert_scale"],
        torch.ones(2, dtype=torch.float32),
    )
 

def test_upcycle_gemma4_model_zero_init_zeroes_expert_tensors(tmp_path: Path) -> None:
    if "/envs/tokclean/" not in Path(sys.executable).as_posix():
        pytest.skip("brainsurgery checkpoint rewrite is validated in the tokclean conda env")

    pytest.importorskip("brainsurgery")
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_text",
                "enable_moe_block": False,
                "intermediate_size": 3,
                "num_experts": None,
                "top_k_experts": None,
                "moe_intermediate_size": None,
                "expert_intermediate_size": None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    gate = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    up = torch.arange(10, 16, dtype=torch.float32).reshape(3, 2)
    down = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    pre_ff_norm = torch.tensor([3.0, 5.0], dtype=torch.float32)
    post_ff_norm = torch.tensor([7.0, 11.0], dtype=torch.float32)
    safetensors_torch.save_file(
        {
            "model.layers.0.mlp.gate_proj.weight": gate,
            "model.layers.0.mlp.up_proj.weight": up,
            "model.layers.0.mlp.down_proj.weight": down,
            "model.layers.0.pre_feedforward_layernorm.weight": pre_ff_norm,
            "model.layers.0.post_feedforward_layernorm.weight": post_ff_norm,
        },
        str(source / "model.safetensors"),
    )

    upcycle_gemma4_model(
        source,
        destination,
        models_dir=tmp_path,
        overwrite=False,
        num_experts=2,
        top_k_experts=2,
        expert_init="zero",
    )

    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "OdinMoE"
    assert config["architectures"] == ["OdinMOEForCausalLM"]

    rewritten = safetensors_torch.load_file(str(destination / "model.safetensors"))
    assert torch.count_nonzero(rewritten["model.layers.0.experts.gate_up_proj"]) == 0
    assert torch.count_nonzero(rewritten["model.layers.0.experts.down_proj"]) == 0


def test_upcycle_gemma4_model_pads_narrow_dense_layers_to_expert_width(tmp_path: Path) -> None:
    if "/envs/tokclean/" not in Path(sys.executable).as_posix():
        pytest.skip("brainsurgery checkpoint rewrite is validated in the tokclean conda env")

    pytest.importorskip("brainsurgery")
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_text",
                "enable_moe_block": False,
                "intermediate_size": 3,
                "num_experts": None,
                "top_k_experts": None,
                "moe_intermediate_size": None,
                "expert_intermediate_size": None,
                "use_double_wide_mlp": True,
                "num_kv_shared_layers": 1,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    gate = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    up = torch.arange(10, 16, dtype=torch.float32).reshape(3, 2)
    down = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    safetensors_torch.save_file(
        {
            "model.layers.0.mlp.gate_proj.weight": gate,
            "model.layers.0.mlp.up_proj.weight": up,
            "model.layers.0.mlp.down_proj.weight": down,
            "model.layers.0.pre_feedforward_layernorm.weight": torch.ones(2, dtype=torch.float32),
            "model.layers.0.post_feedforward_layernorm.weight": torch.ones(2, dtype=torch.float32),
        },
        str(source / "model.safetensors"),
    )

    upcycle_gemma4_model(
        source,
        destination,
        models_dir=tmp_path,
        overwrite=False,
        num_experts=1,
        top_k_experts=1,
        expert_init="copy",
    )

    rewritten = safetensors_torch.load_file(str(destination / "model.safetensors"))
    assert rewritten["model.layers.0.experts.gate_up_proj"].shape == (1, 12, 2)
    assert rewritten["model.layers.0.experts.down_proj"].shape == (1, 2, 6)
    expected_gate = torch.cat((gate, torch.zeros((3, 2), dtype=torch.float32)), dim=0)
    expected_up = torch.cat((up, torch.zeros((3, 2), dtype=torch.float32)), dim=0)
    expected_down = torch.cat((down, torch.zeros((2, 3), dtype=torch.float32)), dim=1)
    assert torch.equal(
        rewritten["model.layers.0.experts.gate_up_proj"][0],
        torch.cat((expected_gate, expected_up), dim=0),
    )
    assert torch.equal(
        rewritten["model.layers.0.experts.down_proj"][0],
        expected_down,
    )
