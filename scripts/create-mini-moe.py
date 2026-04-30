#!/usr/bin/env python
"""Create a freshly initialized small OdinMoE checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tokcleanse.odin_moe import OdinMOEForCausalLM
from tokcleanse.odin_moe_config import OdinMoETextConfig
from tokcleanse.odin_moe_tokenizer import tokenizer_class_name_for_checkpoint


DEFAULT_SOURCE = Path("models/odin-danish")
DEFAULT_DESTINATION = Path("models/mini-moe")
DEFAULT_SEED = 42
DEFAULT_MAX_PARAMETERS = 100_000_000
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_INTERMEDIATE_SIZE = 512
DEFAULT_NUM_ATTENTION_HEADS = 4
DEFAULT_NUM_KEY_VALUE_HEADS = 1
DEFAULT_HEAD_DIM = 64
DEFAULT_GLOBAL_HEAD_DIM = 64
DEFAULT_NUM_EXPERTS = 2
DEFAULT_TOP_K_EXPERTS = 2
DEFAULT_DTYPE = "bfloat16"

_AUXILIARY_FILENAMES = (
    "README.md",
    "chat_template.jinja",
    "generation_config.json",
    "original_tokenizer.json",
    "original_tokenizer_config.json",
    "tokenizer.json",
    "tokenizer_added_token_initializers.json",
    "tokenizer_config.json",
    "tokenizer_mapping.json",
    "tokenizer_token_groups.json",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help=f"Tokenizer/config source directory (default: {DEFAULT_SOURCE})")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION, help=f"Output checkpoint directory (default: {DEFAULT_DESTINATION})")
    parser.add_argument("--overwrite", action="store_true", help="Remove the destination first if it already exists.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random initialization seed (default: {DEFAULT_SEED})")
    parser.add_argument("--max-parameters", type=int, default=DEFAULT_MAX_PARAMETERS, help=f"Maximum unique parameter count (default: {DEFAULT_MAX_PARAMETERS})")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE, help=f"Transformer hidden size (default: {DEFAULT_HIDDEN_SIZE})")
    parser.add_argument("--intermediate-size", type=int, default=DEFAULT_INTERMEDIATE_SIZE, help=f"Dense metadata intermediate size (default: {DEFAULT_INTERMEDIATE_SIZE})")
    parser.add_argument("--moe-intermediate-size", type=int, default=DEFAULT_INTERMEDIATE_SIZE, help=f"Per-expert intermediate size (default: {DEFAULT_INTERMEDIATE_SIZE})")
    parser.add_argument("--num-attention-heads", type=int, default=DEFAULT_NUM_ATTENTION_HEADS, help=f"Attention heads (default: {DEFAULT_NUM_ATTENTION_HEADS})")
    parser.add_argument("--num-key-value-heads", type=int, default=DEFAULT_NUM_KEY_VALUE_HEADS, help=f"KV heads (default: {DEFAULT_NUM_KEY_VALUE_HEADS})")
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM, help=f"Head dimension (default: {DEFAULT_HEAD_DIM})")
    parser.add_argument("--global-head-dim", type=int, default=DEFAULT_GLOBAL_HEAD_DIM, help=f"Full-attention head dimension (default: {DEFAULT_GLOBAL_HEAD_DIM})")
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS, help=f"Number of experts (default: {DEFAULT_NUM_EXPERTS})")
    parser.add_argument("--top-k-experts", type=int, default=DEFAULT_TOP_K_EXPERTS, help=f"Routed experts per token (default: {DEFAULT_TOP_K_EXPERTS})")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default=DEFAULT_DTYPE,
        help=f"Saved weight dtype (default: {DEFAULT_DTYPE})",
    )
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _load_config(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing source config: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    text_config = data.get("text_config")
    if isinstance(text_config, dict):
        return text_config
    return data


def _build_config(source_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    _validate_dimensions(args)
    config = dict(source_config)
    vocab_size = config.get("vocab_size")
    if not isinstance(vocab_size, int) or vocab_size < 1:
        raise ValueError("Source config must define a positive integer vocab_size")
    config["architectures"] = ["OdinMOEForCausalLM"]
    config["model_type"] = "OdinMoE"
    config["dtype"] = args.dtype
    config["hidden_size"] = args.hidden_size
    config["intermediate_size"] = args.intermediate_size
    config["moe_intermediate_size"] = args.moe_intermediate_size
    config["expert_intermediate_size"] = args.moe_intermediate_size
    config["num_attention_heads"] = args.num_attention_heads
    config["num_key_value_heads"] = args.num_key_value_heads
    config["head_dim"] = args.head_dim
    config["global_head_dim"] = args.global_head_dim
    config["num_experts"] = args.num_experts
    config["top_k_experts"] = args.top_k_experts
    config["hidden_size_per_layer_input"] = 0
    config["vocab_size_per_layer_input"] = vocab_size
    config["tie_word_embeddings"] = True
    config.pop("enable_moe_block", None)
    return config


def _validate_dimensions(args: argparse.Namespace) -> None:
    if args.hidden_size < 1:
        raise ValueError("--hidden-size must be positive")
    if args.intermediate_size < 1:
        raise ValueError("--intermediate-size must be positive")
    if args.moe_intermediate_size < 1:
        raise ValueError("--moe-intermediate-size must be positive")
    if args.num_attention_heads < 1:
        raise ValueError("--num-attention-heads must be positive")
    if args.num_key_value_heads < 1:
        raise ValueError("--num-key-value-heads must be positive")
    if args.num_experts < 1:
        raise ValueError("--num-experts must be positive")
    if args.top_k_experts < 1 or args.top_k_experts > args.num_experts:
        raise ValueError("--top-k-experts must be between 1 and --num-experts")
    if args.num_attention_heads % args.num_key_value_heads != 0:
        raise ValueError("--num-attention-heads must be divisible by --num-key-value-heads")
    if args.num_attention_heads * args.head_dim != args.hidden_size:
        raise ValueError("--num-attention-heads * --head-dim must equal --hidden-size")
    if args.num_attention_heads * args.global_head_dim != args.hidden_size:
        raise ValueError("--num-attention-heads * --global-head-dim must equal --hidden-size")


def _prepare_destination(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for filename in _AUXILIARY_FILENAMES:
        source_path = source / filename
        if source_path.exists():
            shutil.copy2(source_path, destination / filename)


def _write_tokenizer_config(destination: Path) -> None:
    path = destination / "tokenizer_config.json"
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    data["tokenizer_class"] = tokenizer_class_name_for_checkpoint(checkpoint_dir=destination)
    data["fix_mistral_regex"] = True
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unique_parameter_count(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        identifier = id(parameter)
        if identifier in seen:
            continue
        seen.add(identifier)
        total += parameter.numel()
    return total


def main() -> None:
    args = _parse_args()
    source = args.source.expanduser()
    destination = args.destination.expanduser()

    source_config = _load_config(source)
    mini_config = _build_config(source_config, args)

    _prepare_destination(source, destination, overwrite=args.overwrite)
    _write_tokenizer_config(destination)
    (destination / "config.json").write_text(
        json.dumps(mini_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _set_seed(args.seed)
    config = OdinMoETextConfig.from_dict(mini_config)
    model = OdinMOEForCausalLM(config)
    model.tie_weights()
    parameter_count = _unique_parameter_count(model)
    if parameter_count > args.max_parameters:
        shutil.rmtree(destination)
        raise SystemExit(
            f"Refusing to save {parameter_count:,} parameters; "
            f"limit is {args.max_parameters:,}. Use smaller dimensions or raise --max-parameters."
        )
    model.to(dtype=_torch_dtype(args.dtype))
    model.save_pretrained(destination, safe_serialization=True)

    print(f"Saved mini OdinMoE to {destination.resolve()}")
    print(f"parameters={parameter_count:,}")
    print(
        "shape="
        f"layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"heads={config.num_attention_heads} head_dim={config.head_dim} "
        f"experts={config.num_experts} top_k={config.top_k_experts} "
        f"moe_intermediate={config.moe_intermediate_size}"
    )


if __name__ == "__main__":
    main()
