from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import pytest

from tokcleanse import load_tokenizer_contents, save_reordered_tokenizer

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_MODEL_DIR = _PROJECT_ROOT / "models" / "odin-id"
_DESTINATION_MODEL_DIR = _PROJECT_ROOT / "models" / "odin-alpha"
_PROMPT = "The capital of Denmark is"
_MAX_NEW_TOKENS = 8


def test_reassigned_model_matches_original_generation() -> None:
    if "/envs/tokclean/" not in Path(sys.executable).as_posix():
        pytest.skip("end-to-end generation equivalence is validated in the tokclean conda env")

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    contents = load_tokenizer_contents(_SOURCE_MODEL_DIR)
    save_reordered_tokenizer(
        contents,
        _DESTINATION_MODEL_DIR,
        order_name="good",
        overwrite=True,
        reassign=True,
    )

    mapping = json.loads((_DESTINATION_MODEL_DIR / "tokenizer_mapping.json").read_text())

    def mapped_ids(ids: list[int]) -> list[int]:
        return [mapping[str(token_id)] for token_id in ids]

    original_tokenizer = transformers.AutoTokenizer.from_pretrained(_SOURCE_MODEL_DIR)
    original_model = transformers.AutoModelForCausalLM.from_pretrained(
        _SOURCE_MODEL_DIR,
        torch_dtype="auto",
    )
    original_inputs = original_tokenizer(_PROMPT, return_tensors="pt")
    with torch.inference_mode():
        original_output = original_model.generate(
            **original_inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,
        )
    original_output_ids = original_output[0].tolist()
    original_text = original_tokenizer.decode(original_output[0], skip_special_tokens=True)
    del original_model
    gc.collect()

    reassigned_tokenizer = transformers.AutoTokenizer.from_pretrained(_DESTINATION_MODEL_DIR)
    reassigned_model = transformers.AutoModelForCausalLM.from_pretrained(
        _DESTINATION_MODEL_DIR,
        torch_dtype="auto",
    )
    reassigned_inputs = reassigned_tokenizer(_PROMPT, return_tensors="pt")
    with torch.inference_mode():
        reassigned_output = reassigned_model.generate(
            **reassigned_inputs,
            max_new_tokens=_MAX_NEW_TOKENS,
            do_sample=False,
        )
    reassigned_output_ids = reassigned_output[0].tolist()
    reassigned_text = reassigned_tokenizer.decode(
        reassigned_output[0], skip_special_tokens=True
    )

    assert mapped_ids(original_inputs["input_ids"][0].tolist()) == reassigned_inputs[
        "input_ids"
    ][0].tolist()
    assert mapped_ids(original_output_ids) == reassigned_output_ids
    assert original_text == reassigned_text
