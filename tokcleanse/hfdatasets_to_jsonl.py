# prepare_datasets.py
from datasets import load_dataset
from transformers import AutoTokenizer
import json
from pathlib import Path

OUT = Path("data/hf")
OUT.mkdir(parents=True, exist_ok=True)

DOLCI_CAP = 2_500_000

tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")


def write_jsonl(path: Path, texts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(texts):,} examples to {path}")


def apply_chat_template(messages: list[dict]) -> str | None:
    clean = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("content") is not None and m["content"].strip()
    ]
    if not clean:
        return None
    return tokenizer.apply_chat_template(
        clean,
        tokenize=False,
        add_generation_prompt=False,
    )


def make_turn(prompt: str, target: str) -> str | None:
    return apply_chat_template([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ])


def make_translation_turn(en: str, da: str) -> str | None:
    return apply_chat_template([
        {"role": "user", "content": f"Translate the following English text to Danish.\n\n{en}"},
        {"role": "assistant", "content": da},
    ])


# --- IND: Dolci-Instruct-SFT (capped at DOLCI_CAP) ---
ds = load_dataset("allenai/Dolci-Instruct-SFT")
texts = []
for split in ds.values():
    for ex in split:
        if len(texts) >= DOLCI_CAP:
            break
        # pass messages directly — Dolci already uses {role, content} format
        text = apply_chat_template(ex.get("messages", []))
        if text:
            texts.append(text)
write_jsonl(OUT / "dolci-instruct-ind.jsonl", texts)


# # --- OOD: dynaword-bt ---
# ds = load_dataset("oliverkinch/dynaword-bt")
# texts = []
# for split in ds.values():
#     for ex in split:
#         prompt = ex.get("prompt", "").strip()
#         target = ex.get("target", "").strip()
#         if prompt and target:
#             text = make_turn(prompt, target)
#             if text:
#                 texts.append(text)
# write_jsonl(OUT / "dynaword-bt-ood.jsonl", texts)


# # --- OOD: doab-da-bt ---
# ds = load_dataset("parquet", data_files="hf://datasets/oliverkinch/danish-university-portals-bt/data/train-00000-of-00001.parquet")
# texts = []
# for split in ds.values():
#     for ex in split:
#         prompt = ex.get("prompt", "").strip()
#         target = ex.get("target", "").strip()
#         if prompt and target:
#             text = make_turn(prompt, target)
#             if text:
#                 texts.append(text)
# write_jsonl(OUT / "doab-da-bt-ood.jsonl", texts)


# # --- OOD: europarl da-en (as translation instructions) ---
# ds = load_dataset("Helsinki-NLP/europarl", "da-en")
# texts = []
# for split in ds.values():
#     for ex in split:
#         translation = ex.get("translation", {})
#         en = translation.get("en", "").strip()
#         da = translation.get("da", "").strip()
#         if en and da:
#             text = make_translation_turn(en, da)
#             if text:
#                 texts.append(text)
# write_jsonl(OUT / "europarl-da-ood.jsonl", texts)