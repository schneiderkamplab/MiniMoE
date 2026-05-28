# prepare_datasets.py
from datasets import load_dataset
import json
from pathlib import Path

OUT = Path("data/hf")
OUT.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, texts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(texts):,} examples to {path}")


# --- inspect field names before writing ---
def inspect(name: str, config: str | None = None) -> None:
    ds = load_dataset(name, config) if config else load_dataset(name)
    print(f"\n=== {name} {config or ''} ===")
    print(ds)
    for split_name, split in ds.items():
        print(f"  [{split_name}] first example:", next(iter(split)))
        break


inspect("oliverkinch/dynaword-bt")
inspect("oliverkinch/doab-da-bt")
#inspect("oliverkinch/instruct-bt")
inspect("allenai/Dolci-Instruct-SFT")
inspect("Helsinki-NLP/europarl", "da-en")