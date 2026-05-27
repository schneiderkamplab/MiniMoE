import json
import subprocess
import sys
from pathlib import Path


def extract_metrics(eval_path: str) -> dict:
    eval_file = Path(eval_path)
    stem = eval_file.stem
    parts = stem.split("_")
    model = parts[0] if len(parts) > 0 else "unknown"
    dataset = "_".join(parts[1:]) if len(parts) > 1 else "unknown"

    result = subprocess.run(
        ["python", "-m", "inspect_ai", "log", "dump", eval_path],
        capture_output=True,
        text=True,
    )

    if not result.stdout.strip():
        print(f"ERROR: Empty output from log dump. stderr: {result.stderr}")
        raise ValueError(f"Empty output for {eval_path}")

    log = json.loads(result.stdout)

    metrics = {}
    for scorer in log.get("results", {}).get("scores", []):
        scorer_name = scorer.get("name", "unknown")
        for name, metric in scorer.get("metrics", {}).items():
            metrics[f"{scorer_name}_{name}"] = metric["value"]

    return {"model": model, "dataset": dataset, "metrics": metrics}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_metrics.py <path-to.eval> [output.json]")
        sys.exit(1)

    eval_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "metrics.json"

    output = extract_metrics(eval_path)

    existing = []
    if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
        with open(output_path) as f:
            existing = json.load(f)

    existing.append(output)
    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Saved to {output_path}:")
    print(json.dumps(output, indent=2))