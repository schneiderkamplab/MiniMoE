# extract_metrics.py
import json
import subprocess
import sys
from pathlib import Path
import zipfile

# def extract_metrics(eval_path: str) -> dict:
#     eval_file = Path(eval_path)

#     stem = eval_file.stem
#     parts = stem.split("_")
#     model = parts[0] if len(parts) > 0 else "unknown"
#     dataset = "_".join(parts[1:]) if len(parts) > 1 else "unknown"

#     result = subprocess.run(
#         ["python", "-m", "inspect_ai", "log", "dump", eval_path],
#         capture_output=True,
#         text=True,
#     )

#     if not result.stdout.strip():
#         print(f"ERROR: Empty output from log dump. stderr: {result.stderr}")
#         raise ValueError(f"Empty output for {eval_path}")

#     log = json.loads(result.stdout)

#     # Extract metrics from results
#     metrics = {}
#     for scorer in log.get("results", {}).get("scores", []):
#         for name, metric in scorer.get("metrics", {}).items():
#             metrics[name] = metric["value"]

#     output = {
#         "model": model,
#         "dataset": dataset,
#         "metrics": metrics,
#     }

#     return output

def extract_metrics(eval_path: str) -> dict:
    eval_file = Path(eval_path)

    stem = eval_file.stem
    parts = stem.split("_")
    model = parts[0] if len(parts) > 0 else "unknown"
    dataset = "_".join(parts[1:]) if len(parts) > 1 else "unknown"

    # .eval files are zip archives containing a JSON file
    try:
        with zipfile.ZipFile(eval_path, 'r') as z:
            names = z.namelist()
            json_file = [n for n in names if n.endswith('.json')][0]
            with z.open(json_file) as f:
                log = json.load(f)
    except zipfile.BadZipFile:
        # Try reading as plain JSON
        with open(eval_path) as f:
            log = json.load(f)

    # Extract metrics from results
    metrics = {}
    for scorer in log.get("results", {}).get("scores", []):
        for name, metric in scorer.get("metrics", {}).items():
            metrics[name] = metric["value"]

    output = {
        "model": model,
        "dataset": dataset,
        "metrics": metrics,
    }

    return output

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_metrics.py <path-to.eval> [output.json]")
        sys.exit(1)

    eval_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "metrics.json"

    output = extract_metrics(eval_path)

    # Load existing results if file exists
    # Load existing results if file exists and is non-empty
    existing = []
    if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
        with open(output_path) as f:
            existing = json.load(f)

    # Append new result
    existing.append(output)

    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Saved to {output_path}:")
    print(json.dumps(output, indent=2))