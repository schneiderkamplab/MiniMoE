#!/usr/bin/env python
"""Plot OdinMoE training metrics from a JSONL log."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any


_CACHE_DIR = Path(tempfile.gettempdir()) / "tokcleanse-plot-cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_METRICS_PATH = Path("models/odin-moe-trained/training_metrics.jsonl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics_path",
        nargs="?",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help=f"Path to training_metrics.jsonl (default: {DEFAULT_METRICS_PATH})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PNG path (default: <metrics_dir>/training_metrics_plot.png)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Rolling average window for noisy loss curves (default: 5)",
    )
    parser.add_argument(
        "--title",
        default="OdinMoE Training Metrics",
        help="Figure title",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"No metrics found in {path}")
    return rows


def _metric(rows: list[dict[str, Any]], key: str) -> list[float | None]:
    return [row.get(key) for row in rows]


def _rolling(values: list[float | None], window: int) -> list[float | None]:
    if window <= 1:
        return values
    averaged: list[float | None] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        window_values = [value for value in values[start : index + 1] if value is not None]
        averaged.append(mean(window_values) if window_values else None)
    return averaged


def _plot_metric(
    ax: Any,
    steps: list[int],
    rows: list[dict[str, Any]],
    key: str,
    label: str,
    *,
    color: str,
    rolling_window: int | None = None,
) -> None:
    values = _metric(rows, key)
    if not any(value is not None for value in values):
        return
    ax.plot(steps, values, label=label, color=color, linewidth=1.2, alpha=0.38)
    if rolling_window and rolling_window > 1 and len(values) >= 3:
        ax.plot(
            steps,
            _rolling(values, rolling_window),
            label=f"{label} {rolling_window}-step avg",
            color=color,
            linewidth=2.2,
        )


def _plot_eval(ax: Any, rows: list[dict[str, Any]]) -> None:
    eval_points = [
        (row["step"], row["eval_lm_loss"])
        for row in rows
        if row.get("eval_lm_loss") is not None
    ]
    if not eval_points:
        return
    steps, values = zip(*eval_points)
    ax.scatter(steps, values, label="eval_lm_loss", s=42, zorder=5, color="black")
    ax.plot(steps, values, color="black", linewidth=1.2, alpha=0.7)


def _plot_metrics(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    rolling_window: int,
) -> None:
    steps = [row["step"] for row in rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(title, fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    _plot_metric(ax, steps, rows, "ood_train_loss", "OOD total", color="#b33c2e", rolling_window=rolling_window)
    _plot_metric(ax, steps, rows, "ood_train_task_loss", "OOD task/lm", color="#e27d2f", rolling_window=rolling_window)
    _plot_eval(ax, rows)
    ax.set_title("OOD Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    _plot_metric(ax, steps, rows, "ind_train_loss", "IND total", color="#245e9c", rolling_window=rolling_window)
    _plot_metric(
        ax,
        steps,
        rows,
        "ind_train_task_loss",
        "IND task/distill+lm",
        color="#4aa3a2",
        rolling_window=rolling_window,
    )
    ax.set_title("IND Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    _plot_metric(ax, steps, rows, "ind_router_prob_expert1", "IND expert1 prob", color="#245e9c")
    _plot_metric(ax, steps, rows, "ood_router_prob_expert1", "OOD expert1 prob", color="#b33c2e")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_title("Router Expert-1 Probability")
    ax.set_xlabel("step")
    ax.set_ylabel("mean p(expert1)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    _plot_metric(ax, steps, rows, "route_logit_bias_scale", "bias scale", color="#6d4c8d")
    _plot_metric(ax, steps, rows, "learning_rate", "learning rate", color="#5f8d4e")
    ax2 = ax.twinx()
    route_loss = _metric(rows, "train_route_loss")
    if any(value is not None for value in route_loss):
        ax2.plot(steps, route_loss, label="route loss", color="#c5962b", linewidth=1.8)
    ax.set_title("Bias Anneal, LR, Route Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("bias scale / LR")
    ax2.set_ylabel("route loss")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    ax = axes[2, 0]
    _plot_metric(ax, steps, rows, "router_expert_weight_diff_mean_abs", "router diff mean abs", color="#6d4c8d")
    _plot_metric(ax, steps, rows, "router_expert_weight_diff_mean_squared", "router diff mean squared", color="#aa7cc2")
    ax.set_title("Router Expert Parameter Separation")
    ax.set_xlabel("step")
    ax.set_ylabel("diff")
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    _plot_metric(ax, steps, rows, "expert_weight_diff_mean_abs", "expert diff mean abs", color="#2f7d57")
    _plot_metric(ax, steps, rows, "expert_weight_diff_mean_squared", "expert diff mean squared", color="#7aa95c")
    ax.set_title("FFN Expert Parameter Separation")
    ax.set_xlabel("step")
    ax.set_ylabel("diff")
    ax.legend(fontsize=8)

    for ax in axes.ravel():
        ax.grid(True, alpha=0.28)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    metrics_path = args.metrics_path
    output_path = args.output or metrics_path.with_name("training_metrics_plot.png")
    rows = _load_rows(metrics_path)
    _plot_metrics(rows, output_path, title=args.title, rolling_window=args.rolling_window)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
