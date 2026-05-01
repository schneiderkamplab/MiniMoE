#!/usr/bin/env python
"""Plot OdinMoE training metrics from a JSONL log."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Any


_CACHE_DIR = Path(tempfile.gettempdir()) / "tokcleanse-plot-cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib  # noqa: E402


def _requested_matplotlib_backend(argv: list[str]) -> str:
    if "--show" not in argv:
        return "Agg"
    for index, argument in enumerate(argv):
        if argument == "--show-backend" and index + 1 < len(argv):
            return argv[index + 1]
        if argument.startswith("--show-backend="):
            return argument.split("=", 1)[1]
    return "TkAgg"


matplotlib.use(_requested_matplotlib_backend(sys.argv[1:]))
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
        "--average-steps",
        "--rolling-window",
        type=int,
        default=5,
        help="Number of steps in the rolling average for noisy curves (default: 5)",
    )
    parser.add_argument(
        "--outlier-proportion",
        type=float,
        default=0.01,
        help="Total proportion of finite y-values to exclude when autoscaling each axis; use 0 to disable (default: 0.01)",
    )
    parser.add_argument(
        "--title",
        default="OdinMoE Training Metrics",
        help="Figure title",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep running and refresh the output plot whenever new metric rows are appended",
    )
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=2.0,
        help="Seconds between file checks in --watch mode (default: 2.0)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plots in a Matplotlib window and refresh them after each update",
    )
    parser.add_argument(
        "--show-backend",
        default="TkAgg",
        help="Matplotlib backend to use with --show (default: TkAgg)",
    )
    return parser.parse_args()


def _read_rows(path: Path, *, ignore_partial_final_line: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if ignore_partial_final_line and index == len(lines) - 1:
                break
            raise
    return rows


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_rows(path)
    if not rows:
        raise SystemExit(f"No metrics found in {path}")
    return rows


def _metric(rows: list[dict[str, Any]], key: str) -> list[float | None]:
    return [row.get(key) for row in rows]


def _finite_values(values: list[float | None]) -> list[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * max(0.0, min(1.0, fraction))
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * (position - lower_index)


def _trimmed_limits(
    values: list[float | None],
    *,
    outlier_proportion: float,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[float, float] | None:
    finite_values = sorted(_finite_values(values))
    if not finite_values or outlier_proportion <= 0:
        return None
    tail_fraction = min(outlier_proportion, 0.99) / 2.0
    low = _quantile(finite_values, tail_fraction)
    high = _quantile(finite_values, 1.0 - tail_fraction)
    if lower_bound is not None:
        low = max(low, lower_bound)
    if upper_bound is not None:
        high = min(high, upper_bound)
    if low == high:
        padding = max(abs(low) * 0.05, 1e-6)
    else:
        padding = (high - low) * 0.05
    low -= padding
    high += padding
    if lower_bound is not None:
        low = max(low, lower_bound)
    if upper_bound is not None:
        high = min(high, upper_bound)
    if low >= high:
        return None
    return low, high


def _apply_trimmed_ylim(
    ax: Any,
    values: list[float | None],
    *,
    outlier_proportion: float,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> None:
    limits = _trimmed_limits(
        values,
        outlier_proportion=outlier_proportion,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    if limits is not None:
        ax.set_ylim(*limits)


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
    linestyle: str = "-",
    alpha: float = 0.38,
    linewidth: float = 1.2,
) -> None:
    values = _metric(rows, key)
    if not any(value is not None for value in values):
        return
    ax.plot(steps, values, label=label, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)
    if rolling_window and rolling_window > 1 and len(values) >= 3:
        ax.plot(
            steps,
            _rolling(values, rolling_window),
            label=f"{label} {rolling_window}-step avg",
            color=color,
            linewidth=2.2,
        )


def _metric_values(rows: list[dict[str, Any]], *keys: str) -> list[float | None]:
    values: list[float | None] = []
    for key in keys:
        values.extend(_metric(rows, key))
    return values


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


def _render_metrics_figure(
    rows: list[dict[str, Any]],
    fig: Any,
    axes: Any,
    *,
    title: str,
    rolling_window: int,
    outlier_proportion: float,
) -> None:
    if not 0 <= outlier_proportion < 1:
        raise SystemExit("--outlier-proportion must be at least 0 and less than 1")
    steps = [row["step"] for row in rows]

    fig.suptitle(title, fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    _plot_metric(ax, steps, rows, "ind_train_task_loss", "IND task objective", color="#245e9c", rolling_window=rolling_window)
    _plot_metric(ax, steps, rows, "ood_train_task_loss", "OOD task objective", color="#b33c2e", rolling_window=rolling_window)
    _plot_metric(ax, steps, rows, "ind_train_lm_loss", "IND CE", color="#4aa3a2", rolling_window=rolling_window)
    _plot_metric(ax, steps, rows, "ood_train_lm_loss", "OOD CE", color="#e27d2f", rolling_window=rolling_window)
    _plot_eval(ax, rows)
    ax.set_title("IND/OOD Task Loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    _apply_trimmed_ylim(
        ax,
        _metric_values(
            rows,
            "ind_train_task_loss",
            "ood_train_task_loss",
            "ind_train_lm_loss",
            "ood_train_lm_loss",
            "eval_lm_loss",
        ),
        outlier_proportion=outlier_proportion,
    )
    lines, labels = ax.get_legend_handles_labels()
    if lines:
        ax.legend(lines, labels, fontsize=8)

    ax = axes[0, 1]
    _plot_metric(ax, steps, rows, "ind_train_route_loss", "IND route raw", color="#245e9c")
    _plot_metric(ax, steps, rows, "ood_train_route_loss", "OOD route raw", color="#b33c2e")
    _plot_metric(
        ax,
        steps,
        rows,
        "train_router_objective",
        "combined route objective",
        color="#303030",
        linestyle="--",
        alpha=0.9,
        linewidth=1.8,
    )
    _plot_metric(ax, steps, rows, "ind_train_router_objective", "IND route weighted", color="#4aa3a2")
    _plot_metric(ax, steps, rows, "ood_train_router_objective", "OOD route weighted", color="#e27d2f")
    thresholds = _finite_values(_metric(rows, "route_logit_bias_anneal_loss_threshold"))
    if thresholds:
        ax.axhline(thresholds[-1], color="gray", linestyle="--", linewidth=1, alpha=0.7, label="anneal threshold")
    ax.set_title("Router Objective")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    _apply_trimmed_ylim(
        ax,
        _metric_values(
            rows,
            "ind_train_route_loss",
            "ood_train_route_loss",
            "train_router_objective",
            "ind_train_router_objective",
            "ood_train_router_objective",
        ),
        outlier_proportion=outlier_proportion,
        lower_bound=0.0,
    )
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    _plot_metric(ax, steps, rows, "ind_router_prob_expert1", "IND biased p(e1)", color="#245e9c")
    _plot_metric(ax, steps, rows, "ood_router_prob_expert1", "OOD biased p(e1)", color="#b33c2e")
    _plot_metric(ax, steps, rows, "ind_router_unbiased_prob_expert1", "IND weight-only p(e1)", color="#4aa3a2")
    _plot_metric(ax, steps, rows, "ood_router_unbiased_prob_expert1", "OOD weight-only p(e1)", color="#e27d2f")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_title("Router Probability: Weights vs Bias")
    ax.set_xlabel("step")
    ax.set_ylabel("mean p(expert1)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    _plot_metric(ax, steps, rows, "route_logit_bias_scale", "bias scale", color="#6d4c8d")
    offset_steps = _finite_values(_metric(rows, "route_logit_bias_anneal_offset_steps"))
    if offset_steps and offset_steps[-1] > 0:
        ax.axvline(
            offset_steps[-1],
            color="#6f6f6f",
            linestyle=":",
            linewidth=1.0,
            label="anneal offset",
        )
    ax2 = ax.twinx()
    _plot_metric(ax2, steps, rows, "ind_route_logit_bias", "IND route bias", color="#245e9c")
    _plot_metric(ax2, steps, rows, "ood_route_logit_bias", "OOD route bias", color="#b33c2e")
    ax.set_title("Route Bias Anneal")
    ax.set_xlabel("step")
    ax.set_ylabel("bias scale")
    ax2.set_ylabel("route-logit bias")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    _apply_trimmed_ylim(
        ax,
        _metric_values(rows, "route_logit_bias_scale"),
        outlier_proportion=outlier_proportion,
        lower_bound=0.0,
        upper_bound=1.0,
    )
    _apply_trimmed_ylim(
        ax2,
        _metric_values(rows, "ind_route_logit_bias", "ood_route_logit_bias"),
        outlier_proportion=outlier_proportion,
    )
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    ax = axes[2, 0]
    _plot_metric(ax, steps, rows, "main_grad_norm_post_clip", "main grad norm clipped", color="#2f7d57")
    _plot_metric(ax, steps, rows, "router_grad_norm_post_clip", "router grad norm clipped", color="#6d4c8d")
    max_grad_norm_values = _finite_values(_metric(rows, "max_grad_norm"))
    if max_grad_norm_values:
        ax.axhline(
            max_grad_norm_values[-1],
            color="#6f6f6f",
            linestyle=":",
            linewidth=1.0,
            label="clip max",
        )
    ax.set_title("Gradient Norms")
    ax.set_xlabel("step")
    ax.set_ylabel("grad norm")
    _apply_trimmed_ylim(
        ax,
        _metric_values(rows, "main_grad_norm_post_clip", "router_grad_norm_post_clip"),
        outlier_proportion=outlier_proportion,
        lower_bound=0.0,
    )
    lines, labels = ax.get_legend_handles_labels()
    if lines:
        ax.legend(lines, labels, fontsize=8)

    ax = axes[2, 1]
    _plot_metric(ax, steps, rows, "router_expert_weight_diff_mean_abs", "router diff mean abs", color="#6d4c8d")
    _plot_metric(ax, steps, rows, "router_expert_weight_diff_mean_squared", "router diff mean squared", color="#aa7cc2")
    ax2 = ax.twinx()
    _plot_metric(ax2, steps, rows, "expert_weight_diff_mean_abs", "expert diff mean abs", color="#2f7d57")
    _plot_metric(ax2, steps, rows, "expert_weight_diff_mean_squared", "expert diff mean squared", color="#7aa95c")
    ax.set_title("Router and FFN Expert Weight Separation")
    ax.set_xlabel("step")
    ax.set_ylabel("router weight diff")
    ax2.set_ylabel("FFN expert weight diff")
    _apply_trimmed_ylim(
        ax,
        _metric_values(
            rows,
            "router_expert_weight_diff_mean_abs",
            "router_expert_weight_diff_mean_squared",
        ),
        outlier_proportion=outlier_proportion,
        lower_bound=0.0,
    )
    _apply_trimmed_ylim(
        ax2,
        _metric_values(
            rows,
            "expert_weight_diff_mean_abs",
            "expert_weight_diff_mean_squared",
        ),
        outlier_proportion=outlier_proportion,
        lower_bound=0.0,
    )
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    for ax in axes.ravel():
        ax.grid(True, alpha=0.28)


def _plot_metrics(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    rolling_window: int,
    outlier_proportion: float,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    _render_metrics_figure(
        rows,
        fig,
        axes,
        title=title,
        rolling_window=rolling_window,
        outlier_proportion=outlier_proportion,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


_AxisLimits = tuple[tuple[float, float], tuple[float, float]]


def _axis_identity(ax: Any) -> tuple[str, str, str]:
    return (ax.get_title(), ax.get_xlabel(), ax.get_ylabel())


def _axis_limits(ax: Any) -> _AxisLimits:
    return (
        tuple(float(value) for value in ax.get_xlim()),
        tuple(float(value) for value in ax.get_ylim()),
    )


def _limits_close(left: _AxisLimits, right: _AxisLimits) -> bool:
    return all(
        math.isclose(left_value, right_value, rel_tol=1e-6, abs_tol=1e-9)
        for left_pair, right_pair in zip(left, right)
        for left_value, right_value in zip(left_pair, right_pair)
    )


def _capture_axis_limits(fig: Any) -> dict[tuple[str, str, str], _AxisLimits]:
    return {_axis_identity(ax): _axis_limits(ax) for ax in fig.axes}


class _LivePlotViewer:
    def __init__(self, *, title: str) -> None:
        plt.ion()
        plt.style.use("seaborn-v0_8-whitegrid")
        self._title = title
        self._figure = plt.figure(figsize=(16, 12), constrained_layout=True)
        self._auto_limits: dict[tuple[str, str, str], _AxisLimits] = {}
        manager = getattr(self._figure.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(title)

    def is_open(self) -> bool:
        return bool(plt.fignum_exists(self._figure.number))

    def refresh(
        self,
        rows: list[dict[str, Any]],
        output_path: Path,
        *,
        rolling_window: int,
        outlier_proportion: float,
    ) -> None:
        current_limits = _capture_axis_limits(self._figure)
        preserved_limits = {
            identity: limits
            for identity, limits in current_limits.items()
            if identity in self._auto_limits and not _limits_close(limits, self._auto_limits[identity])
        }
        self._figure.clf()
        axes = self._figure.subplots(3, 2)
        _render_metrics_figure(
            rows,
            self._figure,
            axes,
            title=self._title,
            rolling_window=rolling_window,
            outlier_proportion=outlier_proportion,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(output_path, dpi=160)
        new_auto_limits = _capture_axis_limits(self._figure)
        for ax in self._figure.axes:
            identity = _axis_identity(ax)
            limits = preserved_limits.get(identity)
            if limits is not None:
                ax.set_xlim(*limits[0])
                ax.set_ylim(*limits[1])
        self._auto_limits = new_auto_limits
        self._figure.canvas.draw_idle()
        plt.pause(0.001)

    def pause(self, seconds: float) -> None:
        plt.pause(seconds)

    def show_blocking(self) -> None:
        plt.ioff()
        plt.show()


def _watch_metrics(
    metrics_path: Path,
    output_path: Path,
    *,
    title: str,
    rolling_window: int,
    outlier_proportion: float,
    interval_s: float,
    show: bool,
) -> None:
    if interval_s <= 0:
        raise SystemExit("--watch-interval must be positive")
    last_signature: tuple[int, int | None] | None = None
    viewer = _LivePlotViewer(title=f"{title} Live") if show else None
    print(f"Watching {metrics_path.resolve()} -> {output_path.resolve()}", flush=True)
    try:
        while True:
            if viewer is not None and not viewer.is_open():
                break
            try:
                stat = metrics_path.stat()
                rows = _read_rows(metrics_path, ignore_partial_final_line=True)
            except FileNotFoundError:
                rows = []
                stat = None
            if rows:
                signature = (len(rows), stat.st_size if stat is not None else None)
                if signature != last_signature:
                    if viewer is None:
                        _plot_metrics(
                            rows,
                            output_path,
                            title=title,
                            rolling_window=rolling_window,
                            outlier_proportion=outlier_proportion,
                        )
                    else:
                        viewer.refresh(
                            rows,
                            output_path,
                            rolling_window=rolling_window,
                            outlier_proportion=outlier_proportion,
                        )
                    latest_step = rows[-1].get("step", "?")
                    print(
                        f"updated {output_path.resolve()} rows={len(rows)} latest_step={latest_step}",
                        flush=True,
                    )
                    last_signature = signature
            if viewer is not None:
                viewer.pause(interval_s)
            else:
                time.sleep(interval_s)
    except KeyboardInterrupt:
        print("Stopped watching.", flush=True)


def main() -> None:
    args = _parse_args()
    metrics_path = args.metrics_path
    output_path = args.output or metrics_path.with_name("training_metrics_plot.png")
    if args.watch:
        _watch_metrics(
            metrics_path,
            output_path,
            title=args.title,
            rolling_window=args.average_steps,
            outlier_proportion=args.outlier_proportion,
            interval_s=args.watch_interval,
            show=args.show,
        )
        return
    rows = _load_rows(metrics_path)
    _plot_metrics(
        rows,
        output_path,
        title=args.title,
        rolling_window=args.average_steps,
        outlier_proportion=args.outlier_proportion,
    )
    print(output_path.resolve())
    if args.show:
        viewer = _LivePlotViewer(title=args.title)
        viewer.refresh(
            rows,
            output_path,
            rolling_window=args.average_steps,
            outlier_proportion=args.outlier_proportion,
        )
        viewer.show_blocking()


if __name__ == "__main__":
    main()
