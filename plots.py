import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if TkAgg isn't available
import numpy as np

with open("results/evals.json") as f:
    data = json.load(f)

data = [e for e in data if e["dataset"] != "mmlu"]

PRIMARY_METRIC = {
    "dala": "macro_f1",
    "danish-citizen-tests": "accuracy",
    "gec": ("mean", "accuracy"),  # fallback: some entries use 'accuracy'
    "ifeval-da": "final_acc",
    "multi": "mean",
    "piqa": "accuracy",
}
STDERR_METRIC = {
    "gec": "stderr",
    "ifeval-da": "final_stderr",
    "multi": "stderr",
}

def get_value(ds, metrics):
    key = PRIMARY_METRIC[ds]
    if isinstance(key, tuple):
        for k in key:
            if k in metrics:
                return metrics[k]
        return None
    return metrics.get(key)

def get_stderr(ds, metrics):
    key = STDERR_METRIC.get(ds)
    return metrics.get(key) if key else None

# Infer model order and datasets from the data itself
MODEL_ORDER = list(dict.fromkeys(e["model"] for e in data))
DATASETS = list(dict.fromkeys(e["dataset"] for e in data))

lookup = {}
for e in data:
    lookup[(e["model"], e["dataset"])] = (
        get_value(e["dataset"], e["metrics"]),
        get_stderr(e["dataset"], e["metrics"]),
    )

DATASET_LABELS = {
    "dala": "DALA (macro F1)",
    "danish-citizen-tests": "Danish Citizen Tests (accuracy)",
    "gec": "GEC (mean/accuracy)",
    "ifeval-da": "IFEval-DA (final acc)",
    "multi": "Multi (mean)",
    "piqa": "PIQA (accuracy)",
}

COLORS = ["#e05c5c", "#f0a500", "#4caf7d", "#4a90d9", "#9b59b6", "#e8785a"]

x_labels = [m.replace("minimoe", "moe") for m in MODEL_ORDER]
xs = np.arange(len(MODEL_ORDER))

fig, ax = plt.subplots(figsize=(10, 6))

for i, ds in enumerate(DATASETS):
    values, errs = [], []
    for model in MODEL_ORDER:
        v, e = lookup.get((model, ds), (None, None))
        values.append(v)
        errs.append(e)

    color = COLORS[i % len(COLORS)]
    label = DATASET_LABELS.get(ds, ds)
    ax.plot(xs, values, color=color, linewidth=2, zorder=3)
    ax.plot(xs, values, "o", color=color, markersize=7, zorder=4,
            markeredgecolor="white", markeredgewidth=1.5, label=label)

    for j, (v, e) in enumerate(zip(values, errs)):
        if v is not None and e is not None:
            ax.errorbar(j, v, yerr=e, fmt="none", color=color,
                        capsize=4, linewidth=1.2, alpha=0.6, zorder=2)

ax.set_xticks(xs)
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Performance across model versions", fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0.15, 0.85)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
ax.grid(axis="y", alpha=0.25, zorder=0)
ax.grid(axis="x", alpha=0.1, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9, edgecolor="#ddd")

ax.axvspan(-0.4, 0.4, color="#aaa", alpha=0.08, zorder=0)
ax.text(0, 0.16, "baseline", ha="center", fontsize=8, color="#999")

plt.tight_layout()
plt.show()
#plt.savefig("evals_plot.png", dpi=150, bbox_inches="tight")
#print("Saved evals_plot.png")
