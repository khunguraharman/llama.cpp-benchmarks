import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/nvidia-rtx-3080/throughput-batch-bench-results/ubatch"
OUTPUT = ROOT / "benchmark_plots/rtx3080_generation_batch4096_ubatch.png"

series = {}
for path in sorted(SOURCE.glob("*_p512_b4096_ub*_n*_r10.json")):
    for row in json.loads(path.read_text()):
        if (
            row["n_batch"] == 4096
            and row["n_gen"] > 0
            and row["gpu_info"] == "NVIDIA GeForce RTX 3080"
        ):
            model = Path(row["model_filename"]).name.removesuffix(".gguf")
            series.setdefault(model, {}).setdefault(row["n_ubatch"], []).append(
                (row["n_gen"], row["avg_ts"])
            )

assert len(series) == 3, f"Expected three models, got {list(series)}"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"]
markers = ["o", "s", "^", "D", "x"]
styles = ["-", "--", "-.", ":", "--"]
for ax, (model, batches) in zip(axes, sorted(series.items())):
    for i, (ubatch, points) in enumerate(sorted(batches.items())):
        points.sort()
        assert len(points) == 15 and len(set(x for x, _ in points)) == 15
        x, y = zip(*points)
        ax.plot(x, y, label=str(ubatch), color=colors[i], marker=markers[i],
                linestyle=styles[i], linewidth=1.4, markersize=5,
                markerfacecolor="none", markeredgewidth=1, alpha=0.85)
    ax.set_title(model, loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tokens/s")
    ax.grid(True, color="#e0e0e0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.margins(x=0.02, y=0.12)

axes[-1].set_xlabel("Generated tokens")
axes[-1].set_xticks([512, 1024, 1536, 2048, 2560, 3072, 3584, 4096])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="ubatch_size", loc="upper center",
           bbox_to_anchor=(0.5, 0.952), ncol=5, frameon=False)
fig.suptitle("RTX 3080 generation throughput | batch_size = 4096", fontsize=15, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=2)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=160, facecolor="white")
plt.close(fig)
print(OUTPUT)
