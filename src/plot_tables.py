"""
Generate evaluation result tables as images for a dataset.

Produces two table images:
  1. Evaluation results summary table
  2. Paired t-test results table

Usage:
    python src/plot_tables.py cranfield
    python src/plot_tables.py lisa
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt


def mean_sem(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def plot_tables(dataset: str):
    results_dir = f"{dataset}/results"

    # --- Table 1: Evaluation results ---
    with open(f"{results_dir}/evaluation_scores.json") as f:
        evals = json.load(f)

    conditions = ["Original", "Merge", "Merge×2", "Merge×3"]
    cond_keys = {
        "Original": ("OP", "ORR", "OAP"),
        "Merge": ("MP", "MRR", "MAP"),
        "Merge×2": ("MMP", "MMRR", "MMAP"),
        "Merge×3": ("MMMP", "MMMRR", "MMMAP"),
    }

    table_data = []
    for cond in conditions:
        p, rr, ap = cond_keys[cond]
        p_mean, _ = mean_sem(evals[p])
        rr_mean, _ = mean_sem(evals[rr])
        ap_mean, sem = mean_sem(evals[ap])
        table_data.append([cond, f"{p_mean:.3f}", f"{rr_mean:.3f}", f"{ap_mean:.3f} ± {sem:.3f}"])

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axis("off")
    tbl = ax.table(cellText=table_data,
                   colLabels=["Condition", "Mean P@10", "Mean RR", "Mean AP ± SEM"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    plt.title(f"Evaluation Results ({dataset.capitalize()})", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(f"figures/{dataset}_evaluation_table.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figures/{dataset}_evaluation_table.png")

    # --- Table 2: T-test results ---
    with open(f"{results_dir}/paired_ttest.json") as f:
        ttests = json.load(f)

    def sig(p):
        return "✓" if p < 0.05 else "–"

    rows = []
    for name, (t, p) in ttests.items():
        left, _, right = name.partition("_and_")
        if left.startswith("OP"):
            metric = "P@10"
        elif left.startswith("ORR"):
            metric = "RR"
        else:
            metric = "AP"

        if right in ["MP", "MRR", "MAP"]:
            comp = "Merge"
        elif right in ["MMP", "MMRR", "MMAP"]:
            comp = "Merge×2"
        else:
            comp = "Merge×3"

        rows.append([f"Original vs {comp}", metric, f"{t:.3f}", f"{p:.4f}", sig(p)])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    tbl = ax.table(cellText=rows,
                   colLabels=["Comparison", "Metric", "t-stat", "p-value", "Sig. (p<0.05)"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.5)
    plt.title(f"Paired t-test Results ({dataset.capitalize()})", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(f"figures/{dataset}_ttest_table.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figures/{dataset}_ttest_table.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/plot_tables.py <cranfield|lisa>")
        sys.exit(1)
    plot_tables(sys.argv[1])
