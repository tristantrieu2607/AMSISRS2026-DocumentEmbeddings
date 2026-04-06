"""
Generate evaluation plots for a dataset.

Produces two figures:
  1. Performance decay curve (P@K, RR, AP across conditions)
  2. Mean AP with SEM error bars

Usage:
    python src/plot_results.py cranfield
    python src/plot_results.py lisa
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_results(dataset: str):
    results_dir = f"{dataset}/results"

    with open(f"{results_dir}/mean_evaluation_scores.json") as f:
        mean_evals = json.load(f)
    with open(f"{results_dir}/evaluation_scores.json") as f:
        evals = json.load(f)

    conds = ["Original", "Merge", "Merge×2", "Merge×3"]
    X = np.arange(len(conds))
    n_queries = len(evals["OP"])

    # --- Figure 1: Performance decay curve ---
    PAK = np.array(mean_evals["PAK"])
    RR = np.array(mean_evals["RR"])
    AP = np.array(mean_evals["AP"])

    plt.figure(figsize=(9, 5))
    plt.plot(X, PAK, marker="o", label="Precision@K")
    plt.plot(X, RR, marker="o", label="Reciprocal Rank")
    plt.plot(X, AP, marker="o", label="Average Precision")
    plt.xticks(X, conds)
    plt.ylabel(f"Mean metric over {n_queries} queries")
    plt.title(f"Performance decay vs embedding condition ({dataset.capitalize()})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{dataset}_performance_decay.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figures/{dataset}_performance_decay.png")

    # --- Figure 2: Mean AP with error bars ---
    AP_mat = np.vstack([evals["OAP"], evals["MAP"], evals["MMAP"], evals["MMMAP"]]).T
    mean_ap = AP_mat.mean(axis=0)
    sem_ap = AP_mat.std(axis=0, ddof=1) / np.sqrt(AP_mat.shape[0])

    plt.figure(figsize=(9, 5))
    plt.errorbar(X, mean_ap, yerr=sem_ap, fmt="o-", linewidth=3,
                 capsize=6, label="Mean AP ± SEM", zorder=5)
    plt.xticks(X, conds)
    plt.ylabel("Average Precision (AP)")
    plt.title(f"Mean AP across embedding conditions ({dataset.capitalize()})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/{dataset}_mean_ap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figures/{dataset}_mean_ap.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/plot_results.py <cranfield|lisa>")
        sys.exit(1)
    plot_results(sys.argv[1])
