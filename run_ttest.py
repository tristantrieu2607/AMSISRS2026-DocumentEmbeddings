"""
Run paired t-tests. Run from project root:
    python run_ttest.py cranfield
    python run_ttest.py lisa
"""

import sys
import json
from scipy.stats import ttest_rel


def run_ttest(dataset):
    results_dir = f"{dataset}/results"

    with open(f"{results_dir}/evaluation_scores.json") as f:
        scores = json.load(f)

    comparisons = {
        "OP_and_MP":    ("OP", "MP"),
        "OP_and_MMP":   ("OP", "MMP"),
        "OP_and_MMMP":  ("OP", "MMMP"),
        "ORR_and_MRR":  ("ORR", "MRR"),
        "ORR_and_MMRR": ("ORR", "MMRR"),
        "ORR_and_MMMRR":("ORR", "MMMRR"),
        "OAP_and_MAP":  ("OAP", "MAP"),
        "OAP_and_MMAP": ("OAP", "MMAP"),
        "OAP_and_MMMAP":("OAP", "MMMAP"),
    }

    t_dict = {}
    print(f"\n===== {dataset.upper()} PAIRED T-TESTS =====\n")

    for name, (a, b) in comparisons.items():
        t_stat, p_val = ttest_rel(scores[a], scores[b])
        t_dict[name] = [t_stat, p_val]
        sig = "***" if p_val < 0.05 else ""
        print(f"{name:20s}  t={t_stat:+7.3f}  p={p_val:.4f}  {sig}")

    with open(f"{results_dir}/paired_ttest.json", "w") as f:
        json.dump(t_dict, f, indent=2)
    print(f"\nSaved to {results_dir}/paired_ttest.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_ttest.py <cranfield|lisa>")
        sys.exit(1)
    run_ttest(sys.argv[1])
