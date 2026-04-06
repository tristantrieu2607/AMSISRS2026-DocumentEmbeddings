"""
Main evaluation pipeline. Run from project root:
    python main.py cranfield
    python main.py lisa
"""

import sys
import json
from src.load_embeddings import load_embeddings
from src.eval import precision_at_k, reciprocal_rank, average_precision, build_sim_dict
from src.relevance import load_relevance


def evaluate(dataset, k=10):
    data_dir = f"{dataset}/data"
    results_dir = f"{dataset}/results"

    print(f"Loading embeddings for {dataset}...")
    query_embeds = load_embeddings(f"{data_dir}/query_embeddings.json")
    original = load_embeddings(f"{data_dir}/original_embeddings.json")
    merge = load_embeddings(f"{data_dir}/merge_embeddings.json")
    merge2 = load_embeddings(f"{data_dir}/merge2_embeddings.json")
    merge3 = load_embeddings(f"{data_dir}/merge3_embeddings.json")

    relevant = load_relevance(dataset)

    conditions = {
        "original": original,
        "merge": merge,
        "merge2": merge2,
        "merge3": merge3,
    }

    key_map = {
        "original": ("OP", "ORR", "OAP"),
        "merge": ("MP", "MRR", "MAP"),
        "merge2": ("MMP", "MMRR", "MMAP"),
        "merge3": ("MMMP", "MMMRR", "MMMAP"),
    }

    scores = {v: [] for vals in key_map.values() for v in vals}
    query_ids = sorted(relevant.keys(), key=lambda x: int(x))

    print(f"Evaluating {len(query_ids)} queries across 4 conditions...")
    for qid in query_ids:
        if qid not in query_embeds:
            continue
        q_vec = query_embeds[qid]
        rel = relevant[qid]

        for cond_name, cond_embeds in conditions.items():
            sim_dict = build_sim_dict(q_vec, cond_embeds)
            p_key, rr_key, ap_key = key_map[cond_name]
            scores[p_key].append(precision_at_k(sim_dict, rel, k))
            scores[rr_key].append(reciprocal_rank(sim_dict, rel))
            scores[ap_key].append(average_precision(sim_dict, rel))

    # Means
    mean_scores = {
        "PAK": [sum(scores[k]) / len(scores[k]) for k in ["OP", "MP", "MMP", "MMMP"]],
        "RR":  [sum(scores[k]) / len(scores[k]) for k in ["ORR", "MRR", "MMRR", "MMMRR"]],
        "AP":  [sum(scores[k]) / len(scores[k]) for k in ["OAP", "MAP", "MMAP", "MMMAP"]],
    }

    conds = ["Original", "Merge", "Merge×2", "Merge×3"]
    print(f"\n===== {dataset.upper()} RESULTS (K={k}) =====")
    for i, cond in enumerate(conds):
        print(f"{cond:12s} -> P@{k}: {mean_scores['PAK'][i]:.4f}, "
              f"MRR: {mean_scores['RR'][i]:.4f}, MAP: {mean_scores['AP'][i]:.4f}")

    with open(f"{results_dir}/evaluation_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    with open(f"{results_dir}/mean_evaluation_scores.json", "w") as f:
        json.dump(mean_scores, f, indent=2)
    print(f"Saved to {results_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <cranfield|lisa>")
        sys.exit(1)
    evaluate(sys.argv[1])
