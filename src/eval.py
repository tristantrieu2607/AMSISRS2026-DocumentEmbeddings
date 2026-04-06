"""
Evaluation metrics for document embedding retrieval experiments.

All functions use a unified interface:
    sim_dict : dict[str, float]  — {doc_id: similarity_score}
    rel      : list[str] | set[str] — relevant document IDs
    k        : int — cutoff for top-k metrics

For merged documents (keys like "1.3" or "1.2.3.4"), a merged doc
is considered relevant if ANY of its component IDs is in the
relevant set.

For split documents, the caller should pre-pool chunk scores into
one score per parent doc before calling these functions.
"""

import numpy as np


# --------------- Similarity ---------------

def cosine_sim(x, y):
    """Cosine similarity between two vectors."""
    v1 = np.array(x).reshape(-1)
    v2 = np.array(y).reshape(-1)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# --------------- Relevance helpers ---------------

def _is_relevant(doc_id: str, rel_set: set) -> bool:
    """Check if a (possibly merged) document ID is relevant."""
    parts = doc_id.split(".")
    return any(p in rel_set for p in parts)


def _rank_docs(sim_dict: dict) -> list:
    """Return list of (doc_id, score) sorted by descending similarity."""
    return sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)


# --------------- Precision@K ---------------

def precision_at_k(sim_dict: dict, rel, k=10):
    """Proportion of relevant documents in the top-k results."""
    rel_set = set(map(str, rel))
    ranked = _rank_docs(sim_dict)
    topk = [doc_id for doc_id, _ in ranked[:k]]
    rel_count = sum(1 for d in topk if _is_relevant(d, rel_set))
    return rel_count / k


# --------------- Reciprocal Rank ---------------

def reciprocal_rank(sim_dict: dict, rel):
    """Inverse of the rank of the first relevant document."""
    rel_set = set(map(str, rel))
    ranked = _rank_docs(sim_dict)

    for rank, (doc_id, _) in enumerate(ranked, start=1):
        if _is_relevant(doc_id, rel_set):
            return 1.0 / rank
    return 0.0


# --------------- Average Precision ---------------

def average_precision(sim_dict: dict, rel):
    """Average of precision values at each relevant document's rank."""
    rel_set = set(map(str, rel))
    ranked = _rank_docs(sim_dict)

    total_rel = sum(1 for doc_id in sim_dict if _is_relevant(doc_id, rel_set))
    if total_rel == 0:
        return 0.0

    num_rel = 0
    ap_sum = 0.0

    for rank, (doc_id, _) in enumerate(ranked, start=1):
        if _is_relevant(doc_id, rel_set):
            num_rel += 1
            ap_sum += num_rel / rank

    return ap_sum / total_rel


# --------------- Helper: build sim_dict ---------------

def build_sim_dict(query_vec, doc_embeddings: dict) -> dict:
    """Compute cosine similarity between a query and all documents."""
    return {
        doc_id: cosine_sim(query_vec, doc_vec)
        for doc_id, doc_vec in doc_embeddings.items()
    }


def build_split_sim_dict(query_vec, split_embeddings: dict) -> dict:
    """Compute similarity for split documents, pooling by max."""
    sim_dict = {}
    for doc_id, chunks in split_embeddings.items():
        sims = [cosine_sim(query_vec, chunk) for chunk in chunks]
        sim_dict[doc_id] = max(sims)
    return sim_dict
