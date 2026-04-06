"""Evaluation metrics for document embedding retrieval experiments."""

import numpy as np


def cosine_sim(x, y):
    """Cosine similarity between two vectors."""
    v1 = np.array(x).reshape(-1)
    v2 = np.array(y).reshape(-1)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def _is_relevant(doc_id, rel_set):
    """Check if a (possibly merged) document ID is relevant."""
    parts = doc_id.split(".")
    return any(p in rel_set for p in parts)


def _rank_docs(sim_dict):
    """Return list of (doc_id, score) sorted by descending similarity."""
    return sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)


def precision_at_k(sim_dict, rel, k=10):
    """Proportion of relevant documents in the top-k results."""
    rel_set = set(map(str, rel))
    ranked = _rank_docs(sim_dict)
    topk = [doc_id for doc_id, _ in ranked[:k]]
    return sum(1 for d in topk if _is_relevant(d, rel_set)) / k


def reciprocal_rank(sim_dict, rel):
    """Inverse of the rank of the first relevant document."""
    rel_set = set(map(str, rel))
    for rank, (doc_id, _) in enumerate(_rank_docs(sim_dict), start=1):
        if _is_relevant(doc_id, rel_set):
            return 1.0 / rank
    return 0.0


def average_precision(sim_dict, rel):
    """Average of precision values at each relevant document's rank."""
    rel_set = set(map(str, rel))
    ranked = _rank_docs(sim_dict)
    total_rel = sum(1 for d in sim_dict if _is_relevant(d, rel_set))
    if total_rel == 0:
        return 0.0
    num_rel = 0
    ap_sum = 0.0
    for rank, (doc_id, _) in enumerate(ranked, start=1):
        if _is_relevant(doc_id, rel_set):
            num_rel += 1
            ap_sum += num_rel / rank
    return ap_sum / total_rel


def build_sim_dict(query_vec, doc_embeddings):
    """Compute cosine similarity between a query and all documents."""
    return {did: cosine_sim(query_vec, dvec) for did, dvec in doc_embeddings.items()}


def build_split_sim_dict(query_vec, split_embeddings):
    """Compute similarity for split documents, pooling by max."""
    return {did: max(cosine_sim(query_vec, c) for c in chunks)
            for did, chunks in split_embeddings.items()}
