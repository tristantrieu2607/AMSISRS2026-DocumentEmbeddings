"""Unified embedding loader. Auto-unwraps nested [[768]] -> [768]."""

import json


def _unwrap(vec):
    if isinstance(vec, list) and len(vec) == 1 and isinstance(vec[0], list):
        return vec[0]
    return vec


def load_embeddings(path):
    """Load embeddings from a JSON file. Returns {id: vector}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: _unwrap(v) for k, v in data.items()}
