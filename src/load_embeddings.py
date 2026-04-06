"""
Unified embedding loader. Loads .json files and auto-unwraps
nested embeddings [[768]] -> [768] if present.
"""

import json


def _unwrap(vec):
    """Unwrap [[768 floats]] -> [768 floats] if nested."""
    if isinstance(vec, list) and len(vec) == 1 and isinstance(vec[0], list):
        return vec[0]
    return vec


def load_embeddings(path: str) -> dict:
    """Load embeddings from a JSON file.

    Returns dict of {id: vector} with all vectors unwrapped to flat lists.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: _unwrap(v) for k, v in data.items()}
