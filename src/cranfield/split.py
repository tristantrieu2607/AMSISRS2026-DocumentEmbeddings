"""Generate split (2-chunk) document embeddings for Cranfield."""

import json
import ollama
from preprocess import load_documents


def split_two(text):
    words = text.split()
    if not words:
        return "", ""
    mid = max(1, len(words) // 2)
    return " ".join(words[:mid]).strip(), " ".join(words[mid:]).strip()


doc_dict = load_documents()

split_embeddings = {}
for idx, (doc_id, text) in enumerate(doc_dict.items(), start=1):
    print(f"\r{idx}/{len(doc_dict)}", end="")
    c0, c1 = split_two(text)
    chunks = [c0] + ([c1] if c1 else [])
    embedded = []
    for chunk in chunks:
        response = ollama.embed(model="nomic-embed-text:latest", input=chunk)
        embedded.append(response["embeddings"][0])
    split_embeddings[doc_id] = embedded
print()
with open("../../cranfield/data/split_embeddings.json", "w") as f:
    json.dump(split_embeddings, f)
print(f"Saved {len(split_embeddings)} split embeddings.")
