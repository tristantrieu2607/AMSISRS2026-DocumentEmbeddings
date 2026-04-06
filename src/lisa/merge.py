"""Generate merged document embeddings for LISA."""

import json
import ollama
from preprocess import load_documents, get_missed_ids

doc_dict = load_documents()
missed_ids = get_missed_ids(doc_dict)

merge_dict = {}
i = 1
while i < 6004:
    a, b = str(i), str(i + 1)
    if (i not in missed_ids) and ((i + 1) not in missed_ids) and (a in doc_dict) and (b in doc_dict):
        merge_dict[a] = (doc_dict[a] + " " + doc_dict[b]).strip()
    i += 2

merge_embeddings = {}
for idx, (doc_id, text) in enumerate(merge_dict.items(), start=1):
    print(f"\r{idx}/{len(merge_dict)}", end="")
    response = ollama.embed(model="nomic-embed-text:latest", input=text)
    merge_embeddings[doc_id] = response["embeddings"][0]
print()
with open("../../lisa/data/merge_embeddings.json", "w") as f:
    json.dump(merge_embeddings, f)
print(f"Saved {len(merge_embeddings)} merged embeddings.")
