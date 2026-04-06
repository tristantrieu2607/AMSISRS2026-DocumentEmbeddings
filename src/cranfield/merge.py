"""Generate merged document embeddings for Cranfield."""

import json
import ollama
from preprocess import load_documents

doc_dict = load_documents()
doc_ids = sorted(doc_dict.keys(), key=int)

merge_dict = {}
i = 0
while i < len(doc_ids) - 1:
    merge_dict[doc_ids[i]] = doc_dict[doc_ids[i]] + "\n" + doc_dict[doc_ids[i + 1]]
    i += 2
if len(doc_ids) % 2 == 1:
    merge_dict[doc_ids[-1]] = doc_dict[doc_ids[-1]]

merge_embeddings = {}
for idx, (doc_id, text) in enumerate(merge_dict.items(), start=1):
    print(f"\r{idx}/{len(merge_dict)}", end="")
    response = ollama.embed(model="nomic-embed-text:latest", input=text)
    merge_embeddings[doc_id] = response["embeddings"][0]
print()
with open("../../cranfield/data/merge_embeddings.json", "w") as f:
    json.dump(merge_embeddings, f)
print(f"Saved {len(merge_embeddings)} merged embeddings.")
