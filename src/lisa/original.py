"""Generate original document + query embeddings for LISA."""

import json
import ollama
from preprocess import load_documents, load_queries

doc_dict = load_documents()
query_dict = load_queries()

# Embed documents
doc_embeddings = {}
for idx, (doc_id, text) in enumerate(doc_dict.items(), start=1):
    print(f"\rDocuments: {idx}/{len(doc_dict)}", end="")
    response = ollama.embed(model="nomic-embed-text:latest", input=text)
    doc_embeddings[doc_id] = response["embeddings"][0]
print()

with open("../../lisa/data/original_embeddings.json", "w") as f:
    json.dump(doc_embeddings, f)
print(f"Saved {len(doc_embeddings)} document embeddings.")

# Embed queries
query_embeddings = {}
for qid, text in query_dict.items():
    response = ollama.embed(model="nomic-embed-text:latest", input=text)
    query_embeddings[qid] = response["embeddings"][0]

with open("../../lisa/data/query_embeddings.json", "w") as f:
    json.dump(query_embeddings, f)
print(f"Saved {len(query_embeddings)} query embeddings.")
