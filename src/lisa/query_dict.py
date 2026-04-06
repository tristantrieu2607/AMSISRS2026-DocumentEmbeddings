"""Parse LISA dataset queries and generate query embeddings."""

import ollama
import json

query_dict = {}

with open("lisa_dataset/QUERY", "r", encoding="utf-8") as f:
    lines = f.readlines()

qid = None
buffer = []

for line in lines:
    line = line.strip()

    if not line:
        continue

    if line.isdigit():
        qid = line
        buffer = []

    elif line.endswith("#"):
        line = line[:-1].strip()
        buffer.append(line)
        query_dict[qid] = " ".join(buffer)
        buffer = []

    else:
        buffer.append(line)

# Generate embeddings
query_embeddings = {}

for q in query_dict:
    response = ollama.embed(
        model="nomic-embed-text:latest",
        input=query_dict[q]
    )
    query_embeddings[q] = response["embeddings"][0]

with open("query_embeddings.json", "w") as f:
    json.dump(query_embeddings, f)

print(f"Saved {len(query_embeddings)} query embeddings.")
