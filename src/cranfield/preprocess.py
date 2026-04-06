"""
Cranfield dataset preprocessing.

Loads documents and queries via ir_datasets and provides
them as dictionaries for embedding generation.
"""

import ir_datasets


def load_documents() -> dict:
    """Load Cranfield documents as {doc_id: text}."""
    dataset = ir_datasets.load("cranfield")
    doc_dict = {}
    for d in dataset.docs_iter():
        title = d.title or ""
        text = d.text or ""
        doc_dict[str(d.doc_id)] = f"Title: {title}\nText: {text}".strip()
    return doc_dict


def load_queries() -> dict:
    """Load Cranfield queries as {query_id: text}."""
    dataset = ir_datasets.load("cranfield")
    query_dict = {}
    for q in dataset.queries_iter():
        text = (q.text or "").strip()
        if text:
            query_dict[str(q.query_id)] = text
    return query_dict


def load_relevance() -> dict:
    """Load relevance judgements as {query_id: [doc_ids]}."""
    dataset = ir_datasets.load("cranfield")
    relevant = {}
    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        did = str(qrel.doc_id)
        if qrel.relevance > 0:
            if qid not in relevant:
                relevant[qid] = []
            relevant[qid].append(did)
    return relevant
