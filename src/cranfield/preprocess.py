"""Cranfield dataset preprocessing via ir_datasets."""

import ir_datasets


def load_documents():
    """Load Cranfield documents as {doc_id: text}."""
    dataset = ir_datasets.load("cranfield")
    return {str(d.doc_id): f"Title: {d.title or ''}\nText: {d.text or ''}".strip()
            for d in dataset.docs_iter()}


def load_queries():
    """Load Cranfield queries as {query_id: text}."""
    dataset = ir_datasets.load("cranfield")
    return {str(q.query_id): (q.text or "").strip()
            for q in dataset.queries_iter() if (q.text or "").strip()}
