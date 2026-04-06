"""
LISA dataset preprocessing.

Loads documents from Part1-Part14 files and queries from QUERY file.
Provides them as dictionaries for embedding generation.
"""


def load_documents(data_path: str = "lisa_dataset") -> dict:
    """Load LISA documents as {doc_id: text}."""
    doc_dict = {}

    for part_no in range(1, 15):
        with open(f"{data_path}/Part{part_no}", "r", encoding="utf-8") as f:
            lines = f.readlines()

        did = None
        buffer = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Document"):
                did = line.split()[1]
                buffer = []
            elif line == "********************************************":
                doc_dict[did] = " ".join(buffer)
                buffer = []
            else:
                buffer.append(line)

    return doc_dict


def load_queries(data_path: str = "lisa_dataset") -> dict:
    """Load LISA queries as {query_id: text}."""
    query_dict = {}

    with open(f"{data_path}/QUERY", "r", encoding="utf-8") as f:
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

    return query_dict


def get_missed_ids(doc_dict: dict) -> list:
    """Return list of integer IDs missing from the document collection."""
    return [i for i in range(1, 6004) if str(i) not in doc_dict]
