"""LISA dataset preprocessing from Part1-Part14 files."""


def load_documents(data_path="lisa_dataset"):
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


def load_queries(data_path="lisa_dataset"):
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
            buffer.append(line[:-1].strip())
            query_dict[qid] = " ".join(buffer)
            buffer = []
        else:
            buffer.append(line)
    return query_dict


def get_missed_ids(doc_dict):
    """Return list of integer IDs missing from the document collection."""
    return [i for i in range(1, 6004) if str(i) not in doc_dict]
