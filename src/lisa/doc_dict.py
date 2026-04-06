"""Parse LISA dataset documents from Part1-Part14 files."""

doc_dict = {}


def retrieve_doc(no: int):
    with open(f"lisa_dataset/Part{no}", "r", encoding="utf-8") as f:
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


for parts in range(1, 15):
    retrieve_doc(parts)

missed_ids = []
for i in range(1, 6004):
    if str(i) not in doc_dict:
        missed_ids.append(i)
