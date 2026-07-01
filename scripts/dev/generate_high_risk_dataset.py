# -*- coding: utf-8 -*-
\"\"\"Generate high-risk eval dataset (~60 entries), new scenarios.\"\"\"
import json, os

# Read existing to extract exact doc ID strings
entries_path = "data/evaluation/evaluation_dataset.jsonl"
with open(entries_path, encoding="utf-8") as f:
    existing = [json.loads(line) for line in f if line.strip()]

doc_ids_set = set()
for e in existing:
    for d in e["relevant_doc_ids"]:
        doc_ids_set.add(d)

# Convert to list for indexing
doc_ids = sorted(doc_ids_set)
print(f"Found {len(doc_ids)} unique doc IDs:")
for i, d in enumerate(doc_ids):
    print(f"  [{i}] {repr(d)}")
