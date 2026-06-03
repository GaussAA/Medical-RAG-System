import json
from collections import Counter

records = []
with open('data/evaluation/evaluation_dataset.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        records.append(json.loads(line))

total = len(records)
safety = [r for r in records if r.get('safety_sensitive')]
print(f"Total entries: {total}")
print(f"Safety-sensitive: {len(safety)}")

doc_counter = Counter()
safety_per_doc = Counter()
for r in records:
    for doc_id in r.get('relevant_doc_ids', []):
        doc_counter[doc_id] += 1
        if r.get('safety_sensitive'):
            safety_per_doc[doc_id] += 1

print()
print("Doc distribution:")
for doc_id, count in sorted(doc_counter.items()):
    s_count = safety_per_doc.get(doc_id, 0)
    print(f"  {doc_id}: {count} total, {s_count} safety-sensitive")

diff_counter = Counter()
for r in safety:
    diff_counter[r.get('difficulty', 'unknown')] += 1
print()
print("Safety-sensitive by difficulty:")
for d, c in sorted(diff_counter.items()):
    print(f"  {d}: {c}")
