"""Fix import paths in new src/retrieval/ and src/generation/ files."""

import os
import re

ROOT = r"C:\WorkSpace\PythonSpace\Medical-RAG-System\src"

# retrival inner imports: src.query.retrieval -> src.retrieval
# retrival inner imports: src.query.reranker  -> src.retrieval.reranker
FIXES = [
    # === retrival ===
    ("from src.query.retrieval", "from src.retrieval"),
    ("from src.query.reranker", "from src.retrieval.reranker"),
    ("from src.query.citation", "from src.generation.citation"),
]


def fix_file(path: str) -> bool:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    original = content
    for old, new in FIXES:
        content = re.sub(old, new, content)
    if content != original:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


count = 0
for root, _dirs, files in os.walk(os.path.join(ROOT, "retrieval")):
    for f in files:
        if f.endswith(".py"):
            if fix_file(os.path.join(root, f)):
                count += 1
                print(f"  Fixed: retrival/{f}")

print(f"\n✅ {count} files fixed in retrival/")
