"""Fix test imports for the new architecture."""

import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "tests")

REPLACE = [
    (r"from src\.query\.confidence import", "from src.agent.confidence import"),
    (r"from src\.query\.generation import", "from src.generation import"),
    (r"from src\.query\.retrieval import", "from src.retrieval import"),
    (r"from src\.query\.reranker import", "from src.retrieval.reranker import"),
    (r"from src\.query\.citation import", "from src.generation.citation import"),
    (r"from src\.query\.engine import RAGEngine", "from src.agent.rag_agent import RAGAgent"),
    (r"from src\.query\.engine import", "from src.agent.rag_agent import"),
    (r"from src\.query import (?!api|router)", "from src.retrieval import "),
]

count = 0
for root, _dirs, files in os.walk(ROOT):
    for f in files:
        if not f.endswith(".py"):
            continue
        path = os.path.join(root, f)
        with open(path, encoding="utf-8") as fh:
            content = fh.read()
        if "src.query" not in content:
            continue
        original = content
        for pattern, replacement in REPLACE:
            content = re.sub(pattern, replacement, content)
        if content != original:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            count += 1
            print(f"  Fixed: {os.path.relpath(path, ROOT)}")

print(f"\n✅ {count} test files fixed")
