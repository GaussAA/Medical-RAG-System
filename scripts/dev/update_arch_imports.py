"""Batch-update all cross-module import paths for the new architecture."""

import os
import re

ROOT = r"C:\WorkSpace\PythonSpace\Medical-RAG-System\src"

# Order matters: more specific patterns first
REPLACE = [
    # === src.query → new paths ===
    # agent slice
    (r"from src\.query\.engine import RAGEngine", "from src.agent.rag_agent import RAGAgent"),
    (r"from src\.query\.engine import", "from src.agent.rag_agent import"),
    (r"from src\.query\.confidence import", "from src.agent.confidence import"),
    (r"from src\.query\.api import", "from src.agent.api import"),
    (r"from src\.query import (RAGEnginePort|LLMGeneratorPort|HybridRetrieverPort)", "from src.retrieval import \\1"),
    # retrieval slice
    (r"from src\.query\.retrieval\.hybrid import", "from src.retrieval.hybrid import"),
    (r"from src\.query\.retrieval\.vector import", "from src.retrieval.vector import"),
    (r"from src\.query\.retrieval\.bm25 import", "from src.retrieval.bm25 import"),
    (r"from src\.query\.retrieval\.base import", "from src.retrieval.base import"),
    (r"from src\.query\.retrieval\.boosting import", "from src.retrieval.boosting import"),
    (r"from src\.query\.retrieval import", "from src.retrieval import"),
    # reranker
    (r"from src\.query\.reranker\.cross_encoder import", "from src.retrieval.reranker.cross_encoder import"),
    (r"from src\.query\.reranker import", "from src.retrieval.reranker import"),
    # generation slice
    (r"from src\.query\.generation\.generator import", "from src.generation.generator import"),
    (r"from src\.query\.generation\.prompt import", "from src.generation.prompt import"),
    (r"from src\.query\.generation\.prompt_builder import", "from src.generation.prompt_builder import"),
    (r"from src\.query\.generation\.warnings import", "from src.generation.warnings import"),
    (r"from src\.query\.generation import", "from src.generation import"),
    # citation
    (r"from src\.query\.citation\.verifier import", "from src.generation.citation.verifier import"),
    (r"from src\.query\.citation import", "from src.generation.citation import"),
    # catch-all
    (r"from src\.query\.", "from src."),
]

count_files = 0
count_repl = 0

for root, _dirs, files in os.walk(ROOT):
    for f in files:
        if not f.endswith(".py") or root.endswith("__pycache__"):
            continue
        # Skip the new slices themselves (already handled)
        rel = os.path.relpath(os.path.join(root, f), ROOT)
        if rel.startswith("retrieval") or rel.startswith("generation") or rel.startswith("agent"):
            continue
        path = os.path.join(root, f)
        with open(path, encoding="utf-8") as fh:
            content = fh.read()

        if "src.query" not in content:
            continue

        original = content
        for pattern, replacement in REPLACE:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                count_repl += 1
            content = new_content

        if content != original:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            count_files += 1
            print(f"  Updated: {rel}")

print(f"\n✅ {count_files} files updated, {count_repl} replacements")
