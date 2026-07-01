import os
import re
from pathlib import Path

REPLACE = [
    (r"from app\.core\.cache import", "from src.common.cache.manager import"),
    (r"from app\.core\.confidence import", "from src.query.confidence import"),
    (r"from app\.core\.rag_engine import", "from src.query.engine import"),
    (r"from app\.core\.safety import", "from src.common.safety.checker import"),
    (r"from app\.core\.metrics import ", "from src.common.monitoring.metrics import "),
    (r"from app\.core\.database import", "from src.common.database.engine import"),
    (r"from app\.models\.schemas import ", "from src.common.models import "),
    (r"from app\.models\.database import ", "from src.common.database.models import "),
    (r"from app\.services\.document import", "from src.documents.service import"),
    (r"from app\.services\.document_store import", "from src.documents.store import"),
    (r"from app\.services\.document_processor import", "from src.documents.processor import"),
    (r"from app\.services\.retrieval_indexer import", "from src.documents.indexer import"),
    (r"from app\.services\.session import", "from src.conversation.manager import"),
    (r"from app\.services\.citation_verifier import", "from src.query.citation.verifier import"),
    (r"from app\.api\.deps import", "from src.common.di.deps import"),
    (r"from rag\.reranker\.cross_encoder import", "from src.query.reranker.cross_encoder import"),
    (r"from rag\.generation\.llm_generator import", "from src.query.generation.generator import"),
    (r"from rag\.generation\.warnings_generator import", "from src.query.generation.warnings import"),
    (r"from rag\.generation\.prompt import", "from src.query.generation.prompt import"),
    (r"from rag\.retrieval\.hybrid_retriever import", "from src.query.retrieval.hybrid import"),
    (r"from rag\.retrieval\.vector_retriever import", "from src.query.retrieval.vector import"),
    (r"from rag\.retrieval\.bm25_retriever import", "from src.query.retrieval.bm25 import"),
    (r"from rag\.retrieval\.base import", "from src.query.retrieval.base import"),
    (r"from rag\.chunking\.hierarchical_chunker import", "from src.documents.chunker import"),
    (r"from rag\.parser\.markdown_parser import", "from src.documents.parser import"),
    (r"from rag\.evaluation\.evaluator import", "from src.evaluation.evaluator import"),
    (r"from rag\.evaluation\.retrieval_eval import", "from src.evaluation.retrieval_eval import"),
    (r"from rag\.evaluation\.generation_eval import", "from src.evaluation.generation_eval import"),
    (r"from rag\.evaluation\.benchmark_runner import", "from src.evaluation.benchmark_runner import"),
    (r"from rag\.evaluation\.dataset_manager import", "from src.evaluation.dataset_manager import"),
    (r"from config\.settings import", "from src.common.config.settings import"),
]

test_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

count = 0
for root, _dirs, files in os.walk(test_root):
    for f in files:
        if not f.endswith(".py"):
            continue
        path = os.path.join(root, f)
        with open(path, encoding="utf-8") as fh:
            content = fh.read()

        original = content
        for pattern, replacement in REPLACE:
            content = re.sub(pattern, replacement, content)

        if content != original:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            count += 1
            print(f"  Updated: {os.path.relpath(path, test_root)}")

print(f"\n✅ {count} test files updated")
