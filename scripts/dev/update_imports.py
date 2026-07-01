"""
Batch update import paths in src/ files.
Run this after copying files from old structure to new src/ structure.
"""
import os
import re


REPLACEMENTS = [
    # Config
    (r'from config\.settings import', 'from src.common.config.settings import'),
    (r'import config\.settings', 'import src.common.config.settings'),

    # Common models (was app.models.schemas)
    (r'from app\.models\.schemas import', 'from src.common.models import'),
    (r'from app\.models import', 'from src.common.database import'),

    # Database models (was app.models.database)
    (r'from app\.models\.database import', 'from src.common.database.models import'),
    (r'from app\.core\.database import', 'from src.common.database.engine import'),

    # Cache
    (r'from app\.core\.cache import', 'from src.common.cache.manager import'),

    # Metrics
    (r'from app\.core\.metrics import', 'from src.common.monitoring.metrics import'),

    # Safety
    (r'from app\.core\.safety import', 'from src.common.safety.checker import'),

    # RAG Engine
    (r'from app\.core\.rag_engine import', 'from src.query.engine import'),
    (r'from app\.core\.confidence import', 'from src.query.confidence import'),

    # Document service
    (r'from app\.services\.document import', 'from src.documents.service import'),
    (r'from app\.services\.document_processor import', 'from src.documents.processor import'),
    (r'from app\.services\.document_store import', 'from src.documents.store import'),
    (r'from app\.services\.retrieval_indexer import', 'from src.documents.indexer import'),
    (r'from app\.services\.consistency import', 'from src.conversation.consistency import'),

    # Session
    (r'from app\.services\.session import', 'from src.conversation.manager import'),

    # Citation
    (r'from app\.services\.citation_verifier import', 'from src.query.citation.verifier import'),

    # rag imports
    (r'from rag\.generation\.llm_generator import', 'from src.query.generation.generator import'),
    (r'from rag\.generation\.warnings_generator import', 'from src.query.generation.warnings import'),
    (r'from rag\.generation\.prompt import', 'from src.query.generation.prompt import'),
    (r'from rag\.generation\.prompt_builder import', 'from src.query.generation.prompt_builder import'),
    (r'from rag\.reranker\.cross_encoder import', 'from src.query.reranker.cross_encoder import'),
    (r'from rag\.retrieval\.hybrid_retriever import', 'from src.query.retrieval.hybrid import'),
    (r'from rag\.retrieval\.vector_retriever import', 'from src.query.retrieval.vector import'),
    (r'from rag\.retrieval\.bm25_retriever import', 'from src.query.retrieval.bm25 import'),
    (r'from rag\.retrieval\.query_boosting import', 'from src.query.retrieval.boosting import'),
    (r'from rag\.retrieval\.base import', 'from src.query.retrieval.base import'),
    (r'from rag\.evaluation\.evaluator import', 'from src.evaluation.evaluator import'),
    (r'from rag\.evaluation\.retrieval_eval import', 'from src.evaluation.retrieval_eval import'),
    (r'from rag\.evaluation\.generation_eval import', 'from src.evaluation.generation_eval import'),
    (r'from rag\.evaluation\.medical_safety_eval import', 'from src.evaluation.medical_safety_eval import'),
    (r'from rag\.evaluation\.benchmark_runner import', 'from src.evaluation.benchmark_runner import'),
    (r'from rag\.evaluation\.dataset_manager import', 'from src.evaluation.dataset_manager import'),
    (r'from rag\.evaluation\.synthetic_generator import', 'from src.evaluation.synthetic_generator import'),
    (r'from rag\.evaluation\.interfaces import', 'from src.evaluation.interfaces import'),
    (r'from rag\.evaluation\.cli import', 'from src.evaluation.cli import'),

    # Evaluation reporters
    (r'from rag\.evaluation\.reporters\.csv_reporter import', 'from src.evaluation.reporters.csv import'),
    (r'from rag\.evaluation\.reporters\.html_reporter import', 'from src.evaluation.reporters.html import'),
    (r'from rag\.evaluation\.reporters\.json_reporter import', 'from src.evaluation.reporters.json import'),

    # rag.evaluation internal imports (from within the evaluation package)
    (r'from rag\.evaluation\.reporters\.csv_reporter import', 'from src.evaluation.reporters.csv import'),
    (r'from rag\.evaluation\.reporters\.html_reporter import', 'from src.evaluation.reporters.html import'),
    (r'from rag\.evaluation\.reporters\.json_reporter import', 'from src.evaluation.reporters.json import'),
]

# For rag.evaluation internal references (e.g., evaluation modules referencing each other)
INTERNAL_EVAL_REPLACEMENTS = [
    (r'from evaluation\.', 'from src.evaluation.'),
]

# Also handle the parser and chunker import patterns
PARSER_CHUNKER_REPLACEMENTS = [
    (r'from rag\.parser\.markdown_parser import', 'from src.documents.parser import'),
    (r'from rag\.chunking\.hierarchical_chunker import', 'from src.documents.chunker import'),
    (r'from rag\.chunking\.semantic_chunker import', 'from src.documents.chunker import'),
    (r'from rag\.chunking\.chunker import', 'from src.documents.chunker import'),
]

# Import from app.api.deps - keep as-is but need to handle the deps module reference
# For now we just leave them, they'll be migrated later

# Import within rag package (e.g., rag.retrieval imports rag.retrieval.xxx)
INTERNAL_RAG_REPLACEMENTS = [
    (r'from rag\.retrieval\.', 'from src.query.retrieval.'),
    (r'from rag\.reranker\.', 'from src.query.reranker.'),
    (r'from rag\.generation\.', 'from src.query.generation.'),
    (r'from rag\.evaluation\.', 'from src.evaluation.'),
    (r'from rag\.parser\.', 'from src.documents.'),
    (r'from rag\.chunking\.', 'from src.documents.'),
]

# Combine all replacements
ALL_REPLACEMENTS = REPLACEMENTS + PARSER_CHUNKER_REPLACEMENTS + INTERNAL_RAG_REPLACEMENTS


def update_imports_in_file(filepath):
    """Update import paths in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content
    for pattern, replacement in ALL_REPLACEMENTS:
        content = re.sub(pattern, replacement, content)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    # Project root is two levels up from scripts/dev/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src_dir = os.path.join(project_root, 'src')
    updated_files = []

    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.endswith('.py') and f != '__init__.py':
                filepath = os.path.join(root, f)
                if update_imports_in_file(filepath):
                    updated_files.append(os.path.relpath(filepath, src_dir))

    print(f"Total files updated: {len(updated_files)}")
    for f in sorted(updated_files):
        print(f"  + {f}")


if __name__ == '__main__':
    main()
