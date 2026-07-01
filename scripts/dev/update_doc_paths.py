"""Batch-update file path references in docs/ to match the new src/* architecture."""
import os
import re

# Mapping: old_path_pattern -> new_path
# Order matters: more specific patterns first
REPLACE = [
    # app/core/ -> src/common/ or src/query/
    (r"`app/core/rag_engine`", "`src/query/engine`"),
    (r"`app/core/database`", "`src/common/database/engine`"),
    (r"`app/core/cache`", "`src/common/cache/manager`"),
    (r"`app/core/safety`", "`src/common/safety/checker`"),
    (r"`app/core/confidence`", "`src/query/confidence`"),
    (r"`app/core/risk_warnings`", "`src/query/generation/warnings`"),
    (r"`app/core/metrics`", "`src/common/monitoring/metrics`"),

    # app/core/ links
    (r"\[.*?\]\(\.\./\.\./app/core/rag_engine\.py\)", "`src/query/engine.py`"),
    (r"\[.*?\]\(\.\./\.\./app/core/safety\.py\)", "`src/common/safety/checker.py`"),
    (r"\[.*?\]\(\.\./\.\./app/core/confidence\.py\)", "`src/query/confidence.py`"),
    (r"\[.*?\]\(\.\./\.\./app/core/risk_warnings\.py\)", "`src/query/generation/warnings.py`"),
    (r"\[.*?\]\(\.\./\.\./app/core/\.py\)", "`src/...` (see new structure)"),

    # app/services/ -> src/*/
    (r"`app/services/document_store", "`src/documents/store"),
    (r"`app/services/document_processor", "`src/documents/processor"),
    (r"`app/services/document`", "`src/documents/service`"),
    (r"`app/services/session`", "`src/conversation/manager`"),
    (r"`app/services/citation_verifier`", "`src/query/citation/verifier`"),
    (r"`app/services/consistency`", "`src/conversation/consistency`"),
    (r"`app/services/retrieval_indexer", "`src/documents/indexer"),

    # app/services/ links
    (r"\[.*?\]\(\.\./\.\./app/services/session\.py\)", "`src/conversation/manager.py`"),
    (r"\[.*?\]\(\.\./\.\./app/services/document\.py\)", "`src/documents/service.py`"),
    (r"\[.*?\]\(\.\./\.\./app/services/citation_verifier\.py\)", "`src/query/citation/verifier.py`"),
    (r"\[.*?\]\(\.\./\.\./app/services/consistency\.py\)", "`src/conversation/consistency.py`"),

    # app/api/routes/ -> src/*/api.py
    (r"`app/api/routes/documents", "`src/documents/api`"),
    (r"`app/api/routes/query", "`src/query/api`"),
    (r"`app/api/routes/conversation", "`src/conversation/api`"),
    (r"`app/api/routes/evaluation", "`src/evaluation/api`"),

    (r"\[.*?\]\(\.\./\.\./app/api/routes/documents\.py\)", "`src/documents/api.py`"),
    (r"\[.*?\]\(\.\./\.\./app/api/routes/query\.py\)", "`src/query/api.py`"),

    # app/models/ -> src/common/
    (r"`app/models/schemas", "`src/common/models`"),
    (r"`app/models/database", "`src/common/database/models`"),

    (r"\[.*?\]\(\.\./\.\./app/models/schemas\.py\)", "`src/common/models.py`"),
    (r"\[.*?\]\(\.\./\.\./app/models/database\.py\)", "`src/common/database/models.py`"),

    # app/main.py -> src/main.py
    (r"\[.*?\]\(\.\./\.\./app/main\.py\)", "`src/main.py`"),

    # app/ references to known modules
    (r"\"app/core/rag_engine\.py\"", "\"src/query/engine.py\""),
    (r"`app/core/gpu_memory_manager`", "`[removed] — GPU models now FP16 permanent resident`"),
    (r"`app/core/deps", "`src/common/di/deps"),

    # General app/ -> src/ (catch less specific)
    (r"`app/services/", "`src/"),
    (r"`app/core/", "`src/"),
    (r"`app/models/", "`src/common/"),
    (r"`app/api/routes/", "`src/"),
    (r"`app/main", "`src/main"),

    # rag/ -> src/query/ or src/documents/
    (r"`rag/retrieval/hybrid_retriever", "`src/query/retrieval/hybrid"),
    (r"`rag/retrieval/vector_retriever", "`src/query/retrieval/vector"),
    (r"`rag/retrieval/bm25_retriever", "`src/query/retrieval/bm25"),
    (r"`rag/retrieval/base", "`src/query/retrieval/base"),
    (r"`rag/reranker/cross_encoder", "`src/query/reranker/cross_encoder"),
    (r"`rag/generation/llm_generator", "`src/query/generation/generator"),
    (r"`rag/generation/prompt", "`src/query/generation/prompt"),
    (r"`rag/generation/warnings_generator", "`src/query/generation/warnings"),
    (r"`rag/generation/prompt_builder", "`src/query/generation/prompt_builder"),
    (r"`rag/parser/", "`src/documents/parser/"),
    (r"`rag/chunking/hierarchical_chunker", "`src/documents/chunker"),
    (r"`rag/evaluation/evaluator", "`src/evaluation/evaluator"),
    (r"`rag/evaluation/", "`src/evaluation/"),
    (r"```rag/", "```src/"),

    # rag/ links
    (r"\[.*?\]\(\.\./\.\./rag/retrieval/hybrid_retriever\.py\)", "`src/query/retrieval/hybrid.py`"),
    (r"\[.*?\]\(\.\./\.\./rag/retrieval/vector_retriever\.py\)", "`src/query/retrieval/vector.py`"),
    (r"\[.*?\]\(\.\./\.\./rag/retrieval/bm25_retriever\.py\)", "`src/query/retrieval/bm25.py`"),
    (r"\[.*?\]\(\.\./\.\./rag/reranker/cross_encoder\.py\)", "`src/query/reranker/cross_encoder.py`"),
    (r"\[.*?\]\(\.\./\.\./rag/generation/llm_generator\.py\)", "`src/query/generation/generator.py`"),
    (r"\[.*?\]\(\.\./\.\./rag/parser/markdown_parser\.py\)", "`src/documents/parser/markdown_parser.py`"),
    (r"\[.*?\]\(\.\./\.\./rag/chunking/hierarchical_chunker\.py\)", "`src/documents/chunker.py`"),
    (r"\[.*?\]\(\.\./\.\./rag/\.py\)", "`src/...` (see new structure)"),

    # config/ -> src/common/config/
    (r"`config/settings", "`src/common/config/settings"),
    (r"`config/config", "`src/common/config/config"),

    # streamlit_app/ -> frontend/
    (r"`streamlit_app/app", "`frontend/app"),
    (r"`streamlit_app/", "`frontend/"),
    (r"streamlit_app/app\.py", "frontend/app.py"),

    # Migration references
    (r"`app/core/gpu_memory_manager\.py`", "`[removed]`"),

    # Architecture section titles
    (r"# API Layer \(`app/api/routes/`\)", "# API Layer (`src/*/api.py`)"),
]

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs")

count_files = 0
count_replacements = 0

for root, _dirs, files in os.walk(DOCS_DIR):
    for f in files:
        if not f.endswith(".md"):
            continue
        path = os.path.join(root, f)
        with open(path, encoding="utf-8") as fh:
            content = fh.read()

        original = content
        for pattern, replacement in REPLACE:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                count_replacements += 1
            content = new_content

        if content != original:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            count_files += 1
            print(f"  Updated: {os.path.relpath(path, DOCS_DIR)}")

print(f"\n✅ {count_files} files updated, {count_replacements} replacements")
