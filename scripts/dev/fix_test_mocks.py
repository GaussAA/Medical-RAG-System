"""Fix old mock/patch paths in test files to match new src/* architecture."""
import os
import re
from pathlib import Path

REPLACE = [
    # hybrid_retriever tests
    (r'"rag\.retrieval\.hybrid_retriever\.get_settings"', '"src.query.retrieval.hybrid.get_settings"'),
    (r"'rag\.retrieval\.hybrid_retriever\.get_settings'", "'src.query.retrieval.hybrid.get_settings'"),

    # cross_encoder tests
    (r'"rag\.reranker\.cross_encoder\.get_settings"', '"src.query.reranker.cross_encoder.get_settings"'),
    (r"'rag\.reranker\.cross_encoder\.get_settings'", "'src.query.reranker.cross_encoder.get_settings'"),

    # document_service tests  
    (r'"rag\.parser\.parse_document_with_headings"', '"src.documents.parser.parse_document_with_headings"'),
    (r"'rag\.parser\.parse_document_with_headings'", "'src.documents.parser.parse_document_with_headings'"),

    # integration tests
    (r'"app\.core\.rag_engine\.RAGEngine\.query"', '"src.query.engine.RAGEngine.query"'),
    (r"'app\.core\.rag_engine\.RAGEngine\.query'", "'src.query.engine.RAGEngine.query'"),

    # cache tests
    (r'"app\.core\.cache\.get_settings"', '"src.common.cache.manager.get_settings"'),
    (r"'app\.core\.cache\.get_settings'", "'src.common.cache.manager.get_settings'"),
    (r'"app\.core\.cache\.redis\.Redis"', '"src.common.cache.manager.redis.Redis"'),
    (r"'app\.core\.cache\.redis\.Redis'", "'src.common.cache.manager.redis.Redis'"),
    (r'"app\.core\.cache\.CacheManager\.get_instance"', '"src.common.cache.manager.CacheManager.get_instance"'),
    (r"'app\.core\.cache\.CacheManager\.get_instance'", "'src.common.cache.manager.CacheManager.get_instance'"),

    # session tests
    (r'"app\.services\.session\.get_session_factory"', '"src.common.database.engine.get_session_factory"'),
    (r"'app\.services\.session\.get_session_factory'", "'src.common.database.engine.get_session_factory'"),
]

test_root = Path(__file__).parent.parent.parent / "tests"

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
            print(f"  Fixed: {os.path.relpath(path, test_root)}")

print(f"\n✅ {count} test files fixed")
