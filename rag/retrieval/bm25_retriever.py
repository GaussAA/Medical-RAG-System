import asyncio
import json
from pathlib import Path
from typing import Any

import jieba  # type: ignore[import-untyped]
from loguru import logger
from rank_bm25 import BM25Plus  # type: ignore[import-untyped]

from app.models.schemas import RetrievedNode
from rag.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self, persist_path: str | None = None):
        self.documents: list[dict[str, Any]] = []
        self.bm25: BM25Plus | None = None
        self.corpus: list[str] = []
        self.persist_path = persist_path

        if self.persist_path:
            self._try_load()

    def _try_load(self) -> None:
        """Try to load BM25 index from persisted file."""
        if not self.persist_path:
            return

        path = Path(self.persist_path)
        if not path.exists():
            logger.debug(f"BM25 persist file not found: {self.persist_path}")
            return

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            self.documents = data.get("documents", [])
            self.corpus = [doc["content"] for doc in self.documents]

            if self.corpus:
                # Try to load pre-tokenized corpus to avoid expensive tokenization on restart
                tokenized_corpus = data.get("tokenized_corpus")
                if tokenized_corpus:
                    self.bm25 = BM25Plus(tokenized_corpus)
                    logger.info(f"BM25 index loaded from {self.persist_path} (pre-tokenized)")
                else:
                    # Fallback: tokenize on load
                    tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
                    self.bm25 = BM25Plus(tokenized_corpus)
                    logger.info(f"BM25 index loaded from {self.persist_path} (tokenized on load)")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index from {self.persist_path}: {e}")
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild BM25 index from documents."""
        if not self.corpus:
            self.bm25 = None
            return

        tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        self.bm25 = BM25Plus(tokenized_corpus)
        logger.info("BM25 index rebuilt")

    def _save(self) -> None:
        """Persist BM25 index to disk."""
        if not self.persist_path:
            logger.warning(
                "BM25 persist_path is not configured. "
                "BM25 index will not be persisted to disk and will be lost on restart."
            )
            return

        try:
            path = Path(self.persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "documents": self.documents,
                "tokenized_corpus": [list(jieba.cut(doc)) for doc in self.corpus] if self.corpus else [],
            }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

            logger.debug(f"BM25 index saved to {self.persist_path}")
        except Exception as e:
            logger.warning(f"Failed to save BM25 index to {self.persist_path}: {e}")

    async def retrieve(self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[RetrievedNode]:
        if not self.bm25 or not self.corpus:
            return []

        loop = asyncio.get_running_loop()
        tokenized_query = await loop.run_in_executor(None, lambda: list(jieba.cut(query)))
        assert self.bm25 is not None
        bm25 = self.bm25
        scores = await loop.run_in_executor(None, lambda: bm25.get_scores(tokenized_query))

        scored_docs = list(enumerate(scores))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: scored_docs.sort(key=lambda x: x[1], reverse=True))

        results = []
        for idx, score in scored_docs[: top_k * 3]:  # Get more to allow filtering
            if score > 0:
                doc = self.documents[idx]
                metadata = doc.get("metadata", {})

                # Apply filters if provided
                if filters:
                    if filters.get("doc_id") and metadata.get("doc_id") != filters["doc_id"]:
                        continue
                    if filters.get("source_file") and metadata.get("source_file") != filters["source_file"]:
                        continue
                    if filters.get("heading_id") and metadata.get("heading_id") != filters["heading_id"]:
                        continue
                    if filters.get("content_type") and metadata.get("content_type") != filters["content_type"]:
                        continue

                node = RetrievedNode(
                    node_id=doc.get("id", str(idx)),
                    content=doc.get("content", ""),
                    score=float(score),
                    metadata=doc.get("metadata", {}),
                )
                results.append(node)

                if len(results) >= top_k:
                    break

        return results

    async def add(self, nodes: list[RetrievedNode]) -> None:
        for node in nodes:
            # Build metadata for BM25 (content stored at top level only, not duplicated in metadata)
            metadata = {
                "doc_id": node.metadata.get("doc_id", ""),
                "source_file": node.metadata.get("source_file", ""),
                "heading_tree": node.metadata.get("heading_tree", {}),
                "content_type": node.metadata.get("content_type", "text"),
                "section_title": node.metadata.get("section_title", ""),
                "position": node.metadata.get("position", 0),
            }
            self.documents.append(
                {
                    "id": node.node_id,
                    "content": node.content,
                    "metadata": metadata,
                }
            )

        self.corpus = [doc["content"] for doc in self.documents]

        if self.corpus:
            tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
            self.bm25 = BM25Plus(tokenized_corpus)
            self._save()

    async def delete(self, ids: list[str]) -> None:
        remaining_docs = []
        for doc in self.documents:
            if doc.get("id") not in ids:
                remaining_docs.append(doc)

        self.documents = remaining_docs
        self.corpus = [doc["content"] for doc in self.documents]

        if self.corpus:
            tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
            self.bm25 = BM25Plus(tokenized_corpus)
        else:
            self.bm25 = None

        self._save()

    def clear(self) -> None:
        self.documents = []
        self.corpus = []
        self.bm25 = None
