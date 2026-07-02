"""Vector retriever — Qdrant-based embedding search with lazy GPU model loading."""

import asyncio
from typing import Any

import torch
from loguru import logger

from src.common.config import get_settings
from src.common.models import RetrievedNode
from src.retrieval.base import BaseRetriever


class VectorRetriever(BaseRetriever):
    """Vector retriever with lazy singleton client and configurable timeout."""

    _client: Any = None

    def __init__(self, client: Any = None):
        settings = get_settings()
        self.config = settings.database.qdrant
        self.retrieval_config = settings.rag.retrieval

        if client is not None:
            VectorRetriever._client = client

        self.collection_name = self.config.collection
        self.top_k = self.retrieval_config.vector_top_k
        self.similarity_threshold = self.retrieval_config.similarity_threshold

        self._embedding_model = None

    @classmethod
    def _get_client(cls) -> Any:
        """Get or create singleton Qdrant client with config options."""
        if cls._client is None:
            from qdrant_client import QdrantClient

            cls._client = QdrantClient(
                url=cls._get_config().url,
                timeout=10,
                prefer_grpc=False,
            )
        return cls._client

    @classmethod
    def _get_config(cls) -> Any:
        """Get Qdrant config."""
        return get_settings().database.qdrant

    @classmethod
    def reset_client(cls) -> None:
        """Reset client (for testing)."""
        cls._client = None

    @property
    def client(self) -> Any:
        return VectorRetriever._get_client()

    @property
    def embedding_model(self):
        """Get embedding model, lazily loaded to configured device."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            settings = get_settings()
            embedding_model_name = settings.models.embedding.name
            device = settings.models.embedding.device
            self._embedding_model = SentenceTransformer(
                embedding_model_name,
                device=device,
                model_kwargs={"torch_dtype": torch.float16},
            )
        return self._embedding_model

    def load_embedding_to_gpu(self) -> bool:
        """No-op: embedding model is already on GPU (FP16)."""
        return True

    def move_embedding_to_cpu(self) -> bool:
        """No-op: embedding model stays on GPU permanently."""
        return True

    def is_on_gpu(self) -> bool:
        """Embedding model is always on GPU (FP16)."""
        return True

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Synchronous batch encoding for use in executor."""
        return self.embedding_model.encode(texts).tolist()

    async def retrieve(self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[RetrievedNode]:
        try:
            loop = asyncio.get_running_loop()
            query_vector = await loop.run_in_executor(None, lambda: self.embedding_model.encode(query).tolist())

            search_filter = self._build_filter(filters) if filters else None

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=search_filter,
                score_threshold=self.similarity_threshold,
            ).points

            nodes = []
            for result in results:
                payload = result.payload or {}
                node = RetrievedNode(
                    node_id=payload.get("node_id", str(result.id)),
                    content=payload.get("content", ""),
                    score=result.score,
                    metadata=payload,
                )
                nodes.append(node)

            return nodes

        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")
            return []

    async def add(self, nodes: list[RetrievedNode]) -> None:
        import asyncio
        import uuid as uuid_lib

        # Process nodes - encode any that don't have pre-encoded embeddings
        nodes_to_encode = []
        for node in nodes:
            if not node.metadata.get("embedding"):
                nodes_to_encode.append(node)

        # Batch encode only the nodes that need embeddings
        if nodes_to_encode:
            texts = [node.content for node in nodes_to_encode]
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(None, self._encode_batch, texts)
            # Fill in embeddings for nodes that needed encoding
            for i, node in enumerate(nodes_to_encode):
                node.metadata["embedding"] = embeddings[i]

        points = []
        for node in nodes:
            embedding = node.metadata.get("embedding", [])
            if not embedding:
                raise ValueError(f"No embedding for node {node.node_id}")

            # Build enriched payload with heading and content type info
            payload = {
                "content": node.content,
                "node_id": node.node_id,
                "chunk_id": node.metadata.get("chunk_id", node.node_id),
                "doc_id": node.metadata.get("doc_id", ""),
                "source_file": node.metadata.get("source_file", ""),
                "heading_tree": node.metadata.get("heading_tree", {}),
                "content_type": node.metadata.get("content_type", "text"),
                "section_title": node.metadata.get("section_title", ""),
                "position": node.metadata.get("position", 0),
            }

            point = {
                "id": str(uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, node.node_id)),
                "vector": embedding,
                "payload": payload,
            }
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    async def delete(self, ids: list[str]) -> None:
        from qdrant_client.models import PointIdsList

        logger.info(f"VectorRetriever.delete called with {len(ids)} IDs: {ids}")
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids),  # type: ignore[arg-type]
            )
            logger.info(f"Qdrant delete result: {result}")
        except Exception as e:
            logger.error(f"Qdrant delete failed: {e}")
            raise

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all points for a document by doc_id filter.
        More robust than delete by IDs because it doesn't depend on knowing chunk count.

        Returns the number of deleted points.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        logger.info(f"VectorRetriever.delete_by_doc_id called for doc_id: {doc_id}")
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
            )
            logger.info(f"Qdrant delete_by_doc_id result: {result}")
            # Note: result may not have count, but operation succeeded if no exception
            return 0  # We can't get count from delete result, caller should verify
        except Exception as e:
            logger.error(f"Qdrant delete_by_doc_id failed: {e}")
            raise

    def _build_filter(self, filters: dict[str, Any] | None) -> Any | None:
        if not filters:
            return None

        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            conditions = []

            if "doc_id" in filters:
                conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=filters["doc_id"])))

            if "source_file" in filters:
                conditions.append(
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=filters["source_file"]),
                    )
                )

            if "heading_id" in filters:
                conditions.append(FieldCondition(key="heading_id", match=MatchValue(value=filters["heading_id"])))

            if "content_type" in filters:
                conditions.append(
                    FieldCondition(
                        key="content_type",
                        match=MatchValue(value=filters["content_type"]),
                    )
                )

            if not conditions:
                return None

            return Filter(must=conditions)  # type: ignore[arg-type]

        except Exception as e:
            logger.warning(f"Failed to build filter: {e}")
            return None
