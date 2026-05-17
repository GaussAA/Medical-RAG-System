from typing import Any

from loguru import logger

from app.models.schemas import RetrievedNode
from config.settings import get_settings
from rag.retrieval.base import BaseRetriever


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
        self._embedding_on_gpu = False
        self._embedding_memory_mb = settings.models.embedding.estimated_memory_mb

    @classmethod
    def _get_client(cls) -> Any:
        """Get or create singleton Qdrant client with config options."""
        if cls._client is None:
            from qdrant_client import QdrantClient

            cls._client = QdrantClient(
                url=cls._get_config().url,
                timeout=cls._get_config().timeout,
                prefer_grpc=cls._get_config().prefer_grpc,
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
        """Get embedding model, default CPU."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            settings = get_settings()
            embedding_model_name = settings.models.embedding.name
            device = "cpu"
            self._embedding_model = SentenceTransformer(embedding_model_name, device=device)
            self._embedding_on_gpu = False
        return self._embedding_model

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Synchronous batch encoding for use in executor."""
        return self.embedding_model.encode(texts).tolist()

    def load_embedding_to_gpu(self) -> bool:
        """Load embedding model to GPU."""
        from app.core.gpu_memory_manager import GPUMemoryManager, get_gpu_memory_status

        if self._embedding_on_gpu:
            return True

        _ = self.embedding_model

        settings = get_settings()
        safety_margin_mb = settings.models.gpu_safety_margin_mb
        safety_margin_gb = safety_margin_mb / 1024

        status = get_gpu_memory_status()
        usable_gb = status.free_gb - safety_margin_gb
        required_gb = self._embedding_memory_mb / 1024

        logger.debug(
            f"GPU memory check for embedding: required={required_gb:.2f}GB, "
            f"free={status.free_gb:.2f}GB, usable={usable_gb:.2f}GB"
        )

        if usable_gb < required_gb:
            logger.warning(
                f"GPU memory insufficient for embedding: "
                f"required={required_gb:.2f}GB, usable={usable_gb:.2f}GB"
            )
            return False

        self._embedding_model.to("cuda")
        self._embedding_on_gpu = True

        gpu_manager = GPUMemoryManager.get_instance()
        gpu_manager.register_model("embedding", self._embedding_memory_mb)

        logger.info(f"Embedding model loaded to GPU ({self._embedding_memory_mb}MB)")
        return True

    def move_embedding_to_cpu(self) -> bool:
        """Move embedding model from GPU to CPU."""
        from app.core.gpu_memory_manager import GPUMemoryManager

        gpu_manager = GPUMemoryManager.get_instance()

        if not self._embedding_on_gpu:
            return True

        self._embedding_model.to("cpu")
        self._embedding_on_gpu = False

        gpu_manager.unregister_model("embedding")

        import torch

        torch.cuda.empty_cache()

        logger.info("Embedding model moved to CPU")
        return True

    def is_on_gpu(self) -> bool:
        """Check if embedding model is on GPU."""
        return self._embedding_on_gpu

    async def retrieve(
        self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None
    ) -> list[RetrievedNode]:
        try:
            query_vector = self.embedding_model.encode(query).tolist()

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
            embeddings = await loop.run_in_executor(
                None, self._encode_batch, texts
            )
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
                points_selector=PointIdsList(points=ids),
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
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        logger.info(f"VectorRetriever.delete_by_doc_id called for doc_id: {doc_id}")
        try:
            result = self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                    ]
                ),
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
                conditions.append(
                    FieldCondition(key="doc_id", match=MatchValue(value=filters["doc_id"]))
                )

            if "source_file" in filters:
                conditions.append(
                    FieldCondition(
                        key="source_file", match=MatchValue(value=filters["source_file"])
                    )
                )

            if "heading_id" in filters:
                conditions.append(
                    FieldCondition(
                        key="heading_id", match=MatchValue(value=filters["heading_id"])
                    )
                )

            if "content_type" in filters:
                conditions.append(
                    FieldCondition(
                        key="content_type", match=MatchValue(value=filters["content_type"])
                    )
                )

            if not conditions:
                return None

            return Filter(must=conditions)

        except Exception as e:
            logger.warning(f"Failed to build filter: {e}")
            return None
