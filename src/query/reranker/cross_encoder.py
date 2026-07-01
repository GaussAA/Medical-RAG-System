"""Cross-encoder reranker — re-ranks retrieved candidates using a CrossEncoder model."""
from typing import Any

import torch

from src.common.config.settings import get_settings
from src.common.models import RerankedNode, RetrievedNode


class Reranker:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
    ):
        settings = get_settings()
        reranker_config = settings.models.reranker

        self.model_name = model_name or reranker_config.name
        self.device = device or reranker_config.device
        self.batch_size = batch_size or reranker_config.batch_size
        self.max_length = max_length or reranker_config.max_length

        self.model = None  # 懒加载，不立即加载模型
        self.apply_normalization = True

    def _ensure_model_loaded(self) -> None:
        """确保模型已加载到 GPU (FP16)。"""
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
                model_kwargs={"torch_dtype": torch.float16},
            )

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedNode],
        return_documents: bool = True,
    ) -> list[RerankedNode]:
        if not candidates:
            return []

        # 确保模型已加载（懒加载到 GPU）
        self._ensure_model_loaded()

        pairs = [(query, node.content) for node in candidates]

        assert self.model is not None
        scores = self.model.predict(pairs, batch_size=self.batch_size)

        if self.apply_normalization:
            scores = self._normalize_scores(scores)

        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for node, score in scored_candidates:
            reranked_node = RerankedNode(
                node_id=node.node_id,
                content=node.content if return_documents else "",
                score=float(score),
                metadata=node.metadata,
            )
            reranked.append(reranked_node)

        return reranked

    def _normalize_scores(self, scores: Any) -> list[float]:
        if isinstance(scores, (list, tuple)):
            scores_list = scores
        else:
            scores_list = scores.tolist() if hasattr(scores, "tolist") else [scores]

        if not scores_list:
            return []

        min_score = min(scores_list)
        max_score = max(scores_list)

        if max_score == min_score:
            return [0.5] * len(scores_list)

        normalized = [(s - min_score) / (max_score - min_score) for s in scores_list]

        return normalized

    def get_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
