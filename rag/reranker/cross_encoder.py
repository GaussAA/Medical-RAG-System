import gc
from typing import Any

import torch
from loguru import logger

from app.core.gpu_memory_manager import GPUMemoryManager
from app.models.schemas import RetrievedNode, RerankedNode
from config.settings import get_settings


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
        self.estimated_memory_mb = reranker_config.estimated_memory_mb

        self.model = None  # 懒加载，不立即加载模型
        self._model_on_gpu = False
        self._model_on_cpu = False  # 模型是否已加载到 CPU
        self.apply_normalization = True

    def _ensure_model_loaded(self) -> None:
        """确保模型已加载到 CPU"""
        if self.model is None:
            from sentence_transformers import CrossEncoder

            # 默认加载到 CPU
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device="cpu",  # 默认 CPU
            )
            self._model_on_cpu = True
            self._model_on_gpu = False

    def ensure_on_gpu(self) -> bool:
        """
        确保模型在 GPU 上。如果模型在 CPU，则迁移到 GPU。

        Returns:
            bool: 是否成功迁移到 GPU
        """
        gpu_manager = GPUMemoryManager.get_instance()

        # 如果已在 GPU 上，直接返回
        if self._model_on_gpu:
            return True

        # 确保模型已加载
        self._ensure_model_loaded()

        # 强制 GC + 缓存清理，获取准确的可用显存
        gc.collect()
        torch.cuda.empty_cache()

        # 检测 GPU 显存
        info = gpu_manager.get_memory_info()
        settings = get_settings()
        safety_margin = settings.models.gpu_safety_margin_mb
        usable = info["free_mb"] - safety_margin

        logger.debug(
            f"GPU memory check: required={self.estimated_memory_mb}MB, "
            f"free={info['free_mb']}MB, usable={usable}MB"
        )

        if usable < self.estimated_memory_mb:
            logger.warning(
                f"GPU memory insufficient for reranker, will use CPU: "
                f"required={self.estimated_memory_mb}MB, usable={usable}MB"
            )
            # GPU 显存不足时，不尝试加载 GPU，模型已在 CPU 上可用
            return False

        # 迁移到 GPU
        self.model.to("cuda")
        self._model_on_gpu = True
        self._model_on_cpu = False

        gpu_manager.register_model("reranker", self.estimated_memory_mb)

        logger.info(f"Reranker model moved to GPU ({self.estimated_memory_mb}MB)")
        return True

    def move_to_cpu(self) -> bool:
        """
        将模型从 GPU 迁移到 CPU。

        Returns:
            bool: 是否成功迁移
        """
        gpu_manager = GPUMemoryManager.get_instance()

        if not self._model_on_gpu:
            return True  # 不在 GPU 上，无需迁移

        # 确保模型已加载
        self._ensure_model_loaded()

        # 迁移到 CPU
        self.model.to("cpu")
        self._model_on_gpu = False
        self._model_on_cpu = True

        # 从 GPU 管理器注销
        gpu_manager.unregister_model("reranker")

        # 强制 GC + 缓存清理，最小化显存碎片
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Reranker model moved to CPU")
        return True

    def is_on_gpu(self) -> bool:
        """检测模型是否在 GPU 上"""
        return self._model_on_gpu

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedNode],
        return_documents: bool = True,
    ) -> list[RerankedNode]:
        if not candidates:
            return []

        # 确保模型已加载（可能在 CPU 上）
        self._ensure_model_loaded()

        self.ensure_on_gpu()

        pairs = [(query, node.content) for node in candidates]

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
