import threading
from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger


@dataclass
class GPUMemoryStatus:
    """Unified GPU memory status interface."""

    allocated_gb: float
    reserved_gb: float
    total_gb: float
    free_gb: float
    is_known: bool = True

    @classmethod
    def unknown(cls) -> "GPUMemoryStatus":
        return cls(0, 0, 0, 0, is_known=False)


def get_gpu_memory_status() -> GPUMemoryStatus:
    """Get unified GPU memory status."""
    try:
        if not torch.cuda.is_available():
            return GPUMemoryStatus.unknown()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        return GPUMemoryStatus(
            allocated_gb=allocated,
            reserved_gb=reserved,
            total_gb=total,
            free_gb=free,
            is_known=True,
        )
    except Exception:
        return GPUMemoryStatus.unknown()


class GPUMemoryManager:
    """
    GPU 显存统一管理者，提供显存查询、模型加载检测、模型迁移协调功能。

    采用单例模式，确保全局唯一的显存状态管理。
    """

    _instance: Optional["GPUMemoryManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._loaded_models: dict[str, int] = {}  # model_name -> memory_mb
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_instance(cls) -> "GPUMemoryManager":
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_memory_info(self) -> dict:
        """
        返回 GPU 显存状态信息。

        Returns:
            dict: 包含 total_mb, used_mb, free_mb, available
        """
        if not torch.cuda.is_available():
            return {
                "available": False,
                "total_mb": 0,
                "used_mb": 0,
                "free_mb": 0,
            }

        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        allocated = torch.cuda.memory_allocated(0) / (1024**2)
        reserved = torch.cuda.memory_reserved(0) / (1024**2)

        # Use total - reserved as the definition of "free" memory
        # This excludes PyTorch cached memory for consistency with get_gpu_memory_status()
        true_free_mb = total - reserved

        return {
            "available": True,
            "total_mb": round(total, 2),
            "used_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "free_mb": round(true_free_mb, 2),
        }

    def get_loaded_models(self) -> list[dict]:
        """
        获取当前在 GPU 上的模型列表。

        Returns:
            list[dict]: 每个模型包含 name 和 memory_mb
        """
        return [{"name": name, "memory_mb": mem} for name, mem in self._loaded_models.items()]

    def is_model_loaded(self, model_name: str) -> bool:
        """检测指定模型是否在 GPU 上"""
        return model_name in self._loaded_models

    def can_load_model(self, model_name: str, required_mb: float) -> bool:
        """
        检测 GPU 剩余显存是否足够加载指定模型。

        Args:
            model_name: 模型名称（用于记录）
            required_mb: 所需显存（MB）

        Returns:
            bool: 是否可以加载
        """
        if not torch.cuda.is_available():
            return False

        info = self.get_memory_info()
        if not info["available"]:
            return False

        available_for_new = info["free_mb"]

        from config.settings import get_settings

        safe_margin = get_settings().models.gpu_safety_margin_mb
        usable = available_for_new - safe_margin

        return usable >= required_mb

    def register_model(self, model_name: str, memory_mb: float) -> bool:
        """
        记录模型已加载到 GPU。

        Args:
            model_name: 模型名称
            memory_mb: 预估占用显存

        Returns:
            bool: 是否注册成功（如果显存不足则失败）
        """
        if model_name in self._loaded_models:
            logger.debug(f"Model {model_name} already registered")
            return True

        if not self.can_load_model(model_name, memory_mb):
            return False

        self._loaded_models[model_name] = int(memory_mb)
        logger.info(f"Registered {model_name} on GPU: {memory_mb}MB")
        return True

    def unregister_model(self, model_name: str) -> bool:
        """
        记录模型已从 GPU 移除。

        Args:
            model_name: 模型名称

        Returns:
            bool: 是否成功移除
        """
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            logger.info(f"Unregistered {model_name} from GPU")
            return True
        return False

    def move_model_to_cpu(self, model_name: str) -> bool:
        """
        通知 GPUMemoryManager 指定模型即将迁移到 CPU。
        实际迁移由模型自身执行，此方法仅更新状态记录。
        """
        if model_name in self._loaded_models:
            logger.info(f"Model {model_name} moving to CPU")
            return True
        return False

    def move_model_to_gpu(self, model_name: str) -> bool:
        """
        通知 GPUMemoryManager 指定模型即将迁移到 GPU。
        实际迁移由模型自身执行，此方法仅更新状态记录。
        """
        can_load = self.can_load_model(model_name, self._loaded_models.get(model_name, 0))
        if can_load:
            logger.info(f"Model {model_name} moving to GPU")
            return True
        return False

    def reset(self):
        """重置管理器状态（用于测试）"""
        self._loaded_models.clear()
