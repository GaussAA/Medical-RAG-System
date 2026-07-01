"""Base retriever — abstract class defining the retrieval interface."""
from abc import ABC, abstractmethod
from typing import Any

from src.common.models import RetrievedNode


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> list[RetrievedNode]:
        pass

    @abstractmethod
    async def add(self, nodes: list[RetrievedNode]) -> None:
        pass

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        pass
