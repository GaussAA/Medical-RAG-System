from abc import ABC, abstractmethod
from typing import Any

from app.models.schemas import RetrievedNode


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
