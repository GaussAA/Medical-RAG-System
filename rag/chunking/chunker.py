from abc import ABC, abstractmethod
from functools import lru_cache

from app.models.schemas import Chunk


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        pass

    @lru_cache(maxsize=1024)
    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def should_split(self, text: str, separators: list[str]) -> tuple[bool, str | None]:
        for sep in separators:
            if sep in text:
                return True, sep
        return False, None
