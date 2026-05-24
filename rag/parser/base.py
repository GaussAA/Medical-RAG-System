from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.models.schemas import ParsedDocument, TableData


class BaseParser(ABC):
    supported_extensions: list[str] = []

    @classmethod
    def can_parse(cls, file_path: str | Path) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in cls.supported_extensions

    @abstractmethod
    async def parse(self, file_path: str | Path) -> ParsedDocument:
        pass

    def extract_tables(self, content: Any) -> list[TableData]:
        return []
