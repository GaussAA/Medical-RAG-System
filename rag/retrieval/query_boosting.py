import re
from typing import Any

from app.models.schemas import RetrievedNode


class QueryBoosting:
    """Handles query type detection and content-type boosting."""

    TABLE_PATTERNS = [
        r"表[一二三四五六七八九十\d]+",
        r"表格", r"table",
    ]
    LIST_PATTERNS = [
        r"列出", r"列表中", r"列表项", r"哪些.*列表", r"list",
    ]
    DRUG_PATTERNS = [
        r"剂量", r"用法", r"每次", r"每日", r"mg",
        r"毫升", r"不良反应", r"禁忌", r"药物",
    ]

    def detect_query_type(self, query: str) -> str | None:
        """Detect if query asks about specific content types."""
        query_lower = query.lower()
        for pattern in self.TABLE_PATTERNS:
            if re.search(pattern, query_lower):
                return "table"
        for pattern in self.LIST_PATTERNS:
            if re.search(pattern, query_lower):
                return "list"
        for pattern in self.DRUG_PATTERNS:
            if re.search(pattern, query_lower):
                return "list"
        return None

    def boost_by_content_type(
        self, results: list[RetrievedNode], target_type: str
    ) -> list[RetrievedNode]:
        """Boost scores for chunks matching target content type."""
        boost_factor = 1.3
        boosted = []
        for node in results:
            content_type = node.metadata.get("content_type", "text")
            new_score = node.score * boost_factor if content_type == target_type else node.score
            boosted.append(RetrievedNode(
                node_id=node.node_id,
                content=node.content,
                score=new_score,
                metadata=node.metadata,
            ))
        boosted.sort(key=lambda x: x.score, reverse=True)
        return boosted