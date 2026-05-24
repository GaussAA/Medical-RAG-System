from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import RetrievedNode


@dataclass
class ConfidenceConfig:
    """Confidence evaluation configuration for dependency injection."""

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "context_relevance": 0.5,
            "answer_completeness": 0.15,
            "consistency": 0.15,
            "source_reliability": 0.2,
        }
    )


class ConfidenceEvaluator:
    DEFAULT_WEIGHTS = {
        "context_relevance": 0.5,
        "answer_completeness": 0.15,
        "consistency": 0.15,
        "source_reliability": 0.2,
    }

    def __init__(self, config: ConfidenceConfig | None = None):
        if config is not None:
            self.weights = config.weights
        else:
            self.weights = self.DEFAULT_WEIGHTS.copy()

    def evaluate(
        self,
        contexts: list[RetrievedNode],
        answer: str,
        query: str,
    ) -> dict[str, Any]:
        context_relevance = self._calculate_context_relevance(contexts, query)
        answer_completeness = self._calculate_answer_completeness(answer, query)
        consistency = self._calculate_consistency(contexts)
        source_reliability = self._calculate_source_reliability(contexts)

        confidence = self._calculate_weighted_confidence(
            context_relevance,
            answer_completeness,
            consistency,
            source_reliability,
        )

        return {
            "confidence": confidence,
            "context_relevance": context_relevance,
            "answer_completeness": answer_completeness,
            "consistency": consistency,
            "source_reliability": source_reliability,
            "level": self._get_confidence_level(confidence),
        }

    def _calculate_context_relevance(self, contexts: list[RetrievedNode], _query: str) -> float:
        if not contexts:
            return 0.0

        # 直接使用检索得分（向量+BM25混合得分已综合语义相关度）
        # 不再使用简单词重叠计算（词重叠率低会错误拉低语义相关的高分文档）
        scores = [ctx.score for ctx in contexts]

        return round(sum(scores) / len(scores), 2) if scores else 0.0

    def _calculate_answer_completeness(self, answer: str, query: str) -> float:
        if not answer or not query:
            return 0.0

        try:
            import jieba
            query_terms = set(jieba.cut(query.lower()))
            answer_terms = set(jieba.cut(answer.lower()))
        except ImportError:
            # Fallback if jieba not available
            query_terms = set(query.lower().split())
            answer_terms = set(answer.lower().split())

        # Remove stop words
        stop_words = {"的", "了", "是", "在", "和", "与", "或", "有", "我", "你", "他", "她", "它", "什么", "如何", "怎么", "哪些", "哪个", "请", "问", "的", "了"}
        query_terms = query_terms - stop_words

        if not query_terms:
            return 0.5

        covered_terms = sum(1 for term in query_terms if term in answer_terms and len(term) > 1)

        score = covered_terms / len(query_terms) if query_terms else 0

        # Check for substantive answer (not just short acknowledgment)
        if len(answer) < 50:
            score *= 0.7
        elif len(answer) > 200:
            score = min(score + 0.1, 1.0)

        # Check if answer has medical content structure
        medical_content_indicators = ["药物", "治疗", "剂量", "用法", "适应", "禁忌", "血压", "血糖", "糖尿病", "高血压"]
        has_medical_structure = any(ind in answer for ind in medical_content_indicators)
        if has_medical_structure:
            score = min(score + 0.1, 1.0)

        return round(min(score, 1.0), 2)

    def _calculate_consistency(self, contexts: list[RetrievedNode]) -> float:
        if len(contexts) < 2:
            return 0.8 if contexts else 0.0

        scores = [ctx.score for ctx in contexts]
        avg = sum(scores) / len(scores)

        variance = sum((s - avg) ** 2 for s in scores) / len(scores)

        std_dev = variance**0.5

        consistency = 1 - min(std_dev, 1.0)

        return round(consistency, 2)

    def _calculate_source_reliability(self, contexts: list[RetrievedNode]) -> float:
        if not contexts:
            return 0.0

        reliability_scores = []
        for ctx in contexts:
            source = ctx.metadata.get("source_file", "")
            if any(kw in source.lower() for kw in ["指南", "标准", "共识", "教材"]):
                reliability_scores.append(0.9)
            elif any(kw in source.lower() for kw in ["指南", "专家"]):
                reliability_scores.append(0.8)
            else:
                reliability_scores.append(0.6)

        return round(sum(reliability_scores) / len(reliability_scores), 2)

    def _calculate_weighted_confidence(
        self,
        context_relevance: float,
        answer_completeness: float,
        consistency: float,
        source_reliability: float,
    ) -> float:
        confidence = (
            self.weights["context_relevance"] * context_relevance
            + self.weights["answer_completeness"] * answer_completeness
            + self.weights["consistency"] * consistency
            + self.weights["source_reliability"] * source_reliability
        )

        return round(min(max(confidence, 0.0), 1.0), 2)

    def _get_confidence_level(self, confidence: float) -> str:
        if confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "unreliable"

    def get_display_info(self, level: str) -> dict[str, Any]:
        display_info = {
            "high": {"color": "#28a745", "label": "高置信度", "action": "可直接使用"},
            "medium": {"color": "#f0ad4e", "label": "中等置信度", "action": "建议核实"},
            "low": {"color": "#fd7e14", "label": "低置信度", "action": "需要补充信息"},
            "unreliable": {"color": "#dc3545", "label": "不可靠", "action": "不建议使用"},
        }

        return display_info.get(level, display_info["unreliable"])
