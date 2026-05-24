"""Protocol 接口定义 - 评估器抽象层"""

from abc import abstractmethod
from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.schemas import QueryResponse
    from rag.evaluation.evaluator import RAGEvaluationResult
    from rag.evaluation.retrieval_eval import RetrievalMetrics
    from rag.evaluation.generation_eval import GenerationMetrics
    from rag.evaluation.medical_safety_eval import MedicalSafetyMetrics


@runtime_checkable
class RetrievalEvaluatorProtocol(Protocol):
    """检索评估器接口"""

    @abstractmethod
    def evaluate(
        self,
        retrieved_ids: list[str],
        ground_truth_ids: list[str],
    ) -> "RetrievalMetrics": ...

    @abstractmethod
    def evaluate_without_ground_truth(
        self,
        retrieved_ids: list[str],
        min_relevant: int = 1,
    ) -> dict: ...


@runtime_checkable
class GenerationEvaluatorProtocol(Protocol):
    """生成评估器接口"""

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        citations: list,
    ) -> "GenerationMetrics": ...


@runtime_checkable
class MedicalSafetyEvaluatorProtocol(Protocol):
    """医疗安全评估器接口"""

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        warnings: list,
    ) -> "MedicalSafetyMetrics": ...


@runtime_checkable
class ReporterPlugin(Protocol):
    """报告生成器接口"""

    @abstractmethod
    def generate(
        self,
        results: list["RAGEvaluationResult"],
    ) -> str: ...

    @abstractmethod
    def supports_format(self, fmt: str) -> bool: ...