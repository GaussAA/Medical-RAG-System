"""Medical RAG Evaluation Module.

Provides comprehensive evaluation for RAG systems covering:
- Retrieval: Precision@K, Recall@K, NDCG@K, MRR
- Generation: Faithfulness, Answer Relevancy (via LLM judge)
- Medical Safety: Entity accuracy, warning coverage, contradiction detection
"""

from rag.evaluation.benchmark_runner import BenchmarkRunner
from rag.evaluation.evaluator import EvalGroundTruth, RAGEvaluationResult, RAGEvaluator
from rag.evaluation.generation_eval import GenerationEvaluator
from rag.evaluation.interfaces import (
    GenerationEvaluatorProtocol,
    MedicalSafetyEvaluatorProtocol,
    ReporterPlugin,
    RetrievalEvaluatorProtocol,
)
from rag.evaluation.medical_safety_eval import MedicalSafetyEvaluator
from rag.evaluation.reporter import EvaluationReporter
from rag.evaluation.retrieval_eval import RetrievalEvaluator

__all__ = [
    "RAGEvaluator",
    "RAGEvaluationResult",
    "EvalGroundTruth",
    "RetrievalEvaluator",
    "GenerationEvaluator",
    "MedicalSafetyEvaluator",
    "BenchmarkRunner",
    "EvaluationReporter",
    "RetrievalEvaluatorProtocol",
    "GenerationEvaluatorProtocol",
    "MedicalSafetyEvaluatorProtocol",
    "ReporterPlugin",
]
