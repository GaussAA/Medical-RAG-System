"""Medical RAG Evaluation Module.

Provides comprehensive evaluation for RAG systems covering:
- Retrieval: Precision@K, Recall@K, NDCG@K, MRR
- Generation: Faithfulness, Answer Relevancy (via LLM judge)
- Medical Safety: Entity accuracy, warning coverage, contradiction detection
"""

from src.evaluation.benchmark_runner import BenchmarkRunner
from src.evaluation.evaluator import EvalGroundTruth, RAGEvaluationResult, RAGEvaluator
from src.evaluation.generation_eval import GenerationEvaluator
from src.evaluation.interfaces import (
    GenerationEvaluatorProtocol,
    MedicalSafetyEvaluatorProtocol,
    ReporterPlugin,
    RetrievalEvaluatorProtocol,
)
from src.evaluation.medical_safety_eval import MedicalSafetyEvaluator
from src.evaluation.reporter import EvaluationReporter
from src.evaluation.retrieval_eval import RetrievalEvaluator

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
