"""RAGEvaluator Integration Tests.

Tests for the unified RAG evaluation interface combining retrieval,
generation, and medical safety metrics.
"""

import pytest

from app.models.schemas import Citation, CitationPosition, QueryResponse, RiskWarning
from rag.evaluation.evaluator import EvalGroundTruth, RAGEvaluationResult, RAGEvaluator


class TestRAGEvaluatorIntegration:
    """RAGEvaluator integration tests."""

    @pytest.mark.asyncio
    async def test_evaluate_with_full_response(self):
        """Test complete response evaluation."""
        evaluator = RAGEvaluator()

        response = QueryResponse(
            answer="二甲双胍是糖尿病首选药物",
            confidence=0.85,
            citations=[
                Citation(
                    source_id="1",
                    document_id="doc1",
                    file_name="diabetes_guide.md",
                    page_number=1,
                    chunk_content="二甲双胍是2型糖尿病首选药物",
                    relevance_score=0.9,
                    position=CitationPosition.DIRECT,
                    verified=True,
                )
            ],
            warnings=[
                RiskWarning(
                    type="medication",
                    message="请遵医嘱用药",
                    priority="medium",
                )
            ],
            session_id="test_session",
            processing_time=1.5,
            metadata={"retrieved_chunks": 5},
        )

        result = await evaluator.evaluate(
            query="糖尿病患者如何选择降糖药物？",
            response=response,
        )

        assert result.overall_score >= 0
        assert result.faithfulness >= 0
        assert result.safety_score >= 0
        assert result.answer_relevancy >= 0
        assert result.citation_accuracy >= 0

    @pytest.mark.asyncio
    async def test_evaluate_without_ground_truth(self):
        """Test evaluation without ground truth uses hit rate only."""
        evaluator = RAGEvaluator()

        response = QueryResponse(
            answer="糖尿病需要注意饮食控制",
            confidence=0.75,
            citations=[
                Citation(
                    source_id="1",
                    document_id="doc2",
                    file_name="diabetes_tips.md",
                    page_number=3,
                    chunk_content="糖尿病患者应注意饮食控制",
                    relevance_score=0.85,
                    position=CitationPosition.DIRECT,
                    verified=True,
                )
            ],
            warnings=[],
            session_id="test_session_2",
            processing_time=1.0,
            metadata={"retrieved_chunks": 3},
        )

        result = await evaluator.evaluate(
            query="糖尿病患者日常需要注意什么？",
            response=response,
        )

        # Without ground truth, hit rate should be computed
        assert result.retrieval_hit_rate >= 0
        assert result.overall_score >= 0

    @pytest.mark.asyncio
    async def test_evaluate_with_ground_truth(self):
        """Test evaluation with ground truth includes precision/recall."""
        evaluator = RAGEvaluator()

        ground_truth = EvalGroundTruth(
            query_id="query_1",
            relevant_doc_ids=["doc1", "doc2", "doc3"],
            expected_keywords=["糖尿病", "药物"],
            reference_answer="糖尿病常用药物包括二甲双胍",
            difficulty="medium",
            safety_sensitive=True,
        )

        response = QueryResponse(
            answer="二甲双胍是常用的糖尿病药物",
            confidence=0.8,
            citations=[
                Citation(
                    source_id="1",
                    document_id="doc1",
                    file_name="guide1.md",
                    page_number=1,
                    chunk_content="二甲双胍是2型糖尿病首选药物",
                    relevance_score=0.9,
                    position=CitationPosition.DIRECT,
                    verified=True,
                )
            ],
            warnings=[],
            session_id="test_session_3",
            processing_time=0.8,
            metadata={"retrieved_chunks": 5},
        )

        result = await evaluator.evaluate(
            query="糖尿病常用什么药物？",
            response=response,
            ground_truth=ground_truth,
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
        )

        # With ground truth, retrieval metrics should be populated
        assert result.mrr > 0
        assert result.precision_at_k is not None
        assert result.recall_at_k is not None
        assert result.ndcg_at_k is not None

    @pytest.mark.asyncio
    async def test_evaluate_batch(self):
        """Test batch evaluation of multiple responses."""
        evaluator = RAGEvaluator()

        responses = [
            QueryResponse(
                answer="二甲双胍用于治疗2型糖尿病",
                confidence=0.85,
                citations=[
                    Citation(
                        source_id="1",
                        document_id="doc1",
                        file_name="dm_guide.md",
                        page_number=1,
                        chunk_content="二甲双胍是2型糖尿病首选药物",
                        relevance_score=0.9,
                        position=CitationPosition.DIRECT,
                        verified=True,
                    )
                ],
                warnings=[],
                session_id="batch_session_1",
                processing_time=1.0,
                metadata={"retrieved_chunks": 3},
            ),
            QueryResponse(
                answer="高血压患者应定期监测血压",
                confidence=0.78,
                citations=[
                    Citation(
                        source_id="1",
                        document_id="doc2",
                        file_name="bp_guide.md",
                        page_number=2,
                        chunk_content="高血压患者需要定期监测血压",
                        relevance_score=0.88,
                        position=CitationPosition.DIRECT,
                        verified=True,
                    )
                ],
                warnings=[],
                session_id="batch_session_2",
                processing_time=1.2,
                metadata={"retrieved_chunks": 4},
            ),
        ]

        queries = [
            "二甲双胍用于什么病？",
            "高血压患者注意什么？",
        ]

        results = await evaluator.evaluate_batch(
            queries=queries,
            responses=responses,
        )

        assert len(results) == 2
        assert all(isinstance(r, RAGEvaluationResult) for r in results)
        assert all(r.overall_score >= 0 for r in results)

    @pytest.mark.asyncio
    async def test_evaluate_with_medication_warning(self):
        """Test that medication warnings are captured in safety evaluation."""
        evaluator = RAGEvaluator()

        response = QueryResponse(
            answer="阿司匹林可以预防心脑血管疾病",
            confidence=0.82,
            citations=[
                Citation(
                    source_id="1",
                    document_id="doc3",
                    file_name="cardio_guide.md",
                    page_number=5,
                    chunk_content="阿司匹林用于预防心脑血管疾病",
                    relevance_score=0.87,
                    position=CitationPosition.DIRECT,
                    verified=True,
                )
            ],
            warnings=[
                RiskWarning(
                    type="medication",
                    message="请遵医嘱使用阿司匹林",
                    priority="medium",
                )
            ],
            session_id="med_session",
            processing_time=1.1,
            metadata={"retrieved_chunks": 2},
        )

        result = await evaluator.evaluate(
            query="阿司匹林有什么作用？",
            response=response,
        )

        assert result.safety_score >= 0
        assert result.warning_coverage is not None

    def test_evaluation_result_to_dict(self):
        """Test RAGEvaluationResult serialization."""
        result = RAGEvaluationResult(
            query_id="test_query",
            faithfulness=0.85,
            answer_relevancy=0.78,
            safety_score=0.9,
            overall_score=0.84,
        )

        result_dict = result.to_dict()

        assert result_dict["query_id"] == "test_query"
        assert result_dict["generation"]["faithfulness"] == 0.85
        assert result_dict["medical_safety"]["safety_score"] == 0.9
        assert result_dict["overall_score"] == 0.84
        assert "timestamp" in result_dict

    @pytest.mark.asyncio
    async def test_evaluate_empty_citations(self):
        """Test evaluation with empty citations list."""
        evaluator = RAGEvaluator()

        response = QueryResponse(
            answer="这个问题我不确定答案",
            confidence=0.3,
            citations=[],
            warnings=[
                RiskWarning(
                    type="general",
                    message="信息不足以回答此问题",
                    priority="low",
                )
            ],
            session_id="uncertain_session",
            processing_time=0.5,
            metadata={"retrieved_chunks": 0},
        )

        result = await evaluator.evaluate(
            query="某种罕见病的治疗方法？",
            response=response,
        )

        # Should still produce a result even with no citations
        assert result.overall_score >= 0
        assert result.citation_accuracy == 0  # No citations means 0 accuracy


class TestEvalGroundTruth:
    """Test EvalGroundTruth dataclass."""

    def test_default_values(self):
        """Test default values for ground truth."""
        gt = EvalGroundTruth(query_id="test")

        assert gt.query_id == "test"
        assert gt.relevant_doc_ids == []
        assert gt.expected_keywords == []
        assert gt.reference_answer is None
        assert gt.difficulty == "medium"
        assert gt.safety_sensitive is False

    def test_custom_values(self):
        """Test custom values for ground truth."""
        gt = EvalGroundTruth(
            query_id="custom_test",
            relevant_doc_ids=["doc_a", "doc_b"],
            expected_keywords=["关键词1", "关键词2"],
            reference_answer="标准答案是...",
            difficulty="hard",
            safety_sensitive=True,
        )

        assert gt.query_id == "custom_test"
        assert len(gt.relevant_doc_ids) == 2
        assert gt.difficulty == "hard"
        assert gt.safety_sensitive is True
