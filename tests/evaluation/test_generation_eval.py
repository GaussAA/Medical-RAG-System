"""Tests for generation evaluation module."""

import pytest

from rag.evaluation.generation_eval import GenerationEvaluator, GenerationMetrics


class TestGenerationEvaluator:
    """Tests for GenerationEvaluator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = GenerationEvaluator(llm_generator=None)  # Use rule-based

    def test_faithfulness_rule_with_matching_terms(self):
        """Test rule-based faithfulness with matching medical terms."""
        contexts = [
            "糖尿病患者应首选二甲双胍治疗，剂量为每次500mg，每日两次。",
            "血糖监测对糖尿病患者非常重要，建议每周监测3-4次。",
        ]
        answer = "糖尿病患者首选二甲双胍，剂量500mg每日两次，需定期监测血糖。"

        # This should return a moderate to high score since terms match
        score = self.evaluator._evaluate_faithfulness_rule(answer, contexts)
        assert 0.3 <= score <= 0.9

    def test_faithfulness_rule_with_no_matching_terms(self):
        """Test rule-based faithfulness with no matching terms."""
        contexts = ["这是一段完全不相关的内容"]
        answer = "糖尿病患者应该使用某种药物治疗。"

        # Low score when no terms match
        score = self.evaluator._evaluate_faithfulness_rule(answer, contexts)
        assert score == 0.3  # Low baseline

    def test_faithfulness_rule_empty_contexts(self):
        """Test faithfulness with empty contexts."""
        score = self.evaluator._evaluate_faithfulness_rule("Some answer", [])
        assert score == 0.0

    def test_faithfulness_rule_empty_answer(self):
        """Test faithfulness with empty answer."""
        score = self.evaluator._evaluate_faithfulness_rule("", ["Some context"])
        assert score == 0.0

    def test_relevancy_rule_query_term_coverage(self):
        """Test rule-based relevancy with query term coverage."""
        query = "糖尿病患者如何选择降糖药物？"
        answer = "糖尿病患者可以选择二甲双胍作为首选降糖药物。"

        score = self.evaluator._evaluate_relevancy_rule(query, answer)
        assert 0.3 <= score <= 1.0

    def test_relevancy_rule_no_coverage(self):
        """Test relevancy when answer doesn't cover query terms."""
        query = "糖尿病患者如何选择降糖药物？"
        answer = "今天天气很好，适合出去散步。"

        score = self.evaluator._evaluate_relevancy_rule(query, answer)
        assert score < 0.5  # Low coverage

    def test_relevancy_rule_empty_inputs(self):
        """Test relevancy with empty inputs."""
        score = self.evaluator._evaluate_relevancy_rule("", "Some answer")
        assert score == 0.0

        score = self.evaluator._evaluate_relevancy_rule("Some query", "")
        assert score == 0.0

    def test_context_precision(self):
        """Test context precision evaluation."""
        query = "糖尿病 药物 治疗"
        contexts = [
            "糖尿病患者需要药物治疗。",
            "这是一段完全不相关的内容。",
            "糖尿病治疗常用药物包括二甲双胍。",
        ]

        precision = self.evaluator._evaluate_context_precision(contexts, query)
        assert 0.0 <= precision <= 1.0

    def test_context_precision_empty_contexts(self):
        """Test context precision with no contexts."""
        precision = self.evaluator._evaluate_context_precision([], "some query")
        assert precision == 0.0

    def test_citation_accuracy_all_verified(self):
        """Test citation accuracy when all citations are verified."""
        # Mock citations with verified=True
        class MockCitation:
            def __init__(self, verified):
                self.verified = verified

        citations = [
            MockCitation(verified=True),
            MockCitation(verified=True),
            MockCitation(verified=True),
        ]

        accuracy = self.evaluator._calculate_citation_accuracy(citations)
        assert accuracy == 1.0

    def test_citation_accuracy_none_verified(self):
        """Test citation accuracy when no citations are verified."""
        class MockCitation:
            def __init__(self, verified):
                self.verified = verified

        citations = [
            MockCitation(verified=False),
            MockCitation(verified=False),
        ]

        accuracy = self.evaluator._calculate_citation_accuracy(citations)
        assert accuracy == 0.0

    def test_citation_accuracy_partial_verified(self):
        """Test citation accuracy with partial verification."""
        class MockCitation:
            def __init__(self, verified):
                self.verified = verified

        citations = [
            MockCitation(verified=True),
            MockCitation(verified=False),
            MockCitation(verified=True),
        ]

        accuracy = self.evaluator._calculate_citation_accuracy(citations)
        assert accuracy == pytest.approx(2 / 3, rel=0.01)

    def test_citation_accuracy_empty_citations(self):
        """Test citation accuracy with no citations."""
        accuracy = self.evaluator._calculate_citation_accuracy([])
        assert accuracy == 0.0

    def test_hallucination_ratio_all_unverified(self):
        """Test hallucination ratio when all citations are unverified."""
        class MockCitation:
            def __init__(self, verified):
                self.verified = verified

        citations = [
            MockCitation(verified=False),
            MockCitation(verified=False),
        ]

        ratio = self.evaluator._calculate_hallucination_ratio(citations)
        assert ratio == 1.0

    def test_hallucination_ratio_none_unverified(self):
        """Test hallucination ratio when all citations are verified."""
        class MockCitation:
            def __init__(self, verified):
                self.verified = verified

        citations = [
            MockCitation(verified=True),
            MockCitation(verified=True),
        ]

        ratio = self.evaluator._calculate_hallucination_ratio(citations)
        assert ratio == 0.0

    def test_evaluate_returns_generation_metrics(self):
        """Test that evaluate returns GenerationMetrics."""
        import asyncio

        async def run_test():
            evaluator = GenerationEvaluator(llm_generator=None)
            metrics = await evaluator.evaluate(
                query="糖尿病患者如何用药？",
                answer="糖尿病患者可以使用二甲双胍治疗。",
                contexts=["糖尿病治疗首选二甲双胍。"],
                citations=[],
            )

            assert isinstance(metrics, GenerationMetrics)
            assert hasattr(metrics, "faithfulness")
            assert hasattr(metrics, "answer_relevancy")
            assert hasattr(metrics, "citation_accuracy")
            assert hasattr(metrics, "hallucination_ratio")
            return True

        assert asyncio.run(run_test())


class TestGenerationMetrics:
    """Tests for GenerationMetrics dataclass."""

    def test_generation_metrics_creation(self):
        """Test creating GenerationMetrics with default values."""
        metrics = GenerationMetrics()

        assert metrics.faithfulness == 0.0
        assert metrics.answer_relevancy == 0.0
        assert metrics.context_precision == 0.0
        assert metrics.citation_accuracy == 0.0
        assert metrics.hallucination_ratio == 0.0

    def test_generation_metrics_with_values(self):
        """Test creating GenerationMetrics with specific values."""
        metrics = GenerationMetrics(
            faithfulness=0.85,
            answer_relevancy=0.9,
            context_precision=0.75,
            citation_accuracy=0.95,
            hallucination_ratio=0.05,
        )

        assert metrics.faithfulness == 0.85
        assert metrics.answer_relevancy == 0.9
        assert metrics.citation_accuracy == 0.95
        assert metrics.hallucination_ratio == 0.05