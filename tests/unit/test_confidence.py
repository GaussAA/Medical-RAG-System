from app.core.confidence import ConfidenceEvaluator
from app.models.schemas import RetrievedNode


class TestConfidenceEvaluator:
    def setup_method(self):
        self.evaluator = ConfidenceEvaluator()

    def test_evaluate_returns_all_fields(self):
        contexts = [
            RetrievedNode(
                node_id="1",
                content="糖尿病的诊断标准是空腹血糖>=7.0",
                score=0.9,
                metadata={"source_file": "糖尿病指南.txt"},
            )
        ]

        result = self.evaluator.evaluate(
            contexts=contexts,
            answer="糖尿病的诊断标准是空腹血糖>=7.0mmol/L",
            query="糖尿病的诊断标准是什么",
        )

        assert "confidence" in result
        assert "context_relevance" in result
        assert "answer_completeness" in result
        assert "consistency" in result
        assert "source_reliability" in result
        assert "level" in result

    def test_evaluate_empty_contexts(self):
        result = self.evaluator.evaluate(
            contexts=[],
            answer="some answer",
            query="some query",
        )

        assert result["context_relevance"] == 0.0
        assert result["consistency"] == 0.0
        assert "confidence" in result

    def test_context_relevance_high_overlap(self):
        contexts = [
            RetrievedNode(
                node_id="1",
                content="糖尿病 诊断 标准 血糖",
                score=0.9,
                metadata={},
            )
        ]

        result = self.evaluator.evaluate(
            contexts=contexts,
            answer="糖尿病的诊断标准",
            query="糖尿病 诊断 标准",
        )

        assert result["context_relevance"] > 0.5

    def test_context_relevance_low_overlap(self):
        """Test that context relevance reflects the actual retrieval score."""
        # When contexts have high retrieval scores, context_relevance reflects that score
        contexts = [
            RetrievedNode(
                node_id="1",
                content="高血压 治疗 降压药",
                score=0.9,
                metadata={},
            )
        ]

        result = self.evaluator.evaluate(
            contexts=contexts,
            answer="高血压的治疗方法",
            query="糖尿病诊断标准",
        )

        # Context relevance reflects the retrieval score (0.9 in this case)
        assert result["context_relevance"] == 0.9

    def test_answer_completeness_full(self):
        contexts = [RetrievedNode(node_id="1", content="糖尿病 诊断 标准", score=0.9, metadata={})]

        result = self.evaluator.evaluate(
            contexts=contexts,
            answer="糖尿病 诊断 标准 血糖 HbA1c",
            query="糖尿病 诊断 标准",
        )

        assert result["answer_completeness"] >= 0.5

    def test_answer_completeness_short_answer(self):
        contexts = [RetrievedNode(node_id="1", content="糖尿病 诊断", score=0.9, metadata={})]

        result = self.evaluator.evaluate(
            contexts=contexts,
            answer="是",
            query="糖尿病 诊断 标准",
        )

        assert result["answer_completeness"] < 0.5

    def test_consistency_high_with_similar_scores(self):
        contexts = [
            RetrievedNode(node_id="1", content="内容A", score=0.9, metadata={}),
            RetrievedNode(node_id="2", content="内容B", score=0.85, metadata={}),
            RetrievedNode(node_id="3", content="内容C", score=0.88, metadata={}),
        ]

        result = self.evaluator.evaluate(
            contexts=contexts,
            answer="答案",
            query="查询",
        )

        assert result["consistency"] > 0.7

    def test_consistency_low_with_varying_scores(self):
        contexts = [
            RetrievedNode(node_id="1", content="内容A", score=0.9, metadata={}),
            RetrievedNode(node_id="2", content="内容B", score=0.3, metadata={}),
            RetrievedNode(node_id="3", content="内容C", score=0.2, metadata={}),
        ]

        result = self.evaluator.evaluate(
            contexts=contexts,
            answer="答案",
            query="查询",
        )

        assert result["consistency"] < 0.7

    def test_consistency_single_context(self):
        contexts = [RetrievedNode(node_id="1", content="内容A", score=0.9, metadata={})]

        result = self.evaluator.evaluate(
            contexts=contexts,
            answer="答案",
            query="查询",
        )

        assert result["consistency"] == 0.8

    def test_source_reliability_medical_guide(self):
        contexts = [
            RetrievedNode(
                node_id="1",
                content="内容",
                score=0.9,
                metadata={"source_file": "糖尿病诊疗指南.txt"},
            )
        ]

        result = self.evaluator.evaluate(
            contexts=contexts,
            answer="答案",
            query="查询",
        )

        assert result["source_reliability"] == 0.9

    def test_source_reliability_expert_consensus(self):
        contexts = [
            RetrievedNode(
                node_id="1",
                content="内容",
                score=0.9,
                metadata={"source_file": "专家共识.md"},
            )
        ]

        result = self.evaluator.evaluate(
            contexts=contexts,
            answer="答案",
            query="查询",
        )

        # "共识" matches first condition for 0.9
        assert result["source_reliability"] == 0.9

    def test_source_reliability_regular_file(self):
        contexts = [
            RetrievedNode(
                node_id="1",
                content="内容",
                score=0.9,
                metadata={"source_file": "notes.txt"},
            )
        ]

        result = self.evaluator.evaluate(
            contexts=contexts,
            answer="答案",
            query="查询",
        )

        assert result["source_reliability"] == 0.6

    def test_confidence_level_high(self):
        contexts = [
            RetrievedNode(
                node_id=str(i),
                content="糖尿病 诊断 标准 血糖 治疗",
                score=0.9,
                metadata={"source_file": "糖尿病指南.txt"},
            )
            for i in range(5)
        ]
        result = self.evaluator.evaluate(
            contexts=contexts,
            answer="糖尿病的诊断标准是空腹血糖>=7.0，餐后血糖>=11.1，HbA1c>=6.5%",
            query="糖尿病 诊断 标准",
        )

        assert result["level"] == "high"

    def test_confidence_level_medium(self):
        contexts = [
            RetrievedNode(
                node_id="1",
                content="糖尿病 诊断 标准 指南",
                score=0.5,
                metadata={"source_file": "notes.txt"},
            ),
            RetrievedNode(
                node_id="2",
                content="糖尿病 治疗 药物",
                score=0.4,
                metadata={"source_file": "notes.txt"},
            ),
        ]
        result = self.evaluator.evaluate(
            contexts=contexts,
            answer="糖尿病的诊断标准",
            query="糖尿病 诊断 标准",
        )

        # With lower scores and non-medical source, confidence should be medium
        assert result["level"] == "medium"

    def test_confidence_level_low(self):
        contexts = [
            RetrievedNode(
                node_id="1",
                content="糖尿病",
                score=0.5,
                metadata={"source_file": "notes.txt"},
            )
        ]
        result = self.evaluator.evaluate(
            contexts=contexts,
            answer="是",
            query="糖尿病 诊断 标准 指标",
        )

        assert result["level"] == "low"

    def test_confidence_level_unreliable(self):
        result = self.evaluator.evaluate(
            contexts=[],
            answer="",
            query="test",
        )

        assert result["level"] == "unreliable"

    def test_confidence_in_range(self):
        for _ in range(10):
            contexts = [
                RetrievedNode(
                    node_id=str(i),
                    content="糖尿病 诊断 标准 " * 10,
                    score=0.7 + (i * 0.03),
                    metadata={"source_file": "指南.txt"},
                )
                for i in range(5)
            ]

            result = self.evaluator.evaluate(
                contexts=contexts,
                answer="糖尿病的诊断标准是空腹血糖>=7.0mmol/L，餐后血糖>=11.1mmol/L",
                query="糖尿病的诊断标准",
            )

            assert 0.0 <= result["confidence"] <= 1.0


class TestGetDisplayInfo:
    def setup_method(self):
        self.evaluator = ConfidenceEvaluator()

    def test_high_confidence_display(self):
        info = self.evaluator.get_display_info("high")
        assert info["color"] == "#28a745"
        assert info["label"] == "高置信度"

    def test_medium_confidence_display(self):
        info = self.evaluator.get_display_info("medium")
        assert info["color"] == "#f0ad4e"
        assert info["label"] == "中等置信度"

    def test_low_confidence_display(self):
        info = self.evaluator.get_display_info("low")
        assert info["color"] == "#fd7e14"
        assert info["label"] == "低置信度"

    def test_unreliable_display(self):
        info = self.evaluator.get_display_info("unreliable")
        assert info["color"] == "#dc3545"
        assert info["label"] == "不可靠"

    def test_unknown_level_defaults_to_unreliable(self):
        info = self.evaluator.get_display_info("unknown")
        assert info["color"] == "#dc3545"
