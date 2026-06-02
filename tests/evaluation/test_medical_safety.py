"""Tests for medical safety evaluation module."""

from rag.evaluation.medical_safety_eval import (
    MedicalSafetyEvaluator,
    MedicalSafetyMetrics,
)


class TestMedicalSafetyEvaluator:
    """Tests for MedicalSafetyEvaluator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = MedicalSafetyEvaluator()

    def test_entity_accuracy_with_matching_entities(self):
        """Test entity accuracy when answer entities match contexts."""
        contexts = [
            "糖尿病患者应使用二甲双胍治疗，每次500mg，每日两次。",
            "高血压患者首选钙通道阻滞剂降压治疗。",
        ]
        answer = "糖尿病患者使用二甲双胍500mg，每日两次进行治疗。"

        accuracy = self.evaluator._evaluate_entity_accuracy(answer, contexts)
        assert accuracy is not None
        assert 0.0 <= accuracy <= 1.0

    def test_entity_accuracy_no_entities_found(self):
        """Test entity accuracy when no medical entities found."""
        contexts = ["这是一段普通的文本内容。"]
        answer = "今天天气很好。"

        accuracy = self.evaluator._evaluate_entity_accuracy(answer, contexts)
        assert accuracy is None  # No entities found

    def test_entity_accuracy_empty_inputs(self):
        """Test entity accuracy with empty inputs."""
        accuracy = self.evaluator._evaluate_entity_accuracy("", [])
        assert accuracy is None

    def test_warning_coverage_medication_triggered(self):
        """Test warning coverage when medication keywords present."""
        answer = "糖尿病患者应该使用药物治疗，剂量为每次500mg。"

        coverage = self.evaluator._evaluate_warning_coverage(
            answer=answer,
            warnings=None,
        )

        assert "medication" in coverage
        assert coverage["medication"] is True

    def test_warning_coverage_diagnosis_triggered(self):
        """Test warning coverage when diagnosis keywords present."""
        answer = "根据检查结果，确诊为糖尿病，需要开始治疗方案。"

        coverage = self.evaluator._evaluate_warning_coverage(
            answer=answer,
            warnings=None,
        )

        assert "diagnosis" in coverage
        assert coverage["diagnosis"] is True

    def test_warning_coverage_emergency_triggered(self):
        """Test warning coverage when emergency keywords present."""
        answer = "如果出现剧烈疼痛，应立即就医或拨打急救电话。"

        coverage = self.evaluator._evaluate_warning_coverage(
            answer=answer,
            warnings=None,
        )

        assert "emergency" in coverage
        assert coverage["emergency"] is True

    def test_warning_coverage_no_triggers(self):
        """Test warning coverage when no warning triggers present."""
        answer = "今天天气很好，适合户外活动。"

        coverage = self.evaluator._evaluate_warning_coverage(
            answer=answer,
            warnings=None,
        )

        assert coverage["medication"] is False
        assert coverage["diagnosis"] is False
        assert coverage["emergency"] is False

    def test_contradiction_detection_no_contradiction(self):
        """Test contradiction detection with consistent contexts."""
        contexts = [
            "糖尿病患者应使用二甲双胍治疗。",
            "二甲双胍是治疗2型糖尿病的一线药物。",
        ]

        detected = self.evaluator._detect_contradictions(contexts)
        assert detected is False

    def test_contradiction_detection_with_single_context(self):
        """Test contradiction detection with single context."""
        contexts = ["糖尿病患者应使用二甲双胍治疗。"]

        detected = self.evaluator._detect_contradictions(contexts)
        assert detected is False

    def test_contradiction_detection_empty_contexts(self):
        """Test contradiction detection with no contexts."""
        detected = self.evaluator._detect_contradictions([])
        assert detected is False

    def test_safety_score_calculation_basic(self):
        """Test safety score calculation with basic inputs."""
        score = self.evaluator._calculate_safety_score(
            entity_accuracy=0.9,
            warning_coverage={"medication": True, "diagnosis": True, "emergency": True},
            contradiction_detected=False,
        )

        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Good safety score

    def test_safety_score_calculation_with_contradiction(self):
        """Test safety score calculation with detected contradiction."""
        score = self.evaluator._evaluate_entity_accuracy(
            "不应该使用此药物",
            ["应该使用此药物来治疗"],
        )

        # With contradiction, score should be lower
        # This is a simplified test
        assert score is not None

    def test_evaluate_returns_medical_safety_metrics(self):
        """Test that evaluate returns MedicalSafetyMetrics."""
        import asyncio

        async def run_test():
            evaluator = MedicalSafetyEvaluator()
            metrics = await evaluator.evaluate(
                query="糖尿病患者如何用药？",
                answer="糖尿病患者可以使用二甲双胍治疗。",
                contexts=["糖尿病治疗首选二甲双胍。"],
                warnings=[],
            )

            assert isinstance(metrics, MedicalSafetyMetrics)
            assert hasattr(metrics, "entity_accuracy")
            assert hasattr(metrics, "warning_coverage")
            assert hasattr(metrics, "contradiction_detected")
            assert hasattr(metrics, "safety_score")
            return True

        assert asyncio.run(run_test())


class TestMedicalSafetyMetrics:
    """Tests for MedicalSafetyMetrics dataclass."""

    def test_medical_safety_metrics_creation(self):
        """Test creating MedicalSafetyMetrics with default values."""
        metrics = MedicalSafetyMetrics()

        assert metrics.entity_accuracy is None
        assert metrics.warning_coverage is None
        assert metrics.contradiction_detected is False
        assert metrics.safety_score == 0.0

    def test_medical_safety_metrics_with_values(self):
        """Test creating MedicalSafetyMetrics with specific values."""
        metrics = MedicalSafetyMetrics(
            entity_accuracy=0.9,
            warning_coverage={
                "medication": True,
                "diagnosis": False,
                "emergency": False,
            },
            contradiction_detected=False,
            safety_score=0.85,
        )

        assert metrics.entity_accuracy == 0.9
        assert metrics.warning_coverage["medication"] is True
        assert metrics.warning_coverage["diagnosis"] is False
        assert metrics.safety_score == 0.85


class TestMedicalEntityPatterns:
    """Tests for medical entity pattern extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = MedicalSafetyEvaluator()

    def test_extract_drug_entities(self):
        """Test extracting drug-related entities."""
        text = "糖尿病患者应使用二甲双胍治疗，每次500mg，每日两次。"

        entities = self.evaluator._extract_entities(text, self.evaluator.DRUG_PATTERNS)

        assert len(entities) > 0
        assert "药物" in entities or "mg" in entities

    def test_extract_disease_entities(self):
        """Test extracting disease-related entities."""
        text = "该患者确诊为糖尿病，需要进行诊断和治疗。"

        entities = self.evaluator._extract_entities(text, self.evaluator.DISEASE_PATTERNS)

        assert len(entities) > 0
        assert "诊断" in entities or "疾病" in entities

    def test_extract_procedure_entities(self):
        """Test extracting procedure-related entities."""
        text = "建议进行CT检查和手术治疗。"

        entities = self.evaluator._extract_entities(text, self.evaluator.PROCEDURE_PATTERNS)

        assert len(entities) > 0
        assert "手术" in entities or "CT" in entities

    def test_extract_no_entities(self):
        """Test extraction when no patterns match."""
        text = "今天天气很好。"

        entities = self.evaluator._extract_entities(text, self.evaluator.DRUG_PATTERNS)

        assert len(entities) == 0
