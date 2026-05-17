# tests/unit/test_warnings_generator.py
import pytest
from rag.generation.warnings_generator import WarningsGenerator
from app.models.schemas import RetrievedNode, RiskWarning

def test_general_warning_always_added():
    gen = WarningsGenerator()
    warnings = gen.generate("some answer")
    assert any(w.type == "general" for w in warnings)

def test_medication_warning_when_drug_keywords():
    gen = WarningsGenerator()
    warnings = gen.generate("推荐使用降压药每日一次")
    assert any(w.type == "medication" for w in warnings)

def test_emergency_warning_when_emergency_keywords():
    gen = WarningsGenerator()
    warnings = gen.generate("如有紧急症状请立即就医")
    assert any(w.type == "emergency" for w in warnings)

def test_hallucination_warning_on_unverified_citations():
    gen = WarningsGenerator()
    mock_citations = [
        type('obj', (object,), {'verified': False})(),
        type('obj', (object,), {'verified': False})(),
    ]
    warnings = gen.generate("answer", citations=mock_citations)
    assert any(w.type == "hallucination" for w in warnings)