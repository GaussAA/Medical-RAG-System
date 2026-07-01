"""Tests for SafetyChecker (new src architecture)."""

from src.common.safety.checker import SafetyChecker


class TestSafetyCheckerDI:
    """Test SafetyChecker initialization."""

    def test_default_initialization(self):
        """Test SafetyChecker initializes with default config from settings."""
        checker = SafetyChecker()
        assert checker.config is not None


class TestSafetyCheckerInputValidation:
    """Test SafetyChecker with input validation scenarios."""

    def test_normal_text_passes(self):
        """Test normal medical question passes safety check."""
        checker = SafetyChecker()
        result = checker.check("请问糖尿病的诊断标准是什么？")
        assert result.passed is True

    def test_phone_number_sanitized(self):
        """Test phone numbers are sanitized."""
        checker = SafetyChecker()
        text = "我的手机号是13812345678"
        result = checker.check(text)
        assert result.passed is True
        assert "13812345678" not in result.sanitized_text

    def test_id_card_sanitized(self):
        """Test ID card numbers are sanitized."""
        checker = SafetyChecker()
        text = "我的身份证是110101199001011234"
        result = checker.check(text)
        assert result.passed is True
        assert "110101199001011234" not in result.sanitized_text
