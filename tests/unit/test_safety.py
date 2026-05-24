from app.core.safety import SafetyChecker, SafetyConfig


class TestSafetyCheckerDI:
    """Test SafetyChecker dependency injection."""

    def test_default_initialization(self):
        """Test SafetyChecker initializes with default config from settings."""
        checker = SafetyChecker()
        assert checker.config is not None
        assert checker.sensitive_patterns is not None

    def test_custom_config_initialization(self):
        """Test SafetyChecker can be initialized with custom config."""
        config = SafetyConfig(
            enable=True,
            sensitive_words_check=True,
            privacy_protection=True,
            political_check=True,
            adult_content_check=True,
        )
        checker = SafetyChecker(config=config)
        assert checker.config == config
        assert checker.config.enable is True


class TestSafetyCheckerPatterns:
    """Test SafetyChecker pattern compilation."""

    def test_pattern_compilation_cached(self):
        """Test that compiled patterns are cached."""
        config = SafetyConfig(
            sensitive_patterns=[],
        )
        checker = SafetyChecker(config=config)

        pattern1 = checker._compile_pattern(r"\d+")
        pattern2 = checker._compile_pattern(r"\d+")

        assert pattern1 is pattern2

    def test_pattern_not_found_returns_empty(self):
        """Test pattern lookup returns empty list for unknown pattern."""
        config = SafetyConfig(
            sensitive_patterns=[],
        )
        checker = SafetyChecker(config=config)

        result = checker._check_personal_info("no sensitive data here")
        assert result["detected"] is False


class TestSafetyCheckerInputValidation:
    """Test SafetyChecker with input validation scenarios."""

    def test_normal_text_passes(self):
        """Test normal medical question passes safety check."""
        checker = SafetyChecker()
        result = checker.check("请问糖尿病的诊断标准是什么？")

        assert result.passed is True
        assert result.risk_level in ["low", "medium", "high"]

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

    def test_political_content_blocked(self):
        """Test politically sensitive content is blocked."""
        checker = SafetyChecker()
        text = "这个问题涉及政治敏感内容"
        result = checker.check(text)

        assert result.passed is False
        assert result.risk_level == "high"

    def test_adult_content_blocked(self):
        """Test adult content is blocked."""
        checker = SafetyChecker()
        text = "这个内容包含色情相关信息"
        result = checker.check(text)

        assert result.passed is False
        assert result.risk_level == "high"

    def test_disabled_safety_check_passes_all(self):
        """Test disabled safety check passes all content."""
        config = SafetyConfig(enable=False)
        checker = SafetyChecker(config=config)

        text = "任何内容包括政治敏感"
        result = checker.check(text)

        assert result.passed is True
