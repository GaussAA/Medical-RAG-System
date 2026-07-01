"""
Safety checker - migrated from app/core/safety.py.

Provides query safety validation and sensitive content detection.
"""

import re
from dataclasses import dataclass

from loguru import logger

from src.common.config.settings import get_settings


@dataclass
class SafetyResult:
    """Result of a safety check."""
    passed: bool
    sanitized_text: str
    warnings: list[str]


class SafetyChecker:
    """Query safety checker with pattern-based sanitization."""

    def __init__(self):
        settings = get_settings()
        self.config = settings.safety
        self._patterns: list[tuple[re.Pattern, str]] = []
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns from config."""
        self._patterns = []
        for sp in self.config.sensitive_patterns:
            try:
                compiled = re.compile(sp.pattern)
                self._patterns.append((compiled, sp.replacement))
            except re.error as e:
                logger.warning(f"Failed to compile pattern '{sp.name}': {e}")

    def check(self, text: str) -> SafetyResult:
        """Check text for safety concerns and sanitize sensitive content."""
        if not self.config.enable:
            return SafetyResult(passed=True, sanitized_text=text, warnings=[])

        warnings: list[str] = []

        if self.config.sensitive_words_check:
            sensitive_found = self._check_sensitive_words(text)
            if sensitive_found:
                warnings.append("检测到敏感词汇")

        sanitized = text
        if self.config.privacy_protection:
            sanitized = self._sanitize_privacy(text)

        return SafetyResult(
            passed=len(warnings) == 0,
            sanitized_text=sanitized,
            warnings=warnings,
        )

    def _check_sensitive_words(self, text: str) -> bool:
        """Check for sensitive keywords. Placeholder for future expansion."""
        return False

    def _sanitize_privacy(self, text: str) -> str:
        """Replace sensitive patterns with placeholders."""
        result = text
        for pattern, replacement in self._patterns:
            result = pattern.sub(replacement, result)
        return result
