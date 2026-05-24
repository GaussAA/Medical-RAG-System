import re
from dataclasses import dataclass, field
from typing import Any

from app.models.schemas import SafetyResult


@dataclass
class SafetyConfig:
    """Safety configuration for dependency injection."""

    enable: bool = True
    sensitive_words_check: bool = True
    privacy_protection: bool = True
    political_check: bool = True
    adult_content_check: bool = True
    sensitive_patterns: list[Any] = field(default_factory=list)
    privacy_keywords: list[str] = field(
        default_factory=lambda: [
            "病历号",
            "患者姓名",
            "主治医生",
            "住院号",
        ]
    )
    political_keywords: list[str] = field(
        default_factory=lambda: [
            "领导人",
            "政治敏感",
            "分裂",
            "颠覆",
        ]
    )
    adult_keywords: list[str] = field(
        default_factory=lambda: [
            "色情",
            "淫秽",
            "赌博",
            "毒品",
            "武器",
        ]
    )


class SafetyChecker:
    _compiled_patterns: dict[str, re.Pattern] = {}

    def __init__(self, config: SafetyConfig | None = None):
        if config is not None:
            self.config = config
        else:
            from config.settings import get_settings

            settings = get_settings()
            self.config = SafetyConfig(
                enable=settings.safety.enable,
                sensitive_words_check=settings.safety.sensitive_words_check,
                privacy_protection=settings.safety.privacy_protection,
                political_check=settings.safety.political_check,
                adult_content_check=settings.safety.adult_content_check,
                sensitive_patterns=settings.safety.sensitive_patterns,
            )

        self.sensitive_patterns = [
            {
                "name": p.name,
                "pattern": self._compile_pattern(p.pattern),
                "replacement": p.replacement,
            }
            for p in self.config.sensitive_patterns
        ]

        self.privacy_keywords = list(self.config.privacy_keywords)
        self.political_keywords = list(self.config.political_keywords)
        self.adult_keywords = list(self.config.adult_keywords)

    @classmethod
    def _compile_pattern(cls, pattern: str) -> re.Pattern:
        """Cache compiled regex patterns."""
        if pattern not in cls._compiled_patterns:
            cls._compiled_patterns[pattern] = re.compile(pattern)
        return cls._compiled_patterns[pattern]

    def check(self, text: str) -> SafetyResult:
        if not self.config.enable:
            return SafetyResult(passed=True, sanitized_text=text, risk_level="low")

        flagged_types = []
        sanitized = text

        if self.config.sensitive_words_check:
            personal_result = self._check_personal_info(sanitized)
            if personal_result["detected"]:
                flagged_types.append("personal_info")
                sanitized = personal_result["sanitized"]

        if self.config.privacy_protection:
            privacy_result = self._check_privacy(sanitized)
            if privacy_result["detected"]:
                flagged_types.extend(privacy_result["types"])
                sanitized = privacy_result["sanitized"]

        if self.config.political_check:
            political_result = self._check_political(sanitized)
            if political_result["detected"]:
                return SafetyResult(
                    passed=False,
                    flagged_types=["political_sensitive"],
                    sanitized_text=text,
                    risk_level="high",
                )

        if self.config.adult_content_check:
            adult_result = self._check_adult_content(sanitized)
            if adult_result["detected"]:
                return SafetyResult(
                    passed=False,
                    flagged_types=["adult_content"],
                    sanitized_text=text,
                    risk_level="high",
                )

        risk_level = self._calculate_risk_level(flagged_types)

        return SafetyResult(
            passed=True,
            flagged_types=flagged_types,
            sanitized_text=sanitized,
            risk_level=risk_level,
        )

    def _check_personal_info(self, text: str) -> dict[str, Any]:
        detected = False
        sanitized = text

        for pattern_info in self.sensitive_patterns:
            if pattern_info["pattern"].search(sanitized):
                detected = True
                sanitized = pattern_info["pattern"].sub(pattern_info["replacement"], sanitized)

        return {"detected": detected, "sanitized": sanitized}

    def _check_privacy(self, text: str) -> dict[str, Any]:
        detected_types = []
        sanitized = text

        for keyword in self.privacy_keywords:
            if keyword in sanitized:
                detected_types.append(f"privacy:{keyword}")
                sanitized = sanitized.replace(keyword, "[隐私信息]")

        return {
            "detected": len(detected_types) > 0,
            "types": detected_types,
            "sanitized": sanitized,
        }

    def _check_political(self, text: str) -> dict[str, Any]:
        for keyword in self.political_keywords:
            if keyword in text:
                return {"detected": True, "sanitized": text}

        return {"detected": False, "sanitized": text}

    def _check_adult_content(self, text: str) -> dict[str, Any]:
        for keyword in self.adult_keywords:
            if keyword in text:
                return {"detected": True, "sanitized": text}

        return {"detected": False, "sanitized": text}

    def _calculate_risk_level(self, flagged_types: list[str]) -> str:
        if not flagged_types:
            return "low"

        privacy_count = sum(1 for t in flagged_types if t.startswith("privacy:"))
        if privacy_count > 2:
            return "high"
        elif privacy_count > 0:
            return "medium"

        return "low"

    def sanitize(self, text: str) -> str:
        result = self.check(text)
        return result.sanitized_text
