"""Safety module — public API.

Other modules MUST import from here instead of safety.internal modules.
"""

from __future__ import annotations

from typing import Protocol

# SafetyResult and SafetyCheckPort are co-located so they share the same type.
# The concrete implementation (SafetyChecker in checker.py) also imports from here,
# ensuring Protocol structural typing works with mypy.
from src.common.safety.checker import SafetyResult


class SafetyCheckPort(Protocol):
    """Minimal interface: what other modules need from SafetyChecker."""

    def check(self, text: str) -> SafetyResult: ...
