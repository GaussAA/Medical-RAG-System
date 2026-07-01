"""
Logging configuration for Medical RAG System.

Provides:
- Console logging (DEBUG level, colored)
- File logging (INFO level, daily rotation, 30-day retention)
- Automatic trace_id injection via contextvar
"""

import sys
from contextvars import ContextVar
from pathlib import Path

from loguru import logger

from src.common.config.settings import get_settings

# ponytail: set per-request via RAGEngine, auto-attached to every log line
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def _inject_request_id(record):
    """Patcher: attach current request_id to every log record as extra."""
    rid = request_id_var.get()
    if rid:
        record["extra"]["request_id"] = rid


def setup_logging() -> None:
    """Configure loguru with console and file sinks.

    Called once at application startup.
    """
    settings = get_settings()
    log_config = settings.logging

    # Remove default sink to reconfigure
    logger.remove()

    # Global patcher: inject request_id into every record
    logger.configure(patcher=_inject_request_id)

    # Console sink: colored, DEBUG+
    logger.add(
        sink=sys.stderr,
        level=log_config.console_level,
        format=log_config.console_format,
        colorize=True,
        backtrace=True,
        diagnose=settings.app.debug,
    )

    # File sink: INFO+, daily rotation, 30-day retention
    if log_config.file_enabled:
        log_dir = Path(log_config.file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        logger.add(
            sink=str(log_config.file_path),
            level=log_config.level,
            format=log_config.file_format,
            rotation=log_config.file_rotation,
            retention=log_config.file_retention,
            compression="gz",
            enqueue=True,  # Thread-safe for async apps
            backtrace=True,
        )

    logger.info("Logging initialized: console={}, file={}", True, log_config.file_enabled)
