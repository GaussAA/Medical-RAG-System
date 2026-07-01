#!/usr/bin/env bash
# 代码风格检查 (ruff)
cd "$(dirname "$0")/../.."
uv run --extra dev ruff check src/ frontend/
