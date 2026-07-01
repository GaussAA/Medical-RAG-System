#!/usr/bin/env bash
# 类型检查 (mypy)
cd "$(dirname "$0")/../.."
uv run --extra dev mypy src/
