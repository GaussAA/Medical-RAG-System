#!/usr/bin/env bash
# 代码格式化 (ruff)
cd "$(dirname "$0")/../.."
uv run ruff format src/ frontend/ tests/ scripts/
echo ""
echo "✅ 格式化完成"
