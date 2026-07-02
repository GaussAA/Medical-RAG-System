#!/usr/bin/env bash
# 运行测试
# 用法: scripts/test/run.sh [路径] [选项...]
#   路径: tests/unit/ (默认) | tests/integration/ | tests/unit/test_xxx.py
cd "$(dirname "$0")/../.."

TARGET="${1:-tests/unit/}"
shift 2>/dev/null || true

uv run pytest "$TARGET" -v --tb=short "$@"
