#!/usr/bin/env bash
# 清理缓存文件
cd "$(dirname "$0")/../.."

echo "[-] 清理 __pycache__ ..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "[-] 清理 .pyc ..."
find . -type f -name "*.pyc" -delete 2>/dev/null

echo "[-] 清理 .coverage .pytest_cache .ruff_cache .mypy_cache ..."
rm -rf .coverage .pytest_cache .ruff_cache .mypy_cache 2>/dev/null

echo "[-] 清理 .eggs *.egg-info ..."
rm -rf .eggs *.egg-info 2>/dev/null

echo "✅ 缓存清理完成"
