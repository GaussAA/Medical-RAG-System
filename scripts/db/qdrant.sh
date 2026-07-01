#!/usr/bin/env bash
# 初始化 Qdrant 向量库
cd "$(dirname "$0")/../.."
uv run python scripts/db/init_qdrant.py
