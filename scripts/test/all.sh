#!/usr/bin/env bash
# 全部测试
cd "$(dirname "$0")/../.."
uv run pytest tests/ -v
