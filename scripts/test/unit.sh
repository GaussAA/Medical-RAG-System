#!/usr/bin/env bash
# 单元测试
cd "$(dirname "$0")/../.."
uv run pytest tests/unit/ -v
