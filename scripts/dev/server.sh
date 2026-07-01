#!/usr/bin/env bash
# 启动后端 (port 8000)
cd "$(dirname "$0")/../.."
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
