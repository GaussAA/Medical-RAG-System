#!/usr/bin/env bash
# 启动前端 (port 8501)
cd "$(dirname "$0")/../.."
uv run streamlit run frontend/app.py
