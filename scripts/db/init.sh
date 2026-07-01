#!/usr/bin/env bash
# 初始化 PostgreSQL 数据库
cd "$(dirname "$0")/../.."
uv run python scripts/db/init_db.py
