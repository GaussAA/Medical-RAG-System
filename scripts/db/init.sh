#!/usr/bin/env bash
# 初始化数据库（PostgreSQL + Qdrant）
# 用法: scripts/db/init.sh [--reset]
set -e
cd "$(dirname "$0")/../.."

if [ "$1" = "--reset" ]; then
  echo "⚠️ 将重置所有表（数据会丢失），确认? [y/N] "
  read -r ok
  [ "$ok" != "y" ] && echo "已取消" && exit 0
fi

echo "=== 初始化 PostgreSQL ==="
uv run python scripts/db/init_db.py

echo ""
echo "=== 初始化 Qdrant ==="
uv run python scripts/db/init_qdrant.py

echo ""
echo "✅ 数据库初始化完成"
echo "   PostgreSQL: 5 张表（documents / conversation schema）"
echo "   Qdrant:     collection 已创建"
