"""
Database initialization script - recreates all tables

Usage:
    uv run python scripts/init_db.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import text

from src.common.database import Base, get_engine, get_session_factory
from src.conversation.models import Conversation, Message  # noqa: F401, E402

# Import slice models so they register with Base.metadata
from src.documents.models import Chunk, Document, Heading  # noqa: F401, E402


async def init_db():
    """Create all database tables in their respective schemas."""
    print("[*] Connecting to database...")

    engine = await get_engine()

    async with engine.begin() as conn:
        try:
            await conn.execute(text("SELECT 1"))
            print("[+] Database connection OK")
        except Exception as e:
            print(f"[!] Database connection FAILED: {e}")
            return False

    factory = get_session_factory()

    print("\n[*] Creating database schemas...")
    async with engine.begin() as conn:
        # Create schemas for each vertical slice
        await conn.execute(text("CREATE SCHEMA IF NOT EXISTS documents"))
        await conn.execute(text("CREATE SCHEMA IF NOT EXISTS conversation"))
        print("  [+] Schemas created: documents, conversation")

    print("\n[*] Creating all tables...")
    async with factory() as session:
        print("  [-] Dropping old tables (if exist)...")
        try:
            await session.execute(text("DROP TABLE IF EXISTS conversation.messages CASCADE"))
            await session.execute(text("DROP TABLE IF EXISTS conversation.conversations CASCADE"))
            await session.execute(text("DROP TABLE IF EXISTS documents.chunks CASCADE"))
            await session.execute(text("DROP TABLE IF EXISTS documents.headings CASCADE"))
            await session.execute(text("DROP TABLE IF EXISTS documents.documents CASCADE"))
            # Also clean old public schema tables if they exist
            await session.execute(text("DROP TABLE IF EXISTS messages CASCADE"))
            await session.execute(text("DROP TABLE IF EXISTS conversations CASCADE"))
            await session.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))
            await session.execute(text("DROP TABLE IF EXISTS headings CASCADE"))
            await session.execute(text("DROP TABLE IF EXISTS documents CASCADE"))
            await session.commit()
            print("    [+] Old tables dropped")
        except Exception as e:
            print(f"    [!] Drop error: {e}")
            await session.rollback()

        print("  [-] Creating new tables in slice schemas...")
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            print("    [+] All tables created!")
        except Exception as e:
            print(f"    [!] Create error: {e}")
            return False

    print("\n[*] Verifying table structure...")
    async with factory() as session:
        result = await session.execute(
            text("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema IN ('documents', 'conversation')
            ORDER BY table_schema, table_name
        """)
        )
        tables = [(row[0], row[1]) for row in result.fetchall()]

        expected_tables = [
            ("documents", "documents"),
            ("documents", "chunks"),
            ("documents", "headings"),
            ("conversation", "conversations"),
            ("conversation", "messages"),
        ]
        for schema, table in expected_tables:
            if (schema, table) in tables:
                print(f"  [+] {schema}.{table}")
            else:
                print(f"  [!] {schema}.{table} (NOT FOUND)")
                return False

    print(f"\n[+] Database initialized! Created {len(expected_tables)} tables in 2 schemas")
    return True


async def main():
    success = await init_db()
    if not success:
        print("\n[!] Database initialization FAILED")
        sys.exit(1)
    print("\n[+] You can now restart the application")


if __name__ == "__main__":
    asyncio.run(main())
