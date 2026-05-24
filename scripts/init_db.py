"""
Database initialization script - recreates all tables

Usage:
    uv run python scripts/init_db.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from app.core.database import get_engine, get_session_factory
from app.models.database import Base


async def init_db():
    """Create all database tables"""
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

    print("\n[*] Creating all tables...")
    async with factory() as session:
        print("  [-] Dropping old tables (if exist)...")
        try:
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

        print("  [-] Creating new tables...")
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            print("    [+] All tables created!")
        except Exception as e:
            print(f"    [!] Create error: {e}")
            return False

    print("\n[*] Verifying table structure...")
    async with factory() as session:
        result = await session.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """))
        tables = [row[0] for row in result.fetchall()]

        expected_tables = ["documents", "chunks", "headings", "conversations", "messages"]
        for table in expected_tables:
            if table in tables:
                print(f"  [+] {table}")
            else:
                print(f"  [!] {table} (NOT FOUND)")
                return False

    print(f"\n[+] Database initialized! Created {len(expected_tables)} tables")
    return True


async def main():
    success = await init_db()
    if not success:
        print("\n[!] Database initialization FAILED")
        sys.exit(1)
    print("\n[+] You can now restart the application")


if __name__ == "__main__":
    asyncio.run(main())