"""
Qdrant collection initialization script — v1.18 features enabled.

Features used:
- TurboQuant 4-bit: 8x vector compression without recall loss
- ScalarQuant + ProductQuant as fallback options (see config)

Usage:
    uv run python scripts/init_qdrant.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    TurboQuantBitSize,
    TurboQuantQuantizationConfig,
    TurboQuantization,
    VectorParams,
)

from src.common.config.settings import get_settings


def init_qdrant():
    """Create Qdrant collection with TurboQuant for optimal storage/performance."""
    settings = get_settings()
    qdrant_config = settings.database.qdrant

    print(f"[*] Connecting to Qdrant at {qdrant_config.url}...")

    client = QdrantClient(
        url=qdrant_config.url,
        timeout=qdrant_config.timeout,
        prefer_grpc=qdrant_config.prefer_grpc,
    )

    collection_name = qdrant_config.collection

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    print(f"[+] Existing collections: {collection_names}")

    if collection_name in collection_names:
        print(f"[!] Collection '{collection_name}' already exists, deleting...")
        client.delete_collection(collection_name=collection_name)
        print("    [+] Deleted")

    # Get embedding dimension from settings
    embedding_dim = settings.models.embedding.dimension
    print(f"[*] Creating collection '{collection_name}' (dim={embedding_dim})...")

    # ── TurboQuant 4-bit: 8x compression, near-zero recall loss ──
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "": VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        },
        quantization_config=TurboQuantization(
            turbo=TurboQuantQuantizationConfig(
                always_ram=True,            # 量化向量常驻 RAM 加速搜索
                bits=TurboQuantBitSize.BITS4,  # 4-bit → 8x 压缩
            ),
        ),
    )
    print(f"    [+] Collection created (TurboQuant 4-bit, 8x compression)")

    # Verify
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    if collection_name in collection_names:
        info = client.get_collection(collection_name=collection_name)
        print(f"\n[+] Collection '{collection_name}' is ready!")
        print(f"    Status: {info.status}")
        print(f"    Vectors count: {info.vectors_count}")
        print(f"    Quantization: {info.config.quantization_config if info.config else 'N/A'}")
        return True
    else:
        print("\n[!] Failed to create collection")
        return False


if __name__ == "__main__":
    success = init_qdrant()
    sys.exit(0 if success else 1)
