"""
Qdrant collection initialization script

Usage:
    uv run python scripts/init_qdrant.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config.settings import get_settings


def init_qdrant():
    """Create Qdrant collection with proper vector configuration"""
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
    print(f"[*] Creating collection '{collection_name}' with vector dimension {embedding_dim}...")

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "": VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            )
        },
    )
    print(f"    [+] Collection '{collection_name}' created successfully!")

    # Verify
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    if collection_name in collection_names:
        print(f"\n[+] Qdrant collection '{collection_name}' is ready!")
        return True
    else:
        print("\n[!] Failed to create collection")
        return False


if __name__ == "__main__":
    success = init_qdrant()
    sys.exit(0 if success else 1)
