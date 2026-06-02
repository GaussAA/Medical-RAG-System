from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas import RetrievedNode


class TestVectorRetriever:
    """Tests for VectorRetriever."""

    @pytest.mark.asyncio
    async def test_add_uses_executor_not_blocking(self):
        """Test that add() does not block on embedding encode."""
        from rag.retrieval.vector_retriever import VectorRetriever

        vr = VectorRetriever()
        mock_client = MagicMock()
        mock_client.upsert = MagicMock()

        # Properly patch the class-level _client via _get_client
        with patch.object(VectorRetriever, "_get_client", return_value=mock_client):
            # Mock embedding model
            vr._embedding_model = MagicMock()
            vr._embedding_model.encode = MagicMock(
                return_value=MagicMock(tolist=MagicMock(return_value=[[0.1] * 1024]))
            )

            nodes = [
                RetrievedNode(
                    node_id="test1",
                    content="test content",
                    score=0.9,
                    metadata={},
                )
            ]

            with patch("asyncio.get_running_loop") as mock_get_loop:
                mock_loop_instance = MagicMock()
                mock_get_loop.return_value = mock_loop_instance
                mock_loop_instance.run_in_executor = AsyncMock(return_value=[[0.1] * 1024])

                await vr.add(nodes)

                # Verify run_in_executor was called (async path)
                mock_loop_instance.run_in_executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_uses_pre_encoded_embedding_when_present(self):
        """Test that add() uses pre-encoded embedding from metadata without calling encode."""
        from rag.retrieval.vector_retriever import VectorRetriever

        vr = VectorRetriever()
        mock_client = MagicMock()
        mock_client.upsert = MagicMock()

        with patch.object(VectorRetriever, "_get_client", return_value=mock_client):
            # Mock embedding model - should NOT be called
            vr._embedding_model = MagicMock()
            vr._embedding_model.encode = MagicMock()

            pre_encoded = [0.1] * 1024
            nodes = [
                RetrievedNode(
                    node_id="test1",
                    content="test content",
                    score=0.9,
                    metadata={"embedding": pre_encoded},
                )
            ]

            await vr.add(nodes)

            # Verify encode was NOT called since embedding was pre-encoded
            vr._embedding_model.encode.assert_not_called()

            # Verify upsert was called with the pre-encoded vector
            mock_client.upsert.assert_called_once()
            call_args = mock_client.upsert.call_args
            points = call_args[1]["points"]
            assert points[0]["vector"] == pre_encoded
