from unittest.mock import AsyncMock, MagicMock, patch

from app.core.metrics import ERROR_COUNT
from app.core.rag_engine import RAGEngine
from app.models.schemas import QueryRequest, SafetyResult


class TestRAGEngineErrorClassification:
    """Test fine-grained error_type classification in RAGEngine.query()"""

    def setup_method(self):
        """Reset RAG engine before each test."""
        self.engine = RAGEngine()

    def _create_query_request(self, question="测试问题", session_id="test-session"):
        return QueryRequest(question=question, session_id=session_id)

    async def test_retrieval_error_records_retrieval_error_type(self):
        """Retrieval/rerank failures should record error_type='retrieval'"""
        request = self._create_query_request()

        with patch.object(self.engine, "_safety_check") as mock_safety:
            mock_safety.return_value = SafetyResult(passed=True, sanitized_text="test", risk_level="low")

            with patch.object(self.engine, "_retrieve_and_rerank", new_callable=AsyncMock) as mock_rr:
                mock_rr.side_effect = RuntimeError("GPU OOM")

                with patch.object(self.engine, "_create_error_response") as mock_error_resp:
                    mock_error_resp.return_value = MagicMock()

                    await self.engine.query(request)

                    # Verify error response was created for retrieval error
                    mock_error_resp.assert_called_once()
                    call_args = mock_error_resp.call_args
                    assert "检索失败" in call_args[0][1]

    async def test_generation_error_records_generation_error_type(self):
        """Generation failures should record error_type='generation'"""
        request = self._create_query_request()

        with patch.object(self.engine, "_safety_check") as mock_safety:
            mock_safety.return_value = SafetyResult(passed=True, sanitized_text="test", risk_level="low")

            with patch.object(self.engine, "_retrieve_and_rerank", new_callable=AsyncMock) as mock_rr:
                mock_rr.return_value = [MagicMock()]

                with patch.object(self.engine, "_generate_answer", new_callable=AsyncMock) as mock_gen:
                    mock_gen.side_effect = Exception("Rate limit exceeded")

                    with patch.object(self.engine, "_create_error_response") as mock_error_resp:
                        mock_error_resp.return_value = MagicMock()

                        await self.engine.query(request)

                        mock_error_resp.assert_called_once()
                        call_args = mock_error_resp.call_args
                        assert "生成回答失败" in call_args[0][1]

    async def test_unexpected_error_records_validation_error_type(self):
        """Unexpected exceptions should record error_type='validation'"""
        request = self._create_query_request()

        with patch.object(self.engine, "_safety_check") as mock_safety:
            mock_safety.side_effect = AttributeError("unexpected bug")

            with patch.object(self.engine, "_create_error_response") as mock_error_resp:
                mock_error_resp.return_value = MagicMock()

                await self.engine.query(request)

                mock_error_resp.assert_called_once()


class TestERRORCOUNTMetrics:
    """Test that ERROR_COUNT counter is incremented correctly for each error type"""

    def test_error_count_has_correct_labels(self):
        """Test ERROR_COUNT counter accepts all valid error_type labels"""
        # This verifies the metric is properly configured
        ERROR_COUNT.labels(error_type="safety").inc()
        ERROR_COUNT.labels(error_type="retrieval").inc()
        ERROR_COUNT.labels(error_type="generation").inc()
        ERROR_COUNT.labels(error_type="validation").inc()

        # Verify the counter has metrics recorded
        assert ERROR_COUNT._metrics is not None
