import pytest
from unittest.mock import patch, MagicMock

from app.models.schemas import RetrievedNode
from rag.reranker.cross_encoder import Reranker


class TestRerankerInit:
    """Test Reranker initialization."""

    def test_init_uses_defaults_from_settings(self):
        """Reranker should use settings when no args provided."""
        with patch('rag.reranker.cross_encoder.get_settings') as mock_settings:
            mock_settings.return_value.models.reranker.name = "test-model"
            mock_settings.return_value.models.reranker.device = "cpu"
            mock_settings.return_value.models.reranker.batch_size = 16
            mock_settings.return_value.models.reranker.max_length = 512
            mock_settings.return_value.models.reranker.estimated_memory_mb = 1800

            reranker = Reranker()

            assert reranker.model_name == "test-model"
            assert reranker.device == "cpu"
            assert reranker.batch_size == 16
            assert reranker.max_length == 512
            assert reranker.estimated_memory_mb == 1800

    def test_init_accepts_custom_args(self):
        """Reranker should accept custom model_name, device, batch_size, max_length."""
        reranker = Reranker(
            model_name="custom-model",
            device="cuda",
            batch_size=32,
            max_length=256,
        )

        assert reranker.model_name == "custom-model"
        assert reranker.device == "cuda"
        assert reranker.batch_size == 32
        assert reranker.max_length == 256

    def test_init_model_not_loaded(self):
        """Reranker should not load model on init (lazy loading)."""
        reranker = Reranker(model_name="test-model")
        assert reranker.model is None
        assert reranker._model_on_gpu is False
        assert reranker._model_on_cpu is False


class TestRerankerEnsureModelLoaded:
    """Test Reranker _ensure_model_loaded."""

    def test_ensure_model_loaded_loads_model_to_cpu(self):
        """First access should load model to CPU."""
        with patch('rag.reranker.cross_encoder.get_settings') as mock_settings:
            mock_settings.return_value.models.reranker.name = "test-model"
            mock_settings.return_value.models.reranker.device = "cpu"
            mock_settings.return_value.models.reranker.batch_size = 16
            mock_settings.return_value.models.reranker.max_length = 512
            mock_settings.return_value.models.reranker.estimated_memory_mb = 1800

            reranker = Reranker()

            mock_cross_encoder = MagicMock()
            with patch('sentence_transformers.CrossEncoder', return_value=mock_cross_encoder):
                reranker._ensure_model_loaded()

            assert reranker.model is not None
            assert reranker._model_on_cpu is True
            assert reranker._model_on_gpu is False

    def test_ensure_model_loaded_called_once(self):
        """Model should only be loaded once."""
        with patch('rag.reranker.cross_encoder.get_settings') as mock_settings:
            mock_settings.return_value.models.reranker.name = "test-model"
            mock_settings.return_value.models.reranker.device = "cpu"
            mock_settings.return_value.models.reranker.batch_size = 16
            mock_settings.return_value.models.reranker.max_length = 512
            mock_settings.return_value.models.reranker.estimated_memory_mb = 1800

            reranker = Reranker()
            reranker.model = MagicMock()  # Pre-set model to avoid actual loading
            reranker._ensure_model_loaded()
            reranker._ensure_model_loaded()

            assert reranker.model is not None


class TestRerankerEnsureOnGpu:
    """Test Reranker ensure_on_gpu."""

    def test_ensure_on_gpu_returns_true_when_already_on_gpu(self):
        """If model already on GPU, should return True without reloading."""
        with patch('rag.reranker.cross_encoder.get_settings') as mock_settings:
            mock_settings.return_value.models.reranker.name = "test-model"
            mock_settings.return_value.models.reranker.device = "cpu"
            mock_settings.return_value.models.reranker.batch_size = 16
            mock_settings.return_value.models.reranker.max_length = 512
            mock_settings.return_value.models.reranker.estimated_memory_mb = 1800
            mock_settings.return_value.models.gpu_safety_margin_mb = 1024

            reranker = Reranker()
            reranker._model_on_gpu = True

            result = reranker.ensure_on_gpu()

            assert result is True

    def test_ensure_on_gpu_loads_model_if_not_loaded(self):
        """Should call _ensure_model_loaded if model is None."""
        with patch('rag.reranker.cross_encoder.get_settings') as mock_settings:
            mock_settings.return_value.models.reranker.name = "test-model"
            mock_settings.return_value.models.reranker.device = "cpu"
            mock_settings.return_value.models.reranker.batch_size = 16
            mock_settings.return_value.models.reranker.max_length = 512
            mock_settings.return_value.models.reranker.estimated_memory_mb = 1800
            mock_settings.return_value.models.gpu_safety_margin_mb = 1024

            reranker = Reranker()
            reranker._ensure_model_loaded = MagicMock()
            reranker.model = MagicMock()

            reranker.ensure_on_gpu()

            reranker._ensure_model_loaded.assert_called_once()

    def test_ensure_on_gpu_insufficient_memory_returns_false(self):
        """Should return False when GPU memory is insufficient."""
        with patch('rag.reranker.cross_encoder.get_settings') as mock_settings:
            mock_settings.return_value.models.reranker.name = "test-model"
            mock_settings.return_value.models.reranker.device = "cpu"
            mock_settings.return_value.models.reranker.batch_size = 16
            mock_settings.return_value.models.reranker.max_length = 512
            mock_settings.return_value.models.reranker.estimated_memory_mb = 1800
            mock_settings.return_value.models.gpu_safety_margin_mb = 1024

            reranker = Reranker()
            reranker.model = MagicMock()
            reranker._ensure_model_loaded = MagicMock()

            mock_gpu_manager = MagicMock()
            mock_gpu_manager.get_memory_info.return_value = {"free_mb": 1024}

            with patch('rag.reranker.cross_encoder.GPUMemoryManager') as mock_gpum:
                mock_gpum.get_instance.return_value = mock_gpu_manager

                result = reranker.ensure_on_gpu()

                assert result is False
                assert reranker._model_on_gpu is False

    def test_ensure_on_gpu_successful_migration(self):
        """Should successfully migrate model to GPU."""
        with patch('rag.reranker.cross_encoder.get_settings') as mock_settings:
            mock_settings.return_value.models.reranker.name = "test-model"
            mock_settings.return_value.models.reranker.device = "cpu"
            mock_settings.return_value.models.reranker.batch_size = 16
            mock_settings.return_value.models.reranker.max_length = 512
            mock_settings.return_value.models.reranker.estimated_memory_mb = 1800
            mock_settings.return_value.models.gpu_safety_margin_mb = 1024

            reranker = Reranker()
            reranker.model = MagicMock()
            reranker._ensure_model_loaded = MagicMock()

            mock_gpu_manager = MagicMock()
            mock_gpu_manager.get_memory_info.return_value = {"free_mb": 8192}

            with patch('rag.reranker.cross_encoder.GPUMemoryManager') as mock_gpum:
                mock_gpum.get_instance.return_value = mock_gpu_manager

                result = reranker.ensure_on_gpu()

                assert result is True
                assert reranker._model_on_gpu is True
                assert reranker._model_on_cpu is False
                reranker.model.to.assert_called_once_with("cuda")
                mock_gpu_manager.register_model.assert_called_once_with("reranker", 1800)


class TestRerankerMoveToCpu:
    """Test Reranker move_to_cpu."""

    def test_move_to_cpu_returns_true_when_not_on_gpu(self):
        """Should return True immediately if model not on GPU."""
        reranker = Reranker(model_name="test-model")
        reranker._model_on_gpu = False

        result = reranker.move_to_cpu()

        assert result is True

    def test_move_to_cpu_migrates_model_to_cpu(self):
        """Should migrate model from GPU to CPU."""
        with patch('rag.reranker.cross_encoder.get_settings') as mock_settings:
            mock_settings.return_value.models.reranker.name = "test-model"
            mock_settings.return_value.models.reranker.device = "cpu"
            mock_settings.return_value.models.reranker.batch_size = 16
            mock_settings.return_value.models.reranker.max_length = 512
            mock_settings.return_value.models.reranker.estimated_memory_mb = 1800

            reranker = Reranker()
            reranker.model = MagicMock()
            reranker._model_on_gpu = True
            reranker._ensure_model_loaded = MagicMock()

            mock_gpu_manager = MagicMock()

            with patch('rag.reranker.cross_encoder.GPUMemoryManager') as mock_gpum:
                mock_gpum.get_instance.return_value = mock_gpu_manager

                result = reranker.move_to_cpu()

                assert result is True
                assert reranker._model_on_gpu is False
                assert reranker._model_on_cpu is True
                reranker.model.to.assert_called_once_with("cpu")
                mock_gpu_manager.unregister_model.assert_called_once_with("reranker")

    def test_move_to_cpu_clears_cuda_cache(self):
        """Should clear CUDA cache after moving to CPU."""
        with patch('rag.reranker.cross_encoder.get_settings') as mock_settings:
            mock_settings.return_value.models.reranker.name = "test-model"
            mock_settings.return_value.models.reranker.device = "cpu"
            mock_settings.return_value.models.reranker.batch_size = 16
            mock_settings.return_value.models.reranker.max_length = 512
            mock_settings.return_value.models.reranker.estimated_memory_mb = 1800

            reranker = Reranker()
            reranker.model = MagicMock()
            reranker._model_on_gpu = True
            reranker._ensure_model_loaded = MagicMock()

            mock_gpu_manager = MagicMock()

            with patch('rag.reranker.cross_encoder.GPUMemoryManager') as mock_gpum:
                mock_gpum.get_instance.return_value = mock_gpu_manager

                with patch('torch.cuda.empty_cache') as mock_clear:
                    reranker.move_to_cpu()

                    mock_clear.assert_called_once()


class TestRerankerIsOnGpu:
    """Test Reranker is_on_gpu."""

    def test_is_on_gpu_returns_true_when_on_gpu(self):
        """Should return True when model is on GPU."""
        reranker = Reranker(model_name="test-model")
        reranker._model_on_gpu = True

        assert reranker.is_on_gpu() is True

    def test_is_on_gpu_returns_false_when_not_on_gpu(self):
        """Should return False when model is not on GPU."""
        reranker = Reranker(model_name="test-model")
        reranker._model_on_gpu = False

        assert reranker.is_on_gpu() is False


class TestRerankerRerank:
    """Test Reranker rerank."""

    def test_rerank_empty_candidates_returns_empty(self):
        """Should return empty list when candidates is empty."""
        reranker = Reranker(model_name="test-model")

        result = reranker.rerank("query", [])

        assert result == []

    def test_rerank_ensures_model_loaded(self):
        """Should ensure model is loaded before reranking."""
        reranker = Reranker(model_name="test-model")
        reranker._ensure_model_loaded = MagicMock()
        reranker.model = MagicMock()
        reranker.model.predict = MagicMock(return_value=[0.8])
        reranker.ensure_on_gpu = MagicMock()

        candidates = [
            RetrievedNode(node_id="1", content="content1", score=0.9, metadata={}),
        ]

        reranker.rerank("query", candidates)

        reranker._ensure_model_loaded.assert_called_once()

    def test_rerank_returns_reranked_nodes_sorted_by_score(self):
        """Should return nodes sorted by reranking score."""
        reranker = Reranker(model_name="test-model")
        reranker._ensure_model_loaded = MagicMock()
        reranker.ensure_on_gpu = MagicMock()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.2, 0.8, 0.5]
        reranker.model = mock_model

        candidates = [
            RetrievedNode(node_id="1", content="content1", score=0.9, metadata={}),
            RetrievedNode(node_id="2", content="content2", score=0.7, metadata={}),
            RetrievedNode(node_id="3", content="content3", score=0.8, metadata={}),
        ]

        result = reranker.rerank("query", candidates)

        assert len(result) == 3
        assert result[0].node_id == "2"
        assert result[1].node_id == "3"
        assert result[2].node_id == "1"

    def test_rerank_with_return_documents_false(self):
        """Should return empty content when return_documents is False."""
        reranker = Reranker(model_name="test-model")
        reranker._ensure_model_loaded = MagicMock()
        reranker.ensure_on_gpu = MagicMock()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]
        reranker.model = mock_model

        candidates = [
            RetrievedNode(node_id="1", content="content1", score=0.9, metadata={}),
        ]

        result = reranker.rerank("query", candidates, return_documents=False)

        assert result[0].content == ""
        assert result[0].node_id == "1"

    def test_rerank_preserves_metadata(self):
        """Should preserve metadata from original candidates."""
        reranker = Reranker(model_name="test-model")
        reranker._ensure_model_loaded = MagicMock()
        reranker.ensure_on_gpu = MagicMock()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]
        reranker.model = mock_model

        metadata = {"doc_id": "doc-1", "source": "test"}
        candidates = [
            RetrievedNode(node_id="1", content="content1", score=0.9, metadata=metadata),
        ]

        result = reranker.rerank("query", candidates)

        assert result[0].metadata == metadata


class TestRerankerNormalizeScores:
    """Test Reranker _normalize_scores."""

    def test_normalize_empty_scores(self):
        """Should return empty list for empty input."""
        reranker = Reranker(model_name="test-model")

        result = reranker._normalize_scores([])

        assert result == []

    def test_normalize_single_score(self):
        """Should return 0.5 for single score (min == max)."""
        reranker = Reranker(model_name="test-model")

        result = reranker._normalize_scores([0.5])

        assert result == [0.5]

    def test_normalize_identical_scores(self):
        """Should return 0.5 for all scores when min == max."""
        reranker = Reranker(model_name="test-model")

        result = reranker._normalize_scores([0.5, 0.5, 0.5])

        assert result == [0.5, 0.5, 0.5]

    def test_normalize_different_scores(self):
        """Should normalize scores to [0, 1] range."""
        reranker = Reranker(model_name="test-model")

        result = reranker._normalize_scores([0.2, 0.4, 0.6, 0.8])

        assert result[0] == 0.0
        assert result[3] == 1.0
        assert result[1] == pytest.approx(0.333, 0.01)
        assert result[2] == pytest.approx(0.666, 0.01)

    def test_normalize_handles_numpy_array(self):
        """Should handle numpy array input."""
        reranker = Reranker(model_name="test-model")

        import numpy as np
        scores = np.array([0.3, 0.6, 0.9])

        result = reranker._normalize_scores(scores)

        assert len(result) == 3

    def test_normalize_handles_tensor(self):
        """Should handle torch tensor input."""
        reranker = Reranker(model_name="test-model")

        import torch
        scores = torch.tensor([0.3, 0.6, 0.9])

        result = reranker._normalize_scores(scores)

        assert len(result) == 3


class TestRerankerGetDevice:
    """Test Reranker get_device."""

    def test_get_device_returns_cuda_when_available(self):
        """Should return 'cuda' when torch.cuda.is_available is True."""
        reranker = Reranker(model_name="test-model")

        with patch('torch.cuda.is_available', return_value=True):
            result = reranker.get_device()
            assert result == "cuda"

    def test_get_device_returns_cpu_when_cuda_unavailable(self):
        """Should return 'cpu' when torch.cuda.is_available is False."""
        reranker = Reranker(model_name="test-model")

        with patch('torch.cuda.is_available', return_value=False):
            result = reranker.get_device()
            assert result == "cpu"