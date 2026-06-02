"""Tests for DatasetManager and DatasetValidator."""

import shutil
import tempfile
from pathlib import Path

import pytest

from rag.evaluation.dataset_manager import (
    DatasetManager,
    DatasetValidator,
    ValidationReport,
)


@pytest.fixture
def temp_dir():
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture
def sample_data():
    return [
        {
            "query_id": "q1",
            "query_text": "糖尿病患者如何选择降糖药物？",
            "query_type": "drug",
            "relevant_doc_ids": ["doc1", "doc2"],
        },
        {
            "query_id": "q2",
            "query_text": "高血压的诊断标准是什么？",
            "query_type": "diagnosis",
            "relevant_doc_ids": ["doc3"],
        },
    ]


class TestDatasetValidator:
    def test_validate_valid_dataset(self, sample_data):
        validator = DatasetValidator()
        report = validator.validate(sample_data)
        assert report.is_valid
        assert len(report.errors) == 0

    def test_validate_missing_query_id(self):
        validator = DatasetValidator()
        data = [{"query_text": "test"}]
        report = validator.validate(data)
        assert not report.is_valid
        assert any("query_id" in e for e in report.errors)

    def test_validate_duplicate_ids(self):
        validator = DatasetValidator()
        data = [
            {"query_id": "q1", "query_text": "test1"},
            {"query_id": "q1", "query_text": "test2"},
        ]
        report = validator.validate(data)
        assert not report.is_valid
        assert any("duplicate" in e.lower() for e in report.errors)


class TestDatasetManager:
    def test_create_dataset(self, temp_dir, sample_data):
        manager = DatasetManager(base_dir=temp_dir)
        metadata = manager.create_dataset("test_dataset", sample_data, tags=["test"])

        assert metadata.dataset_id is not None
        assert metadata.count == 2
        assert metadata.validated is True

    def test_list_datasets(self, temp_dir, sample_data):
        manager = DatasetManager(base_dir=temp_dir)
        manager.create_dataset("dataset1", sample_data)

        datasets = manager.list_datasets()
        assert len(datasets) == 1
        assert datasets[0]["name"] == "dataset1"

    def test_get_dataset(self, temp_dir, sample_data):
        manager = DatasetManager(base_dir=temp_dir)
        manager.create_dataset("dataset1", sample_data)

        # 通过 ID 获取
        dataset_id = manager.list_datasets()[0]["dataset_id"]
        loaded = manager.get_dataset(dataset_id)

        assert len(loaded) == 2
        assert loaded[0]["query_text"] == sample_data[0]["query_text"]

    def test_delete_dataset(self, temp_dir, sample_data):
        manager = DatasetManager(base_dir=temp_dir)
        manager.create_dataset("dataset1", sample_data)

        dataset_id = manager.list_datasets()[0]["dataset_id"]
        manager.delete_dataset(dataset_id)

        assert len(manager.list_datasets()) == 0

    def test_create_version(self, temp_dir, sample_data):
        manager = DatasetManager(base_dir=temp_dir)
        metadata = manager.create_dataset("dataset1", sample_data)

        new_data = sample_data + [
            {
                "query_id": "q3",
                "query_text": "新的测试查询",
                "query_type": "treatment",
            }
        ]
        new_version = manager.create_version(metadata.dataset_id, new_data, tag="v2.0")

        assert new_version == "v2.0"
        loaded = manager.get_dataset(metadata.dataset_id, version="v2.0")
        assert len(loaded) == 3

    def test_validate_dataset_file(self, temp_dir, sample_data):
        # 创建临时 JSONL 文件
        data_file = temp_dir / "test.jsonl"
        with open(data_file, "w", encoding="utf-8") as f:
            for item in sample_data:
                f.write(__import__("json").dumps(item, ensure_ascii=False) + "\n")

        manager = DatasetManager(base_dir=temp_dir)
        report = manager.validate_dataset(data_file)

        assert report.is_valid
        assert report.stats["total"] == 2

    def test_validation_report_structure(self):
        """Test ValidationReport dataclass structure"""
        report = ValidationReport(
            is_valid=True,
            errors=[],
            warnings=["test warning"],
            stats={"total": 10},
        )
        assert report.is_valid is True
        assert len(report.warnings) == 1
        assert report.stats["total"] == 10
