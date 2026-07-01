"""Dataset management for RAG evaluation datasets."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ValidationReport:
    """验证报告"""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    stats: dict[str, int]


class DatasetValidator:
    """数据集验证器"""

    REQUIRED_FIELDS = ["query_id", "query_text"]
    OPTIONAL_FIELDS = [
        "query_type",
        "relevant_doc_ids",
        "expected_keywords",
        "reference_answer",
        "difficulty",
        "safety_sensitive",
    ]

    def validate(self, dataset: list[dict]) -> ValidationReport:
        """验证数据集"""
        errors = []
        warnings = []
        stats = {
            "total": len(dataset),
            "missing_query_id": 0,
            "missing_query_text": 0,
            "duplicate_ids": 0,
        }

        seen_ids = set()
        for i, item in enumerate(dataset):
            # 检查必需字段
            if not item.get("query_id"):
                errors.append(f"Item {i}: missing query_id")
                stats["missing_query_id"] += 1

            if not item.get("query_text"):
                errors.append(f"Item {i}: missing query_text")
                stats["missing_query_text"] += 1

            # 检查重复 ID
            qid = item.get("query_id")
            if qid in seen_ids:
                errors.append(f"Item {i}: duplicate query_id '{qid}'")
                stats["duplicate_ids"] += 1
            seen_ids.add(qid)

        # 检查类型分布
        query_types: list[str] = []
        for item in dataset:
            qt = item.get("query_type")
            if isinstance(qt, str):
                query_types.append(qt)
        if query_types:
            type_dist: dict[str, int] = {}
            for qt in query_types:
                type_dist[qt] = type_dist.get(qt, 0) + 1
            if len(type_dist) < 2:
                warnings.append(f"Only {len(type_dist)} query type(s) found, consider adding variety")

        return ValidationReport(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )


@dataclass
class DatasetMetadata:
    """数据集元信息"""

    dataset_id: str
    name: str
    version: str
    created_at: str
    count: int
    tags: list[str]
    validated: bool = False


class DatasetManager:
    """数据集管理器"""

    def __init__(self, base_dir: Path = Path("data/datasets")):
        self.base_dir = base_dir
        self.manifest_path = base_dir / "manifest.json"
        self._ensure_base_dir()

    def _ensure_base_dir(self):
        """确保目录存在"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._write_manifest({})

    def _read_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest: dict):
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def create_dataset(
        self,
        name: str,
        data: list[dict],
        tags: list[str] | None = None,
    ) -> DatasetMetadata:
        """创建新数据集"""
        # 验证数据
        validator = DatasetValidator()
        report = validator.validate(data)
        if not report.is_valid:
            raise ValueError(f"Dataset validation failed: {report.errors}")

        # 生成 ID
        dataset_id = hashlib.md5(name.encode()).hexdigest()[:8]

        # 保存数据
        version = "v1.0"
        dataset_dir = self.base_dir / dataset_id / version
        dataset_dir.mkdir(parents=True, exist_ok=True)

        data_path = dataset_dir / "data.jsonl"
        with open(data_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 保存元信息
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            name=name,
            version=version,
            created_at=datetime.now().isoformat(),
            count=len(data),
            tags=tags or [],
            validated=True,
        )

        metadata_path = dataset_dir / "metadata.json"
        metadata_path.write_text(json.dumps(asdict(metadata), ensure_ascii=False, indent=2), encoding="utf-8")

        # 更新 manifest
        manifest = self._read_manifest()
        manifest[dataset_id] = {
            "name": name,
            "latest_version": version,
            "tags": tags or [],
        }
        self._write_manifest(manifest)

        return metadata

    def list_datasets(self) -> list[dict]:
        """列出所有数据集"""
        manifest = self._read_manifest()
        return [
            {
                "dataset_id": k,
                "name": v["name"],
                "latest_version": v["latest_version"],
                "tags": v.get("tags", []),
            }
            for k, v in manifest.items()
        ]

    def get_dataset(self, dataset_id: str, version: str | None = None) -> list[dict]:
        """获取数据集内容"""
        manifest = self._read_manifest()
        if dataset_id not in manifest:
            raise KeyError(f"Dataset '{dataset_id}' not found")

        version = version or manifest[dataset_id]["latest_version"]
        data_path = self.base_dir / dataset_id / version / "data.jsonl"

        if not data_path.exists():
            raise FileNotFoundError(f"Version '{version}' not found")

        results = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results

    def validate_dataset(self, dataset_path: str | Path) -> ValidationReport:
        """验证数据集文件"""
        with open(dataset_path, encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]

        validator = DatasetValidator()
        return validator.validate(data)

    def delete_dataset(self, dataset_id: str, version: str | None = None):
        """删除数据集"""
        import shutil

        manifest = self._read_manifest()
        if dataset_id not in manifest:
            raise KeyError(f"Dataset '{dataset_id}' not found")

        if version:
            # 删除特定版本
            version_dir = self.base_dir / dataset_id / version
            if version_dir.exists():
                shutil.rmtree(version_dir)
        else:
            # 删除整个数据集
            dataset_dir = self.base_dir / dataset_id
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            del manifest[dataset_id]
            self._write_manifest(manifest)

    def create_version(self, dataset_id: str, data: list[dict], tag: str) -> str:
        """创建新版本"""
        validator = DatasetValidator()
        report = validator.validate(data)
        if not report.is_valid:
            raise ValueError(f"Validation failed: {report.errors}")

        # 保存新版本
        dataset_dir = self.base_dir / dataset_id / tag
        dataset_dir.mkdir(parents=True, exist_ok=True)

        data_path = dataset_dir / "data.jsonl"
        with open(data_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 更新 manifest
        manifest = self._read_manifest()
        manifest[dataset_id]["latest_version"] = tag
        self._write_manifest(manifest)

        return tag
