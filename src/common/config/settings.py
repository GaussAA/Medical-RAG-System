"""Application configuration - migrated from config/settings.py"""

import os
import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file at module import
load_dotenv()


class AppConfig(BaseModel):
    name: str = "医疗知识库RAG系统"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


class PostgreSQLConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "medical_rag"
    username: str = ""
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def sync_url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection: str = "medical_knowledge"
    timeout: int = 10  # connection timeout in seconds
    prefer_grpc: bool = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class RedisConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    password: str | None = None

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class DatabaseConfig(BaseModel):
    postgresql: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)


class EmbeddingConfig(BaseModel):
    name: str = "BAAI/bge-m3"
    device: str = "cuda"
    dimension: int = 1024  # bge-m3 dense vector dimension


class RerankerConfig(BaseModel):
    name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"
    batch_size: int = 8
    max_length: int = 512


class LLMConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    api_base: str = "https://api.deepseek.com"
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 4000
    top_p: float = 0.9


class ModelsConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


class ChunkingConfig(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 80
    strategy: str = "hierarchical"
    separator: list[str] = ["\n\n", "\n", "。", "！", "？"]
    preserve_tables: bool = True
    min_chunk_length: int = 100
    max_chunk_length: int = 2000


class RetrievalConfig(BaseModel):
    vector_top_k: int = 50
    bm25_top_k: int = 50
    bm25_persist_path: str | None = "data/cache/bm25_index.json"
    fusion_method: str = "rrf"
    rrf_k: int = 60
    weights: dict[str, float] = {"vector": 0.6, "bm25": 0.4}
    final_top_k: int = 5
    similarity_threshold: float = 0.5
    boost_factor: float = 1.3


class CitationVerificationConfig(BaseModel):
    enable: bool = True
    hallucination_threshold: float = 0.3
    warn_on_hallucination: bool = True


class GenerationConfig(BaseModel):
    include_citations: bool = True
    include_confidence: bool = True
    include_warnings: bool = True
    max_context_tokens: int = 4000
    citation_verification: CitationVerificationConfig = Field(default_factory=CitationVerificationConfig)


class RAGConfig(BaseModel):
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


class SensitivePattern(BaseModel):
    name: str
    pattern: str
    replacement: str


class SafetyConfig(BaseModel):
    enable: bool = True
    sensitive_words_check: bool = True
    privacy_protection: bool = True
    sensitive_patterns: list[SensitivePattern] = []


class CorsConfig(BaseModel):
    allow_origins: list[str] = ["http://localhost:8501", "http://localhost:3000"]
    allow_credentials: bool = True
    allow_methods: list[str] = ["GET", "POST", "PUT", "DELETE"]
    allow_headers: list[str] = [
        "Authorization",
        "Content-Type",
        "X-Request-ID",
        "X-Trace-ID",
    ]


class StreamlitConfig(BaseModel):
    page_title: str = "医疗知识库问答系统"
    page_icon: str = "🏥"
    initial_sidebar_state: str = "expanded"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file_enabled: bool = True
    file_path: str = "data/logs/medical-rag-{time:YYYY-MM-DD}.log"
    file_rotation: str = "1 day"
    file_retention: str = "30 days"
    file_format: str = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {name}:{function}:{line} | {message}"
    console_level: str = "DEBUG"
    console_format: str = (
        "<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    cors: CorsConfig = Field(default_factory=CorsConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute ${ENV_VAR} or ${ENV_VAR:-default} patterns."""
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        pattern = r"\$\{([^}:-]+)(?::-([^}]*))?\}"
        matches = re.findall(pattern, obj)
        for match in matches:
            env_name, default_value = match
            env_value = os.getenv(env_name, None)
            if env_value is None or env_value == "":
                env_value = default_value if default_value is not None else ""
            if default_value:
                replace_pattern = f"${{{env_name}:-{default_value}}}"
            else:
                replace_pattern = f"${{{env_name}}}"
            obj = obj.replace(replace_pattern, env_value)
        return obj
    return obj


def load_config(config_path: str | None = None) -> Settings:
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "src/common/config/config.yaml")

    config_file = Path(config_path)
    if not config_file.exists():
        return Settings()

    with open(config_file, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    config_data = _substitute_env_vars(config_data)
    return Settings(**config_data)


_settings: Settings | None = None
_settings_observers: list = []


def _notify_settings_changed() -> None:
    for callback in _settings_observers:
        try:
            callback(_settings)
        except Exception:
            pass


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_config()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = load_config()
    _notify_settings_changed()
    return _settings


def add_settings_observer(callback) -> None:
    _settings_observers.append(callback)


def remove_settings_observer(callback) -> None:
    if callback in _settings_observers:
        _settings_observers.remove(callback)
