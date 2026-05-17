import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file at module import
load_dotenv()


class AppConfig(BaseModel):
    name: str = "医疗知识库RAG系统"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    debug: bool = False


class PostgreSQLConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "medical_rag"
    username: str = "postgres_admin"
    password: str = "postgres_123"
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def sync_url(self) -> str:
        return (
            f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        )


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection: str = "medical_knowledge"
    vector_size: int = 1024
    distance: str = "Cosine"
    timeout: int = 10
    prefer_grpc: bool = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class RedisConfig(BaseModel):
    host: str = "localhost"
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
    dimension: int = 1024
    normalize: bool = True
    batch_size: int = 32
    estimated_memory_mb: int = 1536


class RerankerConfig(BaseModel):
    name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"
    batch_size: int = 8
    max_length: int = 512
    estimated_memory_mb: int = 1843


class LLMConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_base: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 2000
    top_p: float = 0.9


class ModelsConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    strategy: str = "semantic"
    separator: list[str] = ["\n\n", "\n", "。", "！", "？"]
    preserve_tables: bool = True
    min_chunk_length: int = 50
    max_chunk_length: int = 1000


class RetrievalConfig(BaseModel):
    vector_top_k: int = 50
    bm25_top_k: int = 50
    bm25_persist_path: str | None = None
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
    citation_verification: CitationVerificationConfig = Field(
        default_factory=CitationVerificationConfig
    )


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
    political_check: bool = True
    adult_content_check: bool = True
    sensitive_patterns: list[SensitivePattern] = []


class CacheConfig(BaseModel):
    session_context: dict[str, Any] = {"ttl": 3600, "max_size": 10000}
    retrieval_cache: dict[str, Any] = {"ttl": 300}
    llm_response: dict[str, Any] = {"ttl": 1800}


class DataConfig(BaseModel):
    raw_documents: str = "data/raw_documents"
    processed: str = "data/processed"
    cache: str = "data/cache"


class CorsConfig(BaseModel):
    allow_origins: list[str] = ["http://localhost:8501", "http://localhost:3000"]
    allow_credentials: bool = True
    allow_methods: list[str] = ["GET", "POST", "PUT", "DELETE"]
    allow_headers: list[str] = ["*"]


class StreamlitConfig(BaseModel):
    page_title: str = "医疗知识库问答系统"
    page_icon: str = "🏥"
    initial_sidebar_state: str = "expanded"


class EvaluationConfig(BaseModel):
    enable: bool = True
    sample_rate: float = 1.0
    k_values: list[int] = [5, 10, 20]
    output_dir: str = "data/evaluation/reports"
    llm_judge_provider: str = "deepseek"
    llm_judge_model: str = "deepseek-chat"
    faithfulness_threshold: float = 0.8
    relevancy_threshold: float = 0.7


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    cors: CorsConfig = Field(default_factory=CorsConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute ${ENV_VAR} or ${ENV_VAR:-default} patterns.

    Supports two syntaxes:
    - ${ENV_VAR} - substitutes with environment variable value, empty string if not found
    - ${ENV_VAR:-default} - substitutes with default value if env var is not set or empty
    """
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        import re

        # Pattern supports: ${VAR} and ${VAR:-default}
        pattern = r"\$\{([^}:-]+)(?::-([^}]*))?\}"
        matches = re.findall(pattern, obj)
        for match in matches:
            env_name, default_value = match
            env_value = os.getenv(env_name, None)
            if env_value is None or env_value == "":
                env_value = default_value if default_value is not None else ""
            # Only use :-default syntax when default_value has actual content
            if default_value:
                replace_pattern = f"${{{env_name}:-{default_value}}}"
            else:
                replace_pattern = f"${{{env_name}}}"
            obj = obj.replace(replace_pattern, env_value)
        return obj
    return obj


def load_config(config_path: str | None = None) -> Settings:
    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config/config.yaml")

    config_file = Path(config_path)
    if not config_file.exists():
        return Settings()

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Substitute environment variables
    config_data = _substitute_env_vars(config_data)

    return Settings(**config_data)


_settings: Settings | None = None
_settings_observers: list = []

def _notify_settings_changed() -> None:
    """Notify all observers that settings have been reloaded."""
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
    """Reload settings from config file and notify observers."""
    global _settings
    _settings = load_config()
    _notify_settings_changed()
    return _settings


def add_settings_observer(callback) -> None:
    """Add a callback to be called when settings are reloaded."""
    _settings_observers.append(callback)


def remove_settings_observer(callback) -> None:
    """Remove a settings observer callback."""
    if callback in _settings_observers:
        _settings_observers.remove(callback)
