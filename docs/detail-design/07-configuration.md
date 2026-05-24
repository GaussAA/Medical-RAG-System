# Configuration System

## Configuration Architecture

```mermaid
graph LR
    subgraph Files
        ENV[.env file]
        YAML[config/config.yaml]
    end

    subgraph Python
        SETTINGS[Settings Pydantic Model]
        YAML_CONFIG[YAML Config Loader]
    end

    ENV -->|load_dotenv()| SETTINGS
    YAML -->|load_config()| SETTINGS
```

## Settings Hierarchy

**File**: [config/settings.py](../../config/settings.py)

```python
class Settings(BaseModel):
    app: AppConfig
    database: DatabaseConfig
    models: ModelsConfig
    rag: RAGConfig
    safety: SafetyConfig
    cache: CacheConfig
    data: DataConfig
    streamlit: StreamlitConfig
```

## Environment Variable Substitution

```python
# ${VAR_NAME} syntax in YAML
llm:
  api_key: ${DEEPSEEK_API_KEY}

# Resolved at load_config() time
```

**Implementation** ([config/settings.py](../../config/settings.py)):
```python
def _substitute_env_vars(obj: Any) -> Any:
    if isinstance(obj, str):
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, obj)
        for match in matches:
            env_value = os.getenv(match, "")
            obj = obj.replace(f"${{{match}}}", env_value)
        return obj
```

## Database Configuration

```python
class PostgreSQLConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "medical_rag"
    username: str = "postgres_admin"
    password: str = "postgres_123"

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
```

## Model Configuration

### Embedding Model

```python
class EmbeddingConfig(BaseModel):
    name: str = "BAAI/bge-m3"
    device: str = "cuda"      # Config value (actual device is dynamic)
    dimension: int = 1024
    normalize: bool = True
    batch_size: int = 32
    estimated_memory_mb: int = 1536
```

### Reranker Model

```python
class RerankerConfig(BaseModel):
    name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cuda"      # Config value (actual device is dynamic)
    batch_size: int = 8
    max_length: int = 512
    estimated_memory_mb: int = 1843
```

### LLM Model

```python
class LLMConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    api_base: str = "https://api.deepseek.com/v1"
    api_key: str = ""          # From ${DEEPSEEK_API_KEY}
    temperature: float = 0.3
    max_tokens: int = 2000
    top_p: float = 0.9
```

## RAG Configuration

### Chunking

```python
class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    strategy: str = "semantic"
    separator: list[str] = ["\n\n", "\n", "。", "！", "？"]
    preserve_tables: bool = True
    min_chunk_length: int = 50
    max_chunk_length: int = 1000
```

### Retrieval

```python
class RetrievalConfig(BaseModel):
    vector_top_k: int = 50
    bm25_top_k: int = 50
    bm25_persist_path: str | None = None  # Default: data/cache/bm25_index.json
    fusion_method: str = "rrf"
    rrf_k: int = 60
    weights: dict[str, float] = {"vector": 0.6, "bm25": 0.4}
    final_top_k: int = 5
    similarity_threshold: float = 0.5
```

### Generation

```python
class GenerationConfig(BaseModel):
    include_citations: bool = True
    include_confidence: bool = True
    include_warnings: bool = True
    max_context_tokens: int = 4000
```

## Safety Configuration

```python
class SafetyConfig(BaseModel):
    enable: bool = True
    sensitive_words_check: bool = True
    privacy_protection: bool = True
    political_check: bool = True
    adult_content_check: bool = True
    sensitive_patterns: list[SensitivePattern] = []
```

## Data Directories

```python
class DataConfig(BaseModel):
    raw_documents: str = "data/raw_documents"
    processed: str = "data/processed"
    cache: str = "data/cache"
```

## YAML Config Example

**File**: `config/config.yaml`

```yaml
app:
  name: "医疗知识库RAG系统"
  version: "1.0.0"
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"
  debug: false

database:
  postgresql:
    host: "localhost"
    port: 5432
    database: "medical_rag"
    username: "postgres_admin"
    password: "postgres_123"
  qdrant:
    host: "localhost"
    port: 6333
    collection: "medical_knowledge"
    vector_size: 1024
  redis:
    host: "localhost"
    port: 6379

models:
  embedding:
    name: "BAAI/bge-m3"
    device: "cpu"       # Note: actual device is dynamic
    dimension: 1024
    batch_size: 32
  reranker:
    name: "BAAI/bge-reranker-v2-m3"
    device: "cpu"       # Note: actual device is dynamic
    batch_size: 8
    max_length: 512
  llm:
    provider: "deepseek"
    model: "deepseek-chat"
    api_base: "https://api.deepseek.com/v1"
    api_key: "${DEEPSEEK_API_KEY}"

rag:
  chunking:
    chunk_size: 512
    chunk_overlap: 50
    strategy: "semantic"
  retrieval:
    vector_top_k: 50
    bm25_top_k: 50
    rrf_k: 60
    weights:
      vector: 0.6
      bm25: 0.4
    final_top_k: 5
  generation:
    include_citations: true
    include_confidence: true
    include_warnings: true

safety:
  enable: true
  sensitive_words_check: true
  privacy_protection: true

data:
  raw_documents: "data/raw_documents"
  processed: "data/processed"
  cache: "data/cache"
```

## Device Selection Note

**Important**: The `config.yaml` shows `device: "cpu"` for models, but actual device selection is dynamic and handled by [GPUMemoryManager](../detail-design/08-gpu-memory-management.md):

1. Document processing: embedding tries GPU first
2. Query processing: reranker tries GPU first, falls back to CPU if memory insufficient

## Loading Configuration

```python
from config.settings import get_settings, load_config, reload_settings

# Get cached settings (recommended)
settings = get_settings()

# Reload from file (for testing or config refresh)
settings = reload_settings()

# Load specific config file
settings = load_config("config/custom.yaml")
```
