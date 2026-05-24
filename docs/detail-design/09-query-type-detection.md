# Query Type Detection and Content-Type Boosting

## Overview

During retrieval, queries are analyzed to detect content-type intent. When a query type is detected, results matching that content_type are boosted to the top of the result list.

## Detection Logic

**File**: [rag/retrieval/hybrid_retriever.py](../../rag/retrieval/hybrid_retriever.py) § `_detect_query_type()`

### Query Type Patterns

| Type | Patterns | Example Queries |
|------|----------|-----------------|
| `table` | `表[一二三四五六七八九十\d]+`, `表格`, `table` | "表1的数据", "表10的诊断标准", "表格中的结果" |
| `list` | `列出`, `列表中`, `列表项`, `哪些.*列表`, `list` | "列出所有药物", "列表中的项目" |
| `list` (drug) | `剂量`, `用法`, `每次`, `每日`, `mg`, `毫升`, `不良反应`, `禁忌`, `药物` | "每次剂量是多少", "每日服用几次", "药物不良反应" |

### Implementation

```python
def _detect_query_type(self, query: str) -> str | None:
    query_lower = query.lower()

    # Check for table-related queries
    for pattern in self.table_query_patterns:
        if re.search(pattern, query_lower):
            return "table"

    # Check for list-related queries
    for pattern in self.list_query_patterns:
        if re.search(pattern, query_lower):
            return "list"

    # Check for drug-related queries (treated as list content)
    for pattern in self.drug_query_patterns:
        if re.search(pattern, query_lower):
            return "list"  # Treat drug info as list content

    return None
```

## Content-Type Boosting

**File**: [rag/retrieval/hybrid_retriever.py](../../rag/retrieval/hybrid_retriever.py) § `_boost_by_content_type()`

After RRF fusion, results are reordered based on detected query type:

```python
def _boost_by_content_type(
    self, results: list[RetrievedNode], target_type: str
) -> list[RetrievedNode]:
    boosted = []
    other = []

    for node in results:
        content_type = node.metadata.get("content_type", "text")
        if content_type == target_type:
            boosted.append(node)
        else:
            other.append(node)

    # Interleave boosted results with original ordering
    boosted.extend(other)
    return boosted
```

## Content Types

Chunks are tagged with one of three content types:

| Type | Detection Rule | Example |
|------|---------------|---------|
| `text` | Default | Regular paragraphs |
| `table` | Markdown table syntax (`\| col1 \| col2 \|`) or table caption (`**表X**`) | Tabular data |
| `list` | Bullet markers (`·`, `-`) or numbered markers (`（1）`) | Enumerated items |

## Integration with Hybrid Retrieval

```
Query → _detect_query_type() → _parallel_search() → _reciprocal_rank_fusion() → _boost_by_content_type() → Reranker → LLM
```

1. Query is analyzed for content type intent
2. Vector and BM25 searches run in parallel (`asyncio.gather`)
3. RRF fusion combines results
4. Content-type boosting reorders results
5. Reranker further refines the results

## Configuration

Query-type detection is always enabled (no configuration flag). The patterns are hardcoded in `HybridRetriever.__init__()`:

```python
self.table_query_patterns = [
    r"表[一二三四五六七八九十\d]+",
    r"表格",
    r"table",
]
self.list_query_patterns = [
    r"列出",
    r"列表中",
    r"列表项",
    r"哪些.*列表",
    r"list",
]
self.drug_query_patterns = [
    r"剂量",
    r"用法",
    r"每次",
    r"每日",
    r"mg",
    r"毫升",
    r"不良反应",
    r"禁忌",
    r"药物",
]
```