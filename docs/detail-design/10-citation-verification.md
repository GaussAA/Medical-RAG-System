# Citation Verification and Hallucination Detection

## Overview

The system extracts and verifies citations from LLM-generated answers to detect potential hallucinations.

## Citation Verification Flow

**File**: [app/services/citation_verifier.py](../../app/services/citation_verifier.py)

### Extraction

Citations are extracted from the answer text using patterns:

| Pattern | Format | Example |
|---------|--------|---------|
| Old | `[来源X](文件名#页码)` | `[来源1](指南.md#1)` |
| New | `「来源X」（文件名#页码）` | `「来源1」（指南.md#1）` |

### Verification Logic

The `CitationVerifier.extract_and_verify()` method:

1. Extracts citation markers from the answer text
2. For each citation, checks if the cited file matches any retrieval context
3. Sets `verified=True` if match found, `verified=False` with `verification_message` if not

```python
def extract_and_verify(answer: str, contexts: list[RetrievedNode]) -> list[Citation]:
    # Extract markers from answer
    # For each marker:
    #   - Get cited file from marker
    #   - Check if any context has matching source_file
    #   - Set verified=True/False
```

## Hallucination Detection

**File**: [app/core/rag_engine.py](../../app/core/rag_engine.py) § `_generate_warnings()`

Hallucination detection occurs during warning generation:

```python
if citations and self.config.generation.citation_verification.enable:
    unverified_citations = [c for c in citations if not getattr(c, 'verified', False)]
    total_citations = len(citations)
    if total_citations > 0:
        unverified_ratio = len(unverified_citations) / total_citations
        threshold = self.config.generation.citation_verification.hallucination_threshold
        if unverified_ratio > threshold:
            warnings.append(
                RiskWarning(
                    type="hallucination",
                    message=f"检测到 {len(unverified_citations)}/{total_citations} 条引用来源无法验证，AI可能存在幻觉。",
                    priority="high",
                )
            )
```

## CitationPosition Enum

**File**: [app/models/schemas.py](../../app/models/schemas.py)

```python
class CitationPosition(str, Enum):
    DIRECT = "direct"        # Direct quote from source
    INDIRECT = "indirect"    # Paraphrased from source
    PARAPHRASED = "paraphrased"  # Heavily reworded
    UNVERIFIED = "unverified"  # Cannot verify against source
```

**Current Implementation**: All citations are created with `position=CitationPosition.DIRECT` and `verified=True`. The position field is not actually set based on analysis.

## Citation Schema

```python
class Citation(BaseModel):
    source_id: str              # 1-indexed index in contexts
    document_id: str | None     # Document UUID
    file_name: str              # Source file name
    page_number: int | None     # Page/section reference
    chunk_content: str = ""    # First 200 chars of cited chunk
    relevance_score: float = 0.0  # Retrieval relevance score
    position: CitationPosition = CitationPosition.DIRECT
    verified: bool = False      # Verification status
    quote_in_answer: str | None  # Exact quote in answer
    verification_message: str | None  # Verification result
```

## Configuration

**File**: [config/settings.py](../../config/settings.py)

```python
class CitationVerificationConfig(BaseModel):
    enable: bool = True
    hallucination_threshold: float = 0.5  # 50% unverified triggers warning
    warn_on_hallucination: bool = True
```

## Warning Types

The RAGEngine generates these warning types:

| Type | Priority | Trigger |
|------|---------|---------|
| `general` | low | Always added |
| `medication` | medium | Medication keywords detected |
| `diagnosis` | high | Diagnosis keywords detected |
| `emergency` | high | Emergency keywords detected |
| `hallucination` | high | Unverified citation ratio > threshold |

## Keywords for Warning Generation

**File**: [app/core/rag_engine.py](../../app/core/rag_engine.py)

```python
medication_keywords = ["药物", "用药", "剂量", "服药", "吃药"]
diagnosis_keywords = ["诊断", "确诊", "治疗方案"]
emergency_keywords = ["紧急", "急诊", "立即", "马上"]
```

**Note**: `app/core/risk_warnings.py` contains a `RiskWarningGenerator` class but it is NOT used by RAGEngine. The warnings are generated inline in `_generate_warnings()`.