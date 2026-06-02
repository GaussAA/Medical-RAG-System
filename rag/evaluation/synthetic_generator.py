"""Synthetic data generator using LLM."""

import json
from typing import Any

from rag.evaluation.evaluator import EvalGroundTruth


class SyntheticDataGenerator:
    """使用 LLM 生成合成评估数据"""

    SYNTHESIS_PROMPT = """你是一位医学专家。请基于以下文档内容，生成一个合理的医学问题及其正确答案。

文档内容：
{chunk_content}

要求：
1. 问题应该简洁、明确，适合作为 RAG 系统评估查询
2. 提供问题的类型标签（drug/diagnosis/treatment/prevention等）
3. 提供正确答案中应包含的关键医学实体
4. 确保生成的问题与文档内容相关

请以 JSON 格式输出：
{{
  "query_text": "...",
  "query_type": "...",
  "reference_answer": "...",
  "expected_keywords": ["..."],
  "safety_sensitive": true/false
}}
"""

    def __init__(self, llm_generator: Any | None = None):
        """
        初始化生成器

        Args:
            llm_generator: 可选的 LLM 生成器实例
        """
        self.llm = llm_generator

    async def generate(
        self,
        source_docs: list[dict],
        count: int = 50,
        query_types: list[str] | None = None,
    ) -> list[EvalGroundTruth]:
        """
        从源文档生成合成 QA 对

        Args:
            source_docs: 源文档列表，每个包含 chunk_content
            count: 生成数量
            query_types: 指定查询类型

        Returns:
            生成的 EvalGroundTruth 列表
        """
        results = []
        docs_per_query = max(1, len(source_docs) // count)

        for i in range(count):
            # 选择文档片段
            start_idx = (i * docs_per_query) % len(source_docs)
            doc = source_docs[start_idx]
            chunk_content = doc.get("chunk_content", "")

            if not chunk_content:
                continue

            # 生成 QA
            generated = await self._generate_single(chunk_content, query_types)
            if generated:
                results.append(generated)

        return results

    async def _generate_single(
        self,
        chunk_content: str,
        query_types: list[str] | None,
    ) -> EvalGroundTruth | None:
        """生成单条 QA"""
        if self.llm is None:
            # 规则模式：简单生成
            return self._generate_rule_based(chunk_content, query_types)

        # LLM 模式
        try:
            prompt = self.SYNTHESIS_PROMPT.format(chunk_content=chunk_content[:1000])
            response = await self.llm.generate(prompt)

            data = json.loads(response)
            return EvalGroundTruth(
                query_id=f"synthetic_{hash(chunk_content[:20]) & 0xFFFFFFFF}",
                expected_keywords=data.get("expected_keywords", []),
                reference_answer=data.get("reference_answer"),
                query_type=data.get("query_type", "general"),
                safety_sensitive=data.get("safety_sensitive", False),
            )
        except Exception:
            return self._generate_rule_based(chunk_content, query_types)

    def _generate_rule_based(
        self,
        chunk_content: str,
        query_types: list[str] | None,
    ) -> EvalGroundTruth:
        """基于规则的简单生成"""
        # 简单实现：从内容中提取关键词生成问题
        words = chunk_content.split()
        _key_terms = [w for w in words if len(w) > 4][:5]

        return EvalGroundTruth(
            query_id=f"rule_{hash(chunk_content[:20]) & 0xFFFFFFFF}",
            query_type=query_types[0] if query_types else "general",
            safety_sensitive=False,
        )
