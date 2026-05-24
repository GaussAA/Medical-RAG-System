"""Generation Evaluation Module using LLM Judge.

Provides generation metrics:
- Faithfulness: How much of the answer can be derived from context
- Answer Relevancy: How well the answer addresses the query
- Context Precision: Quality of context ranking
- Citation Accuracy: Accuracy of citations in the answer
- Hallucination Ratio: Ratio of unverified citations
"""

from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class GenerationMetrics:
    """Generation evaluation metrics."""

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    citation_accuracy: float = 0.0
    hallucination_ratio: float = 0.0


class GenerationEvaluator:
    """Evaluates generation quality using LLM judge."""

    FAITHFULNESS_PROMPT = """You are a medical fact-checker. Evaluate whether each claim in the answer
can be derived from the provided context.

Context:
{context}

Answer:
{answer}

For each claim in the answer, indicate whether it is:
- SUPPORTED: The claim directly follows from the context
- PARTIAL: The claim is partially supported but needs more context
- UNSUPPORTED: The claim cannot be derived from the context

Calculate the faithfulness score as: (SUPPORTED claims + 0.5 * PARTIAL claims) / Total claims

Output your analysis and final score in JSON format:
{{"claims": [{"text": "...", "status": "SUPPORTED|PARTIAL|UNSUPPORTED"}, ...], "faithfulness_score": 0.0-1.0}}
"""

    ANSWER_RELEVANCY_PROMPT = """You are a medical query analyzer. Evaluate whether the answer properly
addresses the user's query.

Query:
{query}

Answer:
{answer}

Generate 3-5 reverse questions that the answer should be able to answer.
Then evaluate if the answer could reasonably answer those questions.

Output in JSON format:
{{"reverse_questions": ["Q1", "Q2", "Q3"], "relevancy_score": 0.0-1.0}}
"""

    def __init__(self, llm_generator: Any | None = None):
        """
        Initialize generation evaluator.

        Args:
            llm_generator: LLM generator instance for LLM-based evaluation.
                          If None, falls back to rule-based evaluation.
        """
        self.llm_generator = llm_generator

    async def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        citations: list[Any] | None = None,
    ) -> GenerationMetrics:
        """
        Evaluate generation quality.

        Args:
            query: The original user query.
            answer: The generated answer.
            contexts: List of context strings used for generation.
            citations: Optional list of Citation objects.

        Returns:
            GenerationMetrics with all computed metrics.
        """
        metrics = GenerationMetrics()

        # Faithfulness evaluation
        if self.llm_generator:
            metrics.faithfulness = await self._evaluate_faithfulness_llm(answer, contexts)
        else:
            metrics.faithfulness = self._evaluate_faithfulness_rule(answer, contexts)

        # Answer Relevancy evaluation
        if self.llm_generator:
            metrics.answer_relevancy = await self._evaluate_relevancy_llm(query, answer)
        else:
            metrics.answer_relevancy = self._evaluate_relevancy_rule(query, answer)

        # Context Precision (based on retrieval order)
        metrics.context_precision = self._evaluate_context_precision(contexts, query)

        # Citation metrics
        if citations:
            metrics.citation_accuracy = self._calculate_citation_accuracy(citations)
            metrics.hallucination_ratio = self._calculate_hallucination_ratio(citations)

        return metrics

    async def _evaluate_faithfulness_llm(self, answer: str, contexts: list[str]) -> float:
        """
        Use LLM to evaluate faithfulness.

        Returns score between 0.0 and 1.0.
        """
        if not self.llm_generator or not contexts:
            return 0.0

        context_combined = "\n\n".join(f"[Context {i+1}]: {ctx}" for i, ctx in enumerate(contexts))

        # Use string replacement to avoid JSON curly braces being interpreted as format placeholders
        prompt = self.FAITHFULNESS_PROMPT
        prompt = prompt.replace("{context}", context_combined).replace("{answer}", answer)

        try:
            result = await self.llm_generator.generate(
                query="Evaluate faithfulness",
                contexts=[],
                conversation_history=[
                    {"role": "user", "content": prompt}
                ],
            )

            # Parse JSON response
            import json

            response_text = result.get("answer", "")
            # Try to extract JSON from response with robust parsing
            try:
                import re
                # Find JSON block - handle both ```json and bare JSON
                json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    # Try to find a complete JSON object
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = response_text[start_idx:end_idx+1]
                        # Clean up common JSON issues (trailing commas, etc.)
                        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                        parsed = json.loads(json_str)
                        score = parsed.get("faithfulness_score", 0.5)
                        return float(min(max(score, 0.0), 1.0))
                return 0.5
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse faithfulness response: {e}")
                return 0.5

        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return 0.0

    def _evaluate_faithfulness_rule(self, answer: str, contexts: list[str]) -> float:
        """
        Rule-based faithfulness evaluation.

        Checks for basic consistency between answer and contexts.
        """
        if not contexts or not answer:
            return 0.0

        context_text = " ".join(contexts).lower()
        answer_lower = answer.lower()

        # Check 1: Key medical terms from contexts appear in answer
        medical_indicators = [
            "mg", "剂量", "药物", "治疗", "诊断", "患者",
            "血压", "血糖", "药物", "服用", "用法", "疗程",
            "禁忌", "不良反应", "适应症", "并发症",
        ]

        matching_indicators = sum(
            1 for ind in medical_indicators
            if ind in context_text and ind in answer_lower
        )

        # Check 2: Citation markers present in answer
        citation_markers = ["来源", "「", "」", "#"]
        has_citations = any(marker in answer for marker in citation_markers)

        # Check 3: Answer has substantive content (not just short)
        has_substantive = len(answer) > 100

        # Check 4: Numeric values (like dosages) appear in both
        import re

        context_nums = set(re.findall(r"\d+\.?\d*\s*(?:mg|ml|ug|IU|次|天|小时)", context_text))
        answer_nums = set(re.findall(r"\d+\.?\d*\s*(?:mg|ml|ug|IU|次|天|小时)", answer_lower))
        numeric_match = len(context_nums & answer_nums) / max(len(context_nums), 1) if context_nums else 0

        # Calculate score
        score = 0.3  # Base score

        if matching_indicators > 0:
            score += min(matching_indicators * 0.1, 0.3)

        if has_citations:
            score += 0.2

        if has_substantive:
            score += 0.1

        score += numeric_match * 0.2

        return min(max(score, 0.0), 1.0)

    async def _evaluate_relevancy_llm(self, query: str, answer: str) -> float:
        """
        Use LLM to evaluate answer relevancy.

        Returns score between 0.0 and 1.0.
        """
        if not self.llm_generator:
            return 0.0

        prompt = self.ANSWER_RELEVANCY_PROMPT
        prompt = prompt.replace("{query}", query).replace("{answer}", answer)

        try:
            result = await self.llm_generator.generate(
                query="Evaluate relevancy",
                contexts=[],
                conversation_history=[
                    {"role": "user", "content": prompt}
                ],
            )

            import json

            response_text = result.get("answer", "")
            try:
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0]
                elif "{" in response_text:
                    json_str = response_text[response_text.index("{"): response_text.rindex("}") + 1]
                else:
                    return 0.5

                parsed = json.loads(json_str)
                score = parsed.get("relevancy_score", 0.5)
                return float(min(max(score, 0.0), 1.0))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse relevancy response: {e}")
                return 0.5

        except Exception as e:
            logger.error(f"Relevancy evaluation failed: {e}")
            return 0.0

    def _evaluate_relevancy_rule(self, query: str, answer: str) -> float:
        """
        Rule-based relevancy evaluation.

        Checks basic query term coverage and structure in answer.
        """
        if not query or not answer:
            return 0.0

        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())

        # Remove common stop words
        stop_words = {"的", "了", "是", "在", "和", "与", "或", "有", "我", "你", "他", "她", "它", "什么", "如何", "怎么", "哪些", "哪个"}
        query_terms = query_terms - stop_words

        if not query_terms:
            return 0.5

        # Calculate coverage
        coverage = len(query_terms & answer_terms) / len(query_terms)

        # Check for question structure: answer should start with direct response
        # (not "根据", "对于", etc.)
        answer_starts_weak = any(answer.strip().startswith(w) for w in ["根据", "对于", "关于", "关于这个问题"])
        structure_bonus = -0.1 if answer_starts_weak else 0.1

        # Check for complete sentences in answer
        has_complete_sentences = answer.count("。") >= 2

        # Combine scores
        score = coverage + 0.3 + structure_bonus
        if has_complete_sentences:
            score += 0.1

        return min(max(score, 0.0), 1.0)

    def _evaluate_context_precision(self, contexts: list[str], query: str) -> float:
        """
        Evaluate context precision based on query-context similarity.

        Uses a simple keyword overlap heuristic.
        """
        if not contexts:
            return 0.0

        query_terms = set(query.lower().split()) - {"的", "了", "是", "在", "和", "与", "或", "有"}

        precision_scores = []
        for ctx in contexts:
            ctx_terms = set(ctx.lower().split())
            if query_terms:
                overlap = len(query_terms & ctx_terms) / len(query_terms)
                precision_scores.append(overlap)

        return sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

    def _calculate_citation_accuracy(self, citations: list[Any]) -> float:
        """
        Calculate citation accuracy based on verification results.

        Args:
            citations: List of Citation objects.

        Returns:
            Ratio of verified citations to total (0-1).
        """
        if not citations:
            return 0.0

        verified_count = sum(1 for c in citations if getattr(c, "verified", False))
        return verified_count / len(citations)

    def _calculate_hallucination_ratio(self, citations: list[Any]) -> float:
        """
        Calculate hallucination ratio (unverified / total citations).

        Args:
            citations: List of Citation objects.

        Returns:
            Hallucination ratio (0-1).
        """
        if not citations:
            return 0.0

        unverified_count = sum(1 for c in citations if not getattr(c, "verified", False))
        return unverified_count / len(citations)

    async def evaluate_batch(
        self,
        queries: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
        citations_list: list[list[Any]] | None = None,
    ) -> list[GenerationMetrics]:
        """
        Evaluate a batch of query-answer pairs.

        Args:
            queries: List of queries.
            answers: List of answers.
            contexts_list: List of context lists.
            citations_list: Optional list of citation lists.

        Returns:
            List of GenerationMetrics for each query.
        """
        results = []
        for i, (query, answer, contexts) in enumerate(zip(queries, answers, contexts_list)):
            citations = citations_list[i] if citations_list else None
            metrics = await self.evaluate(query, answer, contexts, citations)
            results.append(metrics)

        return results