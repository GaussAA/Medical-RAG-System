import asyncio
import time
from collections.abc import AsyncGenerator
from contextvars import ContextVar
from typing import Any

from loguru import logger

from app.core.confidence import ConfidenceEvaluator
from app.core.metrics import (
    ACTIVE_QUERIES,
    ERROR_COUNT,
    GENERATION_LATENCY,
    LLM_TOKENS,
    QUERY_LATENCY,
    RERANK_LATENCY,
    RETRIEVAL_COUNT,
    RETRIEVAL_LATENCY,
)
from app.core.safety import SafetyChecker
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    RetrievedNode,
    RiskWarning,
)
from app.services.citation_verifier import CitationVerifier
from app.services.session import SessionManager
from config.settings import get_settings
from rag.generation.llm_generator import LLMGenerator
from rag.generation.prompt import FALLBACK_RESPONSES
from rag.generation.warnings_generator import WarningsGenerator
from rag.reranker.cross_encoder import Reranker
from rag.retrieval.hybrid_retriever import HybridRetriever

# Thread-safe context variable for trace_id
_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)


class RAGEngine:
    def __init__(self):
        settings = get_settings()
        self.config = settings.rag

        self.safety_checker = SafetyChecker()
        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()
        self.llm_generator = LLMGenerator()
        self.confidence_evaluator = ConfidenceEvaluator()
        self.warnings_generator = WarningsGenerator(
            hallucination_threshold=self.config.generation.citation_verification.hallucination_threshold
        )

    async def process_document(self, nodes: list[RetrievedNode]) -> bool:
        """处理文档向量化。Embedding 模型常驻 GPU（FP16），直接编码。"""
        try:
            await self.hybrid_retriever.add_documents(nodes)
            logger.info(f"Document processing completed: {len(nodes)} chunks vectorized")
            return True
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return False

    def _build_conversation_history(self, session) -> list[dict[str, Any]] | None:
        """Build conversation history list from session messages."""
        if not session:
            return None
        return [{"role": msg.role, "content": msg.content} for msg in session.messages]

    async def query(
        self,
        request: QueryRequest,
        session_manager: SessionManager | None = None,
        trace_id: str | None = None,
    ) -> QueryResponse:
        start_time = time.time()
        ACTIVE_QUERIES.inc()

        # Use context variable for thread-safe trace_id storage
        token = _trace_id_var.set(trace_id)

        try:
            safety_result = self._safety_check(request)
            if not safety_result.passed:
                ERROR_COUNT.labels(error_type="safety").inc()
                return self._create_error_response(
                    request.session_id or "",
                    "包含敏感内容",
                    time.time() - start_time,
                )

            sanitized_query = safety_result.sanitized_text

            # Build conversation context if session_manager provided
            conversation_history: list[dict[str, Any]] | None = None
            if session_manager and request.session_id:
                session = await session_manager.get_or_load_session(request.session_id)
                conversation_history = self._build_conversation_history(session)

            try:
                reranked_nodes = await self._retrieve_and_rerank(
                    sanitized_query, request.filters, conversation_history
                )
            except Exception as e:
                logger.error(f"Retrieval/rerank error: {e}")
                ERROR_COUNT.labels(error_type="retrieval").inc()
                return self._create_error_response(
                    request.session_id or "",
                    f"检索失败：{str(e)}",
                    time.time() - start_time,
                )

            if not reranked_nodes:
                RETRIEVAL_COUNT.labels(retriever_type="none").inc()
                return self._create_fallback_response(
                    request.session_id or "",
                    "no_results",
                    time.time() - start_time,
                )

            try:
                llm_result = await self._generate_answer(sanitized_query, reranked_nodes, conversation_history)
            except Exception as e:
                logger.error(f"Generation error: {e}")
                ERROR_COUNT.labels(error_type="generation").inc()
                return self._create_error_response(
                    request.session_id or "",
                    f"生成回答失败：{str(e)}",
                    time.time() - start_time,
                )

            if "usage" in llm_result:
                usage = llm_result["usage"]
                LLM_TOKENS.labels(token_type="prompt").inc(usage.get("prompt_tokens", 0))
                LLM_TOKENS.labels(token_type="completion").inc(usage.get("completion_tokens", 0))
            confidence_result = self._evaluate_confidence(reranked_nodes, llm_result["answer"], sanitized_query)

            # Extract and verify citations from the answer
            citations = []
            if self.config.generation.include_citations:
                citation_verifier = CitationVerifier()
                citations = citation_verifier.extract_and_verify(
                    answer=llm_result["answer"],
                    contexts=reranked_nodes,
                )
                llm_result["citations"] = citations

            warnings = []
            if self.config.generation.include_warnings:
                warnings = self.warnings_generator.generate(llm_result["answer"], reranked_nodes, citations)

            processing_time = time.time() - start_time
            QUERY_LATENCY.observe(processing_time)

            # Add messages to session AFTER generation succeeds
            if session_manager and request.session_id:
                await session_manager.add_message(request.session_id, "user", request.question)
                await session_manager.add_message(
                    request.session_id,
                    "assistant",
                    llm_result["answer"],
                    metadata={
                        "confidence": confidence_result.get("confidence"),
                        "citations": [
                            c.model_dump() if hasattr(c, "model_dump") else c for c in llm_result.get("citations", [])
                        ],
                        "warnings": [w.model_dump() if hasattr(w, "model_dump") else w for w in warnings],
                    },
                )

            # Include retrieved node info in metadata for evaluation
            # Use doc_id (UUID matching PostgreSQL) for document-level comparison
            node_ids = [n.node_id for n in reranked_nodes]
            doc_ids = list(dict.fromkeys(n.metadata.get("doc_id", "") for n in reranked_nodes))
            node_contents = [n.content[:1000] for n in reranked_nodes[:5]]

            return QueryResponse(
                answer=llm_result["answer"],
                confidence=confidence_result["confidence"],
                citations=llm_result.get("citations", []),
                warnings=warnings,
                session_id=request.session_id or "",
                processing_time=round(processing_time, 2),
                metadata={
                    "retrieved_chunks": len(reranked_nodes),
                    "retrieved_node_ids": node_ids,
                    "retrieved_contents": node_contents,
                    "retrieved_doc_ids": doc_ids,
                    "context_relevance": confidence_result["context_relevance"],
                    "answer_completeness": confidence_result["answer_completeness"],
                    "tokens_used": llm_result.get("usage", {}),
                    "trace_id": _trace_id_var.get(),
                },
                trace_id=_trace_id_var.get(),
            )

        except Exception as e:
            logger.error(f"Unexpected RAG engine error: {e}")
            ERROR_COUNT.labels(error_type="validation").inc()
            return self._create_error_response(
                request.session_id or "",
                str(e),
                time.time() - start_time,
            )
        finally:
            _trace_id_var.reset(token)
            ACTIVE_QUERIES.dec()

    async def query_stream(
        self,
        request: QueryRequest,
        session_manager: SessionManager | None = None,
        trace_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Streaming version of query().
        Yields events: metadata -> chunks -> done/error.
        """
        start_time = time.time()
        ACTIVE_QUERIES.inc()
        token = _trace_id_var.set(trace_id)

        try:
            # Stage 1: Safety check
            t0 = time.time()
            safety_result = self._safety_check(request)
            safety_time = time.time() - t0
            logger.info(f"[{trace_id}] Stage 1 (safety_check): {safety_time:.3f}s")
            if not safety_result.passed:
                ERROR_COUNT.labels(error_type="safety").inc()
                yield {
                    "type": "error",
                    "data": {
                        "message": "包含敏感内容",
                        "code": "SAFETY_ERROR",
                    },
                }
                return

            sanitized_query = safety_result.sanitized_text

            # Build conversation context (BEFORE retrieval, so retrieval can use history)
            conversation_history: list[dict[str, Any]] | None = None
            if session_manager and request.session_id:
                session = await session_manager.get_or_load_session(request.session_id)
                conversation_history = self._build_conversation_history(session)

            # Stage 2: Retrieval & Rerank (with expanded query from history)
            t0 = time.time()
            try:
                reranked_nodes = await self._retrieve_and_rerank(
                    sanitized_query, request.filters, conversation_history
                )
            except Exception as e:
                logger.error(f"Retrieval/rerank error: {e}")
                ERROR_COUNT.labels(error_type="retrieval").inc()
                yield {
                    "type": "error",
                    "data": {
                        "message": f"检索失败：{str(e)}",
                        "code": "RETRIEVAL_ERROR",
                    },
                }
                return
            retrieval_time = time.time() - t0
            node_count = len(reranked_nodes) if reranked_nodes else 0
            logger.info(f"[{trace_id}] Stage 2 (retrieval+rerank): {retrieval_time:.3f}s, nodes={node_count}")

            if not reranked_nodes:
                RETRIEVAL_COUNT.labels(retriever_type="none").inc()
                yield {
                    "type": "done",
                    "data": self._create_fallback_response(
                        request.session_id or "", "no_results", time.time() - start_time
                    ).model_dump(),
                }
                return

            # Pre-stream: yield placeholder metadata
            yield {
                "type": "metadata",
                "data": {
                    "session_id": request.session_id or "",
                    "trace_id": trace_id or _trace_id_var.get(),
                },
            }

            # Stream generation
            stage_start = time.time()
            full_answer = ""
            try:
                async for chunk in self.llm_generator.generate_stream(
                    query=sanitized_query,
                    contexts=reranked_nodes,
                    conversation_history=conversation_history,
                ):
                    full_answer += chunk
                    yield {"type": "chunk", "data": {"content": chunk}}
            except Exception as e:
                logger.error(f"Generation stream error: {e}")
                ERROR_COUNT.labels(error_type="generation").inc()
                yield {
                    "type": "error",
                    "data": {
                        "message": f"生成失败：{str(e)}",
                        "code": "GENERATION_ERROR",
                    },
                }
                return
            generation_time = time.time() - stage_start
            logger.info(f"[{trace_id}] Stage 3 (LLM streaming): {generation_time:.3f}s, answer_len={len(full_answer)}")

            processing_time = time.time() - start_time

            # Stage 4-6: Parallel post-processing (confidence, citations, warnings)
            stage_start = time.time()

            async def eval_confidence():
                return self._evaluate_confidence(reranked_nodes, full_answer, sanitized_query)

            async def extract_citations():
                if not self.config.generation.include_citations:
                    return []
                verifier = CitationVerifier()
                return verifier.extract_and_verify(answer=full_answer, contexts=reranked_nodes)

            async def gen_warnings():
                if not self.config.generation.include_warnings:
                    return []
                return self.warnings_generator.generate(full_answer, reranked_nodes, citations)

            citations, confidence_result = await asyncio.gather(
                extract_citations(),
                eval_confidence(),
            )
            warnings = await gen_warnings()

            postprocess_time = time.time() - stage_start
            logger.info(f"[{trace_id}] Stage 4-6 (postprocessing): {postprocess_time:.3f}s")

            # Stage 7: Persist messages
            stage_start = time.time()
            if session_manager and request.session_id:
                await session_manager.add_message(request.session_id, "user", request.question)
                await session_manager.add_message(
                    request.session_id,
                    "assistant",
                    full_answer,
                    metadata={
                        "confidence": confidence_result.get("confidence"),
                        "citations": [c.model_dump() if hasattr(c, "model_dump") else c for c in citations],
                        "warnings": [w.model_dump() if hasattr(w, "model_dump") else w for w in warnings],
                    },
                )
            persist_time = time.time() - stage_start
            logger.info(f"[{trace_id}] Stage 7 (message_persist): {persist_time:.3f}s")

            QUERY_LATENCY.observe(processing_time)

            yield {
                "type": "done",
                "data": {
                    "confidence": confidence_result["confidence"],
                    "warnings": [w.model_dump() if hasattr(w, "model_dump") else w for w in warnings],
                    "citations": [c.model_dump() if hasattr(c, "model_dump") else c for c in citations],
                    "processing_time": round(processing_time, 2),
                    "metadata": {
                        "retrieved_chunks": len(reranked_nodes),
                        "context_relevance": confidence_result.get("context_relevance"),
                        "answer_completeness": confidence_result.get("answer_completeness"),
                    },
                },
            }

        except Exception as e:
            logger.error(f"Unexpected streaming error: {e}")
            ERROR_COUNT.labels(error_type="validation").inc()
            yield {
                "type": "error",
                "data": {
                    "message": str(e),
                    "code": "INTERNAL_ERROR",
                },
            }
        finally:
            _trace_id_var.reset(token)
            ACTIVE_QUERIES.dec()

    def _safety_check(self, request: QueryRequest) -> Any:
        """Perform safety check on the query."""
        return self.safety_checker.check(request.question)

    async def _retrieve_and_rerank(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> list[RetrievedNode]:
        """Retrieve and rerank documents. Both models (embedding + reranker)
        live on GPU permanently in FP16, no swapping needed.

        When conversation_history is provided, the query is expanded with the last
        user message to resolve pronouns and maintain conversational context.
        """
        # Expand query with conversation context for better retrieval
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg["role"] == "user":
                    query = f"{msg['content']} {query}"
                    logger.debug(f"[_retrieve_and_rerank] Expanded query with history: {query[:120]}...")
                    break

        # Phase 1: Vector + BM25 retrieval (embedding on GPU)
        t0 = time.time()
        retrieved_nodes = await self.hybrid_retriever.search(
            query=query,
            top_k=self.config.retrieval.final_top_k * 2,
            filters=filters,
        )
        retrieval_elapsed = time.time() - t0
        logger.info(
            f"  [_retrieve_and_rerank] vector+bm25 search: {retrieval_elapsed:.3f}s, "
            f"got {len(retrieved_nodes)} nodes"
        )

        if not retrieved_nodes:
            return []

        RETRIEVAL_COUNT.labels(retriever_type="hybrid").inc()

        # Phase 2: Rerank (reranker on GPU)
        t1 = time.time()
        reranked_nodes = self.reranker.rerank(
            query=query,
            candidates=retrieved_nodes[: self.config.retrieval.final_top_k * 2],
        )
        rerank_elapsed = time.time() - t1
        logger.info(
            f"  [_retrieve_and_rerank] rerank: {rerank_elapsed:.3f}s, "
            f"got {len(reranked_nodes)} nodes"
        )

        RETRIEVAL_LATENCY.observe(retrieval_elapsed)
        RERANK_LATENCY.observe(rerank_elapsed)

        final_nodes = reranked_nodes[: self.config.retrieval.final_top_k]
        return [
            RetrievedNode(
                node_id=node.node_id,
                content=node.content,
                score=node.score,
                metadata=node.metadata,
            )
            for node in final_nodes
        ]

    async def _generate_answer(
        self,
        query: str,
        contexts: list[RetrievedNode],
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate answer using LLM."""
        generation_start = time.time()
        result = await self.llm_generator.generate(
            query=query,
            contexts=contexts,
            include_citations=False,  # RAGEngine extracts citations in post-processing
            conversation_history=conversation_history,
        )
        GENERATION_LATENCY.observe(time.time() - generation_start)
        return result

    def _evaluate_confidence(self, contexts: list[RetrievedNode], answer: str, query: str) -> dict[str, Any]:
        """Evaluate confidence of the answer."""
        return self.confidence_evaluator.evaluate(
            contexts=contexts,
            answer=answer,
            query=query,
        )

    def _generate_warnings(
        self,
        answer: str,
        contexts: list[RetrievedNode],
        citations: list | None = None,
    ) -> list[RiskWarning]:
        """Thin wrapper delegating to WarningsGenerator for backward compatibility."""
        return self.warnings_generator.generate(answer, contexts, citations)

    def _create_fallback_response(self, session_id: str, fallback_type: str, processing_time: float) -> QueryResponse:
        fallback: dict[str, Any] = FALLBACK_RESPONSES.get(  # type: ignore[assignment]
            fallback_type, FALLBACK_RESPONSES["no_results"]
        )

        return QueryResponse(
            answer=fallback["answer"],
            confidence=0.0,
            citations=[],
            warnings=[
                RiskWarning(
                    type="general",
                    message="无法找到相关信息，建议您咨询医疗专业人员。",
                    priority="medium",
                )
            ],
            session_id=session_id,
            processing_time=round(processing_time, 2),
            metadata={},
        )

    def _create_error_response(self, session_id: str, error: str, processing_time: float) -> QueryResponse:
        return QueryResponse(
            answer=f"抱歉，处理您的请求时出现错误：{error}",
            confidence=0.0,
            citations=[],
            warnings=[
                RiskWarning(
                    type="error",
                    message="系统错误，请稍后重试或联系技术支持。",
                    priority="high",
                )
            ],
            session_id=session_id,
            processing_time=round(processing_time, 2),
            metadata={"error": error},
        )
