import time
from contextvars import ContextVar
from typing import Any, AsyncGenerator

from loguru import logger

from app.core.confidence import ConfidenceEvaluator
from app.core.metrics import ACTIVE_QUERIES, ERROR_COUNT, GENERATION_LATENCY, LLM_TOKENS, QUERY_LATENCY, RETRIEVAL_COUNT, RETRIEVAL_LATENCY, RERANK_LATENCY
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
        """
        处理文档向量化，将 embedding 模型加载到 GPU 进行编码。

        流程:
        1. 检测 reranker 是否在 GPU，如果是则迁移到 CPU
        2. 加载 embedding 到 GPU
        3. 执行向量化
        4. 迁移 embedding 到 CPU（保留模型对象）
        5. 恢复 reranker 到 GPU

        Args:
            nodes: 要处理的文档节点列表

        Returns:
            bool: 是否成功处理
        """
        from app.core.gpu_memory_manager import GPUMemoryManager

        GPUMemoryManager.get_instance()
        vector_retriever = self.hybrid_retriever.vector_retriever
        reranker_was_on_gpu = False

        try:
            # Step 1: 如果 reranker 在 GPU，先迁移到 CPU
            if self.reranker.is_on_gpu():
                logger.info("Moving reranker to CPU to free GPU memory")
                self.reranker.move_to_cpu()
                reranker_was_on_gpu = True

            # Step 2: 加载 embedding 到 GPU
            if not vector_retriever.load_embedding_to_gpu():
                logger.error("Failed to load embedding to GPU")
                # 回退: 保持在 CPU 继续处理
                return False

            # Step 3: 执行向量化（使用 GPU 上的 embedding 模型）
            # 向量化在 VectorRetriever.add() 中自动使用 embedding_model
            await self.hybrid_retriever.add_documents(nodes)

            # Step 4: 迁移 embedding 到 CPU（保留模型对象）
            vector_retriever.move_embedding_to_cpu()

            # Step 5: 不需要主动恢复 reranker 到 GPU
            # Reranker 设计为懒加载，只有在 rerank() 时才加载
            # 这样可以保持 GPU 空闲，让后续查询可以直接加载 reranker
            logger.info("Document processing done, reranker remains lazy-loaded")

            logger.info(f"Document processing completed: {len(nodes)} chunks vectorized")
            return True

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            # 出错时保持懒加载，不主动加载 reranker
            return False
        finally:
            # Ensure embedding model is moved back to CPU even on exception
            if vector_retriever.is_on_gpu():
                vector_retriever.move_embedding_to_cpu()

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

            try:
                reranked_nodes = await self._retrieve_and_rerank(sanitized_query, request.filters)
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

            # Build conversation context if session_manager provided
            conversation_history: list[dict[str, Any]] | None = None
            if session_manager and request.session_id:
                session = session_manager.get_session(request.session_id)
                if session:
                    conversation_history = [
                        {"role": msg.role, "content": msg.content}
                        for msg in session.messages
                    ]

            try:
                llm_result = await self._generate_answer(
                    sanitized_query, reranked_nodes, conversation_history
                )
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
            confidence_result = self._evaluate_confidence(
                reranked_nodes, llm_result["answer"], sanitized_query
            )

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
                warnings = self.warnings_generator.generate(
                    llm_result["answer"], reranked_nodes, citations
                )

            processing_time = time.time() - start_time
            QUERY_LATENCY.observe(processing_time)

            # Add messages to session AFTER generation succeeds
            if session_manager and request.session_id:
                await session_manager.add_message(
                    request.session_id, "user", request.question
                )
                await session_manager.add_message(
                    request.session_id,
                    "assistant",
                    llm_result["answer"],
                    metadata={
                        "confidence": confidence_result.get("confidence"),
                        "citations": [c.model_dump() if hasattr(c, 'model_dump') else c for c in llm_result.get("citations", [])],
                        "warnings": [w.model_dump() if hasattr(w, 'model_dump') else w for w in warnings],
                    },
                )

            return QueryResponse(
                answer=llm_result["answer"],
                confidence=confidence_result["confidence"],
                citations=llm_result.get("citations", []),
                warnings=warnings,
                session_id=request.session_id or "",
                processing_time=round(processing_time, 2),
                metadata={
                    "retrieved_chunks": len(reranked_nodes),
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
            # Safety check
            safety_result = self._safety_check(request)
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

            # Retrieval
            try:
                reranked_nodes = await self._retrieve_and_rerank(sanitized_query, request.filters)
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

            if not reranked_nodes:
                RETRIEVAL_COUNT.labels(retriever_type="none").inc()
                yield {
                    "type": "done",
                    "data": self._create_fallback_response(
                        request.session_id or "", "no_results", time.time() - start_time
                    ).model_dump(),
                }
                return

            # Build conversation context
            conversation_history: list[dict[str, Any]] | None = None
            if session_manager and request.session_id:
                session = session_manager.get_session(request.session_id)
                if session:
                    conversation_history = [
                        {"role": msg.role, "content": msg.content}
                        for msg in session.messages
                    ]

            # Pre-stream: yield placeholder metadata
            yield {
                "type": "metadata",
                "data": {
                    "session_id": request.session_id or "",
                    "trace_id": trace_id or _trace_id_var.get(),
                },
            }

            # Stream generation
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

            processing_time = time.time() - start_time

            # Evaluate confidence
            confidence_result = self._evaluate_confidence(
                reranked_nodes, full_answer, sanitized_query
            )

            # Extract and verify citations
            citations = []
            if self.config.generation.include_citations:
                citation_verifier = CitationVerifier()
                citations = citation_verifier.extract_and_verify(
                    answer=full_answer,
                    contexts=reranked_nodes,
                )

            # Generate warnings
            warnings = []
            if self.config.generation.include_warnings:
                warnings = self.warnings_generator.generate(full_answer, reranked_nodes, citations)

            # Persist messages
            if session_manager and request.session_id:
                await session_manager.add_message(
                    request.session_id, "user", request.question
                )
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
        self, query: str, filters: dict[str, Any] | None = None
    ) -> list[RetrievedNode]:
        """Retrieve and rerank documents for the query."""
        retrieval_start = time.time()
        retrieved_nodes = await self.hybrid_retriever.search(
            query=query,
            top_k=self.config.retrieval.final_top_k * 2,
            filters=filters,
        )
        RETRIEVAL_LATENCY.observe(time.time() - retrieval_start)

        if not retrieved_nodes:
            return []

        RETRIEVAL_COUNT.labels(retriever_type="hybrid").inc()

        rerank_start = time.time()
        reranked_nodes = self.reranker.rerank(
            query=query,
            candidates=retrieved_nodes[: self.config.retrieval.final_top_k * 2],
        )
        RERANK_LATENCY.observe(time.time() - rerank_start)

        # Convert RerankedNode back to RetrievedNode for type consistency
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
            include_citations=self.config.generation.include_citations,
            conversation_history=conversation_history,
        )
        GENERATION_LATENCY.observe(time.time() - generation_start)
        return result

    def _evaluate_confidence(
        self, contexts: list[RetrievedNode], answer: str, query: str
    ) -> dict[str, Any]:
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

    def _create_fallback_response(
        self, session_id: str, fallback_type: str, processing_time: float
    ) -> QueryResponse:
        fallback = FALLBACK_RESPONSES.get(fallback_type, FALLBACK_RESPONSES["no_results"])

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

    def _create_error_response(
        self, session_id: str, error: str, processing_time: float
    ) -> QueryResponse:
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
