"""Query API routes — streaming and synchronous RAG query endpoints."""
import json
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from src.common.di.deps import APIKeyDep, RAGEngineDep, limiter
from src.common.models import QueryRequest, QueryResponse, RiskWarning

router = APIRouter(prefix="/api/v1", tags=["query"])

# ponytail: question length validation / truncation handled by Pydantic field_validator


@router.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(
    request: Request,
    request_data: QueryRequest,
    rag_engine: RAGEngineDep,
    api_key: APIKeyDep,
) -> QueryResponse:
    # Generate trace_id if not provided, for request correlation
    trace_id = request_data.trace_id or str(uuid.uuid4())

    # ponytail: question is already stripped + truncated by Pydantic field_validator
    was_truncated = len(request_data.question) >= 2000

    session_manager = request.app.state.container.session_manager

    # Create session for anonymous queries BEFORE calling rag_engine
    if not request_data.session_id:
        new_session = await session_manager.create_session_db()
        request_data.session_id = new_session.session_id

    # Pass session_manager to RAGEngine for conversation context
    response = await rag_engine.query(request_data, session_manager, trace_id=trace_id)

    # Ensure response has the session_id (for new sessions)
    if not response.session_id:
        response.session_id = request_data.session_id or ""

    # Echo back trace_id for correlation
    response.trace_id = trace_id

    if was_truncated:
        response.warnings.insert(
            0,
            RiskWarning(
                type="input_truncation",
                message="输入问题长度超过2000字符限制，已自动截断",
                priority="low",
            ),
        )

    return response


@router.post("/query/stream")
async def query_stream(
    request: Request,
    request_data: QueryRequest,
    rag_engine: RAGEngineDep,
) -> StreamingResponse:
    trace_id = request_data.trace_id or str(uuid.uuid4())

    session_manager = request.app.state.container.session_manager

    # Create session for anonymous queries
    if not request_data.session_id:
        new_session = await session_manager.create_session_db()
        request_data.session_id = new_session.session_id

    async def event_generator():
        try:
            async for event in rag_engine.query_stream(request_data, session_manager, trace_id=trace_id):
                event_type = event["type"]
                event_data = event["data"]
                yield f"event: {event_type}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"SSE generator error: {e}")
            error_data = json.dumps(
                {"message": f"流式处理出错：{e}", "code": "INTERNAL_ERROR"},
                ensure_ascii=False,
            )
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
