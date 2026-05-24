import json
import uuid
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from app.api.deps import RAGEngineDep, APIKeyDep
from app.models.schemas import QueryRequest, QueryResponse, RiskWarning
from app.api.deps import limiter

router = APIRouter(prefix="/api/v1", tags=["query"])

MAX_QUESTION_LENGTH = 2000


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

    if not request_data.question or not request_data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    original_length = len(request_data.question)
    was_truncated = False

    if original_length > MAX_QUESTION_LENGTH:
        request_data.question = request_data.question[:MAX_QUESTION_LENGTH]
        was_truncated = True

    session_manager = request.app.state.session_manager

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
                message=f"输入问题长度({original_length}字符)超过限制，已截断至{MAX_QUESTION_LENGTH}字符",
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

    if not request_data.question or not request_data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    original_length = len(request_data.question)
    if original_length > MAX_QUESTION_LENGTH:
        request_data.question = request_data.question[:MAX_QUESTION_LENGTH]

    session_manager = request.app.state.session_manager

    # Create session for anonymous queries
    if not request_data.session_id:
        new_session = await session_manager.create_session_db()
        request_data.session_id = new_session.session_id

    async def event_generator():
        try:
            async for event in rag_engine.query_stream(
                request_data, session_manager, trace_id=trace_id
            ):
                event_type = event["type"]
                event_data = event["data"]
                yield f"event: {event_type}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"SSE generator error: {e}")
            error_data = {
                "type": "error",
                "data": {
                    "message": f"流式处理出错：{str(e)}",
                    "code": "INTERNAL_ERROR",
                },
            }
            yield f"event: error\ndata: {json.dumps(error_data['data'], ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )