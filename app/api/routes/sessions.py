from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select

from app.core.database import get_session_factory
from app.models.database import Conversation, Message
from app.models.schemas import ConversationSession
from app.models.schemas import Message as MessageSchema


class DeleteResponse(BaseModel):
    message: str


router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


@router.post("", response_model=ConversationSession)
async def create_session(
    request: Request,
) -> ConversationSession:
    session_manager = request.app.state.session_manager
    return await session_manager.create_session_db()


@router.get("", response_model=list[ConversationSession])
async def list_sessions(
    request: Request,
) -> list[ConversationSession]:
    """从数据库直接读取会话列表（包含所有会话，用于历史页面）"""
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(select(Conversation).order_by(Conversation.updated_at.desc()))
        conversations = result.scalars().all()

        sessions = []
        for conv in conversations:
            sessions.append(
                ConversationSession(
                    session_id=str(conv.id),
                    session_title=conv.session_title,
                    created_at=conv.created_at,
                    updated_at=conv.updated_at,
                    messages=[],  # 不加载消息列表，只获取计数
                    context_documents=[],
                    is_active=conv.is_active,
                    db_confirmed=True,
                    msg_count=conv.message_count,
                )
            )
        return sessions


@router.get("/{session_id}/messages", response_model=list[MessageSchema])
async def get_session_messages(
    request: Request,
    session_id: str,
) -> list[MessageSchema]:
    """从数据库直接读取会话消息"""
    import uuid

    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            select(Message).where(Message.session_id == uuid.UUID(session_id)).order_by(Message.created_at)
        )
        messages = result.scalars().all()

        if not messages:
            # 检查会话是否存在
            conv_result = await session.execute(select(Conversation).where(Conversation.id == uuid.UUID(session_id)))
            if not conv_result.scalar_one_or_none():
                raise HTTPException(status_code=404, detail="Session not found")

        return [
            MessageSchema(
                message_id=str(m.id),
                role=m.role,
                content=m.content,
                timestamp=m.created_at,
                metadata=m.extra_data or {},
                confidence=m.confidence,
                citations=m.citations,
                warnings=m.warnings,
            )
            for m in messages
        ]


@router.post("/{session_id}/messages")
async def add_message(
    request: Request,
    session_id: str,
    role: str,
    content: str,
) -> MessageSchema:
    session_manager = request.app.state.session_manager

    if role not in ["user", "assistant", "system"]:
        raise HTTPException(status_code=400, detail="Invalid role")

    message = await session_manager.add_message(
        session_id=session_id,
        role=role,
        content=content,
    )

    if not message:
        raise HTTPException(status_code=404, detail="Session not found")

    return message


@router.delete("/{session_id}", response_model=DeleteResponse)
async def delete_session(
    request: Request,
    session_id: str,
) -> DeleteResponse:
    session_manager = request.app.state.session_manager
    success = await session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return DeleteResponse(message="Session deleted successfully")
