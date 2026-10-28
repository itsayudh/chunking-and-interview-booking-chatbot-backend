"""Conversational RAG endpoints."""
from fastapi import APIRouter, status
from app.api.deps import ChatServiceDep
from app.schemas.booking import BookingResponse
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    SourceReference,
)
from app.services.chat_service import ChatResult

router = APIRouter(prefix="/chat", tags=["chat"])


def _to_response(result: ChatResult) -> ChatResponse:
    return ChatResponse(
        session_id=result.session_id,
        answer=result.reply,
        intent=result.intent,
        answered=result.answered,
        sources=[
            SourceReference(
                document_id=hit.document_id,
                filename=hit.filename,
                chunk_index=hit.chunk_index,
                score=hit.score,
            )
            for hit in result.sources
        ],
        booking=BookingResponse.model_validate(result.booking) if result.booking else None,
        booking_status=result.booking_status.value if result.booking_status else None,
    )


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    """Send a message. Answers questions from the documents, or handles interview booking."""
    result = await service.chat(message=request.message, session_id=request.session_id)
    return _to_response(result)


@router.get("/{session_id}/history", response_model=ChatHistoryResponse)
async def get_history(session_id: str, service: ChatServiceDep) -> ChatHistoryResponse:
    """Return the stored conversation of a session."""
    messages = await service.get_history(session_id)
    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            ChatMessageResponse(role=m.role, content=m.content, timestamp=m.timestamp)
            for m in messages
        ],
    )


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, service: ChatServiceDep) -> None:
    """Delete a conversation and any booking in progress."""
    await service.clear_session(session_id)