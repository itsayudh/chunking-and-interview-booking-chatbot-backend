"""Custom RAG orchestration: rewrite -> retrieve -> prompt -> generate (no prebuilt chain)."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from app.memory.redis_memory import ChatMessage
from app.rag.llm import LLMMessage, LLMService
from app.rag.prompts import (
    ANSWER_SYSTEM_PROMPT,
    REFUSAL_MESSAGE,
    build_answer_prompt,
    format_context,
)
from app.rag.query_rewriter import QueryRewriter
from app.rag.retriever import Retriever
from app.rag.vector_store import SearchHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """Result of answering one question."""

    answer: str
    answered: bool  # False when the documents did not contain the answer
    standalone_question: str
    sources: list[SearchHit]


class RagPipeline:
    """Answers questions strictly from the ingested documents."""

    def __init__(
        self,
        *,
        llm: LLMService,
        rewriter: QueryRewriter,
        retriever: Retriever,
        temperature: float | None = None,
    ) -> None:
        self._llm = llm
        self._rewriter = rewriter
        self._retriever = retriever
        self._temperature = temperature

    async def answer(self, question: str, history: Sequence[ChatMessage]) -> RagAnswer:
        """Answer a question using the conversation history to resolve follow-ups."""
        standalone = await self._rewriter.rewrite(question, history)
        hits = await self._retriever.retrieve(standalone)

        if not hits:
            logger.info("No relevant chunks for: %s", standalone)
            return RagAnswer(REFUSAL_MESSAGE, False, standalone, [])

        prompt = build_answer_prompt(format_context(hits), standalone)
        text = await self._llm.generate(
            system_instruction=ANSWER_SYSTEM_PROMPT,
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=self._temperature,
        )

        if text.lower().startswith(REFUSAL_MESSAGE.lower()):
            return RagAnswer(REFUSAL_MESSAGE, False, standalone, [])
        return RagAnswer(text, True, standalone, hits)