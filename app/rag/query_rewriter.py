"""Turns follow-up questions into standalone questions using the chat history."""

import logging
from collections.abc import Sequence

from app.core.exceptions import LLMError
from app.memory.redis_memory import ChatMessage
from app.rag.llm import LLMMessage, LLMService
from app.rag.prompts import REWRITE_SYSTEM_PROMPT, build_rewrite_prompt

logger = logging.getLogger(__name__)

_MAX_REWRITTEN_CHARS = 500


class QueryRewriter:
    """Rewrites a question so retrieval does not depend on earlier messages."""

    def __init__(self, llm: LLMService, max_history_messages: int = 6) -> None:
        if max_history_messages <= 0:
            raise ValueError("max_history_messages must be greater than 0")
        self._llm = llm
        self._max_history_messages = max_history_messages

    async def rewrite(self, question: str, history: Sequence[ChatMessage]) -> str:
        """Return a standalone version of the question.

        Rewriting is an optimisation, not a requirement: with no history, or if the model
        fails or returns something unusable, the original question is returned.
        """
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        recent = list(history)[-self._max_history_messages :]
        if not recent:
            return question

        prompt = build_rewrite_prompt(recent, question)
        try:
            rewritten = await self._llm.generate(
                system_instruction=REWRITE_SYSTEM_PROMPT,
                messages=[LLMMessage(role="user", content=prompt)],
            )
        except LLMError:
            logger.warning("Query rewriting failed; using the original question")
            return question

        first_line = rewritten.splitlines()[0].strip().strip("\"'")
        if not first_line or len(first_line) > _MAX_REWRITTEN_CHARS:
            return question
        return first_line