from typing import Any

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.models.schemas import RetrievedNode
from app.services.citation_verifier import CitationVerifier
from config.settings import get_settings
from rag.generation.prompt_builder import PromptBuilder


class LLMGenerator:
    _client: AsyncOpenAI | None = None
    _client_config: tuple | None = None

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        settings = get_settings()
        llm_config = settings.models.llm

        self.api_key = api_key or llm_config.api_key
        self.api_base = api_base or llm_config.api_base
        self.model = model or llm_config.model
        self.temperature = temperature or llm_config.temperature
        self.max_tokens = max_tokens or llm_config.max_tokens

        self._ensure_client()

    def _ensure_client(self) -> None:
        """Ensure class-level singleton client is initialized."""
        config_key = (self.api_key, self.api_base)
        if LLMGenerator._client is None or LLMGenerator._client_config != config_key:
            LLMGenerator._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=120.0,
            )
            LLMGenerator._client_config = config_key

    @property
    def client(self) -> AsyncOpenAI:
        """Get the singleton client instance."""
        if LLMGenerator._client is None:
            self._ensure_client()
        return LLMGenerator._client  # type: ignore[return-value]

    async def generate(
        self,
        query: str,
        contexts: list[RetrievedNode],
        include_citations: bool = True,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        system_prompt, user_prompt = PromptBuilder.build(query, contexts, conversation_history)

        response = await self._call_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        answer = response.choices[0].message.content or ""

        citations = []
        if include_citations:
            verifier = CitationVerifier()
            citations = verifier.extract_and_verify(answer, contexts)

        return {
            "answer": answer,
            "citations": citations,
            "confidence": self._estimate_confidence(contexts),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }

    async def generate_stream(
        self,
        query: str,
        contexts: list[RetrievedNode],
        conversation_history: list[dict[str, Any]] | None = None,
    ):
        system_prompt, user_prompt = PromptBuilder.build(query, contexts, conversation_history)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            timeout=120.0,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        reraise=True,
    )
    async def _call_with_retry(self, system_prompt: str, user_prompt: str):
        """Call LLM API with exponential backoff retry."""
        return await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _estimate_confidence(self, contexts: list[RetrievedNode]) -> float:
        if not contexts:
            return 0.0

        scores = [ctx.score for ctx in contexts]
        avg_score = sum(scores) / len(scores)

        return round(min(avg_score, 1.0), 2)
