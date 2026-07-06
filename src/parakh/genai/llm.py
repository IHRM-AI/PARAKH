from __future__ import annotations

import httpx

from parakh.config import settings


class GemmaClient:
    """Client for the self-hosted Gemma-4 model served through vLLM.

    The endpoint is OpenAI-compatible. When no endpoint is configured the client
    reports itself unavailable so callers can fall back to a deterministic path.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        self._url = base_url if base_url is not None else settings.llm_base_url
        self._model = model or settings.llm_model
        self._timeout = timeout or settings.llm_timeout_seconds

    @property
    def available(self) -> bool:
        return bool(self._url)

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 700
    ) -> str:
        if not self.available:
            raise RuntimeError("LLM endpoint is not configured (set VLLM_URL).")
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = httpx.post(self._url, json=payload, timeout=self._timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
