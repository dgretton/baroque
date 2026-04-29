"""OpenAI-compatible inference gateway."""

from __future__ import annotations

from typing import Any

import httpx

from baroque.core.hashing import content_hash
from baroque.core.models import ProviderRequest, ProviderResponse


class OpenAICompatibleGateway:
    """Send chat-completion requests to OpenAI-compatible endpoints."""

    def __init__(self, *, timeout_s: float = 600) -> None:
        self._timeout_s = timeout_s

    async def send(self, request: ProviderRequest) -> ProviderResponse:
        headers: dict[str, str] = {"content-type": "application/json"}
        if request.api_key:
            headers["authorization"] = f"Bearer {request.api_key}"

        url = request.base_url.rstrip("/") + "/chat/completions"
        payload = request.chat_payload()
        request_hash = content_hash(payload)

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            raw_body: dict[str, Any] = response.json()

        choice = (raw_body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return ProviderResponse(
            request_hash=request_hash,
            raw_body=raw_body,
            text=message.get("content"),
            finish_reason=choice.get("finish_reason"),
            usage=raw_body.get("usage") or {},
        )

