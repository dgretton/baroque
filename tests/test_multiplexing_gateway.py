"""Tests for the provider-routing MultiplexingGateway."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from baroque.core.models import Message, ProviderRequest, ProviderResponse
from baroque.gateways import MultiplexingGateway


class _RecordingGateway:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[ProviderRequest] = []

    async def send(self, request: ProviderRequest) -> ProviderResponse:
        self.calls.append(request)
        return ProviderResponse(request_hash="sha256:x", text=self.label)


def _request(provider: str, **kwargs: Any) -> ProviderRequest:
    return ProviderRequest(
        endpoint_id="endpoint-x",
        provider=provider,
        base_url="https://example.invalid",
        model="m",
        messages=[Message(role="user", content="hi")],
        **kwargs,
    )


def test_multiplexing_gateway_dispatches_by_provider() -> None:
    ollama = _RecordingGateway("ollama")
    anthropic = _RecordingGateway("anthropic")
    multiplexer = MultiplexingGateway(
        {
            "ollama_openai": ollama,
            "anthropic_native": anthropic,
        }
    )

    response_ollama = asyncio.run(multiplexer.send(_request("ollama_openai")))
    response_anthropic = asyncio.run(multiplexer.send(_request("anthropic_native")))

    assert response_ollama.text == "ollama"
    assert response_anthropic.text == "anthropic"
    assert len(ollama.calls) == 1
    assert len(anthropic.calls) == 1


def test_multiplexing_gateway_raises_for_unknown_provider() -> None:
    ollama = _RecordingGateway("ollama")
    multiplexer = MultiplexingGateway({"ollama_openai": ollama})

    with pytest.raises(ValueError, match="no gateway registered for provider: openai"):
        asyncio.run(multiplexer.send(_request("openai")))


def test_multiplexing_gateway_requires_at_least_one_gateway() -> None:
    with pytest.raises(ValueError, match="at least one gateway"):
        MultiplexingGateway({})
