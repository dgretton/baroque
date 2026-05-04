"""Tests for the native Anthropic SDK gateway."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from baroque.core.models import Message, ProviderRequest
from baroque.gateways.anthropic_native import AnthropicGateway


@dataclass
class _FakeUsage:
    input_tokens: int = 11
    output_tokens: int = 22


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeToolUseBlock:
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class _FakeMessage:
    content: list[Any]
    stop_reason: str = "end_turn"
    usage: _FakeUsage = field(default_factory=_FakeUsage)

    def model_dump(self, *, mode: str = "python") -> dict[str, Any]:
        del mode
        return {"stop_reason": self.stop_reason}


class _FakeMessages:
    def __init__(self, response: _FakeMessage) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeMessage:
        self.calls.append(kwargs)
        return self._response


class _FakeAsyncAnthropic:
    def __init__(self, response: _FakeMessage) -> None:
        self.messages = _FakeMessages(response)


def _build_request(extra_body: dict[str, Any] | None = None) -> ProviderRequest:
    return ProviderRequest(
        endpoint_id="anthropic_api",
        provider="anthropic_native",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-6",
        api_key="key",
        messages=[
            Message(role="system", content="You are precise."),
            Message(role="user", content="What is two plus two?"),
        ],
        extra_body=extra_body or {},
    )


def test_anthropic_gateway_hoists_system_message_and_returns_text() -> None:
    fake_message = _FakeMessage(content=[_FakeTextBlock(text="Four.")])
    fake_client = _FakeAsyncAnthropic(fake_message)
    gateway = AnthropicGateway(client=fake_client)

    response = asyncio.run(gateway.send(_build_request({"temperature": 0.2, "max_tokens": 64})))

    call = fake_client.messages.calls[0]
    assert call["model"] == "claude-sonnet-4-6"
    assert call["system"] == "You are precise."
    assert call["messages"] == [{"role": "user", "content": "What is two plus two?"}]
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 64
    assert "tools" not in call
    assert response.text == "Four."
    assert response.finish_reason == "end_turn"
    assert response.usage["input_tokens"] == 11
    assert response.usage["output_tokens"] == 22


def test_anthropic_gateway_translates_json_schema_response_format_to_tool_use() -> None:
    schema = {
        "type": "object",
        "required": ["overall_rating"],
        "properties": {
            "overall_rating": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "rationale": {"type": "string"},
        },
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "grader_assessment",
            "schema": schema,
            "strict": True,
        },
    }
    tool_use_payload = {"overall_rating": 0.8, "rationale": "Strong extraction."}
    fake_message = _FakeMessage(
        content=[_FakeToolUseBlock(name="grader_assessment", input=tool_use_payload)],
        stop_reason="tool_use",
    )
    fake_client = _FakeAsyncAnthropic(fake_message)
    gateway = AnthropicGateway(client=fake_client)

    response = asyncio.run(gateway.send(_build_request({"response_format": response_format})))

    call = fake_client.messages.calls[0]
    assert len(call["tools"]) == 1
    tool = call["tools"][0]
    assert tool["name"] == "grader_assessment"
    assert tool["input_schema"] == schema
    assert call["tool_choice"] == {"type": "tool", "name": "grader_assessment"}
    assert response.text is not None
    assert json.loads(response.text) == tool_use_payload
    assert response.finish_reason == "tool_use"


def test_anthropic_gateway_appends_json_object_rider_for_json_object_mode() -> None:
    fake_message = _FakeMessage(content=[_FakeTextBlock(text='{"x": 1}')])
    fake_client = _FakeAsyncAnthropic(fake_message)
    gateway = AnthropicGateway(client=fake_client)

    asyncio.run(
        gateway.send(_build_request({"response_format": {"type": "json_object"}}))
    )

    call = fake_client.messages.calls[0]
    assert "tools" not in call
    assert "respond with a single JSON object" in call["system"]
    assert "You are precise." in call["system"]


def test_anthropic_gateway_rejects_unsupported_role() -> None:
    request = ProviderRequest(
        endpoint_id="anthropic_api",
        provider="anthropic_native",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-6",
        messages=[Message(role="developer", content="hello")],
    )
    fake_client = _FakeAsyncAnthropic(_FakeMessage(content=[]))
    gateway = AnthropicGateway(client=fake_client)

    with pytest.raises(ValueError, match="cannot send role"):
        asyncio.run(gateway.send(request))
