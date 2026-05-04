"""Native Anthropic SDK inference gateway.

Translates the project's OpenAI-shaped `ProviderRequest` into Anthropic's
messages API call. The first system-role message in the request is hoisted
into the top-level `system` parameter (Anthropic does not accept system
messages inline). When the request carries a `response_format` of type
`json_schema` in `extra_body`, the gateway translates it into a single tool
definition and forces a tool call so the response payload is schema-strict.
The tool's input is then serialized to JSON and returned as
`ProviderResponse.text`, keeping handler-side parsing identical to the
OpenAI-compatible path.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from baroque.core.hashing import content_hash
from baroque.core.models import ProviderRequest, ProviderResponse

_DEFAULT_MAX_TOKENS = 1024


class AnthropicGateway:
    """Send chat-completion requests to Anthropic's native messages API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_s: float = 600,
        max_retries: int = 2,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._timeout_s = timeout_s
        if client is not None:
            self._client = client
        else:
            self._client = anthropic.AsyncAnthropic(
                api_key=api_key,
                timeout=timeout_s,
                max_retries=max_retries,
            )

    async def send(self, request: ProviderRequest) -> ProviderResponse:
        system_text, anthropic_messages = _split_system_and_messages(request.messages)
        params = _base_params(request)
        request_hash = content_hash(
            {
                "model": request.model,
                "messages": [message.model_dump(exclude_none=True) for message in request.messages],
                **request.extra_body,
            }
        )
        params["model"] = request.model
        params["messages"] = anthropic_messages
        if system_text:
            params["system"] = system_text

        response_format = request.extra_body.get("response_format")
        tool_name: str | None = None
        if isinstance(response_format, dict):
            type_value = response_format.get("type")
            if type_value == "json_schema":
                tool_name, tool = _tool_from_json_schema(response_format)
                params["tools"] = [tool]
                params["tool_choice"] = {"type": "tool", "name": tool_name}
            elif type_value == "json_object":
                # Anthropic has no 1:1 equivalent. Append a system rider; the
                # caller's user-message schema instruction still does the heavy
                # lifting. Documented degradation.
                rider = (
                    "Output rules: respond with a single JSON object and no other "
                    "text."
                )
                params["system"] = (
                    f"{params['system']}\n\n{rider}" if params.get("system") else rider
                )

        message = await self._client.messages.create(**params)
        text, finish_reason = _extract_text_and_finish(message, expected_tool_name=tool_name)
        usage = _usage(message)

        return ProviderResponse(
            request_hash=request_hash,
            raw_body=_safe_dump(message),
            text=text,
            finish_reason=finish_reason,
            usage=usage,
        )


def _split_system_and_messages(
    messages: list[Any],
) -> tuple[str, list[dict[str, Any]]]:
    system_chunks: list[str] = []
    chat_messages: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            system_chunks.append(_message_text(message.content))
            continue
        if message.role not in {"user", "assistant"}:
            raise ValueError(f"Anthropic gateway cannot send role: {message.role}")
        chat_messages.append({"role": message.role, "content": message.content})
    return "\n\n".join(chunk for chunk in system_chunks if chunk), chat_messages


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _base_params(request: ProviderRequest) -> dict[str, Any]:
    params: dict[str, Any] = {}
    extra = request.extra_body
    max_tokens = extra.get("max_tokens", _DEFAULT_MAX_TOKENS)
    params["max_tokens"] = int(max_tokens)
    if "temperature" in extra:
        params["temperature"] = float(extra["temperature"])
    if "top_p" in extra:
        params["top_p"] = float(extra["top_p"])
    if "top_k" in extra:
        params["top_k"] = int(extra["top_k"])
    return params


def _tool_from_json_schema(response_format: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    spec = response_format.get("json_schema") or {}
    name = str(spec.get("name") or "structured_response")
    schema = spec.get("schema") or {"type": "object"}
    description = str(spec.get("description") or f"Return a {name} object.")
    return name, {
        "name": name,
        "description": description,
        "input_schema": schema,
    }


def _extract_text_and_finish(
    message: Any,
    *,
    expected_tool_name: str | None,
) -> tuple[str | None, str | None]:
    finish_reason = getattr(message, "stop_reason", None)
    blocks = getattr(message, "content", None) or []
    if expected_tool_name is not None:
        for block in blocks:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == expected_tool_name
            ):
                tool_input = getattr(block, "input", None)
                if tool_input is None:
                    return None, finish_reason
                return json.dumps(tool_input, ensure_ascii=False), finish_reason
        return None, finish_reason
    text_parts: list[str] = []
    for block in blocks:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
    return ("\n".join(text_parts) if text_parts else None), finish_reason


def _usage(message: Any) -> dict[str, Any]:
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
    }


def _safe_dump(message: Any) -> dict[str, Any] | None:
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:
            return None
    return None
