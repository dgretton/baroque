"""Provider request construction.

The builder is intentionally small for the first implementation: it compiles the
prompt-compatible part of a role genome into an OpenAI-style chat request while
preserving requested controls for storage and later analysis.
"""

from __future__ import annotations

from typing import Any

from baroque.core.models import Message, ProviderRequest


class QueryBuilder:
    """Build a provider request from role controls and messages."""

    def __init__(
        self,
        *,
        endpoint_id: str,
        provider: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
    ) -> None:
        self._endpoint_id = endpoint_id
        self._provider = provider
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._messages: list[Message] = []
        self._requested_controls: dict[str, Any] = {}
        self._effective_controls: dict[str, Any] = {}
        self._extra_body: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}

    def with_persona(self, persona_text: str) -> QueryBuilder:
        self._messages.append(Message(role="system", content=persona_text))
        self._requested_controls["persona_text"] = persona_text
        self._effective_controls["persona_text"] = persona_text
        return self

    def with_message(self, role: str, content: str | list[dict[str, Any]]) -> QueryBuilder:
        self._messages.append(Message(role=role, content=content))
        return self

    def with_user(self, content: str | list[dict[str, Any]]) -> QueryBuilder:
        return self.with_message("user", content)

    def with_requested_control(self, name: str, value: Any) -> QueryBuilder:
        self._requested_controls[name] = value
        return self

    def with_effective_control(self, name: str, value: Any) -> QueryBuilder:
        self._effective_controls[name] = value
        return self

    def with_extra_body(self, name: str, value: Any) -> QueryBuilder:
        self._extra_body[name] = value
        self._effective_controls[name] = value
        return self

    def with_metadata(self, name: str, value: Any) -> QueryBuilder:
        self._metadata[name] = value
        return self

    def build(self) -> ProviderRequest:
        return ProviderRequest(
            endpoint_id=self._endpoint_id,
            provider=self._provider,
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._model,
            messages=self._messages,
            requested_controls=self._requested_controls,
            effective_controls=self._effective_controls,
            extra_body=self._extra_body,
            metadata=self._metadata,
        )

