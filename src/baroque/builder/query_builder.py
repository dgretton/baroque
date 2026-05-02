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
        capability_profile: dict[str, Any] | None = None,
    ) -> None:
        self._endpoint_id = endpoint_id
        self._provider = provider
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._capability_profile = capability_profile or {}
        self._messages: list[Message] = []
        self._requested_controls: dict[str, Any] = {}
        self._effective_controls: dict[str, Any] = {}
        self._dropped_controls: dict[str, Any] = {}
        self._extra_body: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {
            "control_compilation": {
                "capability_profile": self._capability_profile.get("id"),
                "provider_requirements": self._capability_profile.get(
                    "provider_requirements",
                    {},
                ),
            }
        }
        self._validate_provider_requirements()

    def with_controls(self, controls: dict[str, Any]) -> QueryBuilder:
        for name, value in controls.items():
            self._compile_control(name, value)
        return self

    def with_persona(self, persona_text: str) -> QueryBuilder:
        self._compile_control("persona_text", persona_text)
        return self

    def with_message(self, role: str, content: str | list[dict[str, Any]]) -> QueryBuilder:
        self._messages.append(Message(role=role, content=content))
        return self

    def with_user(self, content: str | list[dict[str, Any]]) -> QueryBuilder:
        return self.with_message("user", content)

    def with_requested_control(self, name: str, value: Any) -> QueryBuilder:
        self._compile_control(name, value)
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
            dropped_controls=self._dropped_controls,
            extra_body=self._extra_body,
            metadata=self._metadata,
        )

    def _compile_control(self, name: str, value: Any) -> None:
        self._requested_controls[name] = value
        allowed, reason = self._control_allowed(name)
        if not allowed:
            self._drop_control(name, value, reason)
            return

        if name == "persona_text":
            persona_text = _control_value(value)
            if persona_text:
                self._messages.append(Message(role="system", content=str(persona_text)))
                self._effective_controls[name] = value
            else:
                self._drop_control(name, value, "empty_persona_text")
        elif name == "plain_output_instructions":
            instructions = _control_value(value)
            if instructions:
                self._messages.append(
                    Message(role="system", content=f"Output instructions: {instructions}")
                )
                self._effective_controls[name] = value
            else:
                self._drop_control(name, value, "empty_plain_output_instructions")
        elif name == "message_examples":
            self._compile_message_examples(name, value)
        elif name == "transcript_policy":
            self._effective_controls[name] = value
        elif name == "sampling":
            self._compile_sampling(name, value)
        elif name == "seed":
            seed = _control_value(value)
            if seed is None:
                self._drop_control(name, value, "empty_seed")
            else:
                self._extra_body["seed"] = seed
                self._effective_controls[name] = value
        elif name == "model_choice":
            self._effective_controls[name] = value
        else:
            self._drop_control(name, value, "allowed_but_not_compiled")

    def _compile_message_examples(self, name: str, value: Any) -> None:
        examples = value.get("examples") if isinstance(value, dict) else value
        if examples is None:
            self._drop_control(name, value, "empty_message_examples")
            return
        if not isinstance(examples, list):
            self._drop_control(name, value, "invalid_message_examples")
            return

        for example in examples:
            if not isinstance(example, dict):
                self._drop_control(name, value, "invalid_message_example")
                return
            role = example.get("role")
            content = example.get("content")
            if not isinstance(role, str) or content is None:
                self._drop_control(name, value, "invalid_message_example")
                return
            self._messages.append(Message(role=role, content=content))
        self._effective_controls[name] = value

    def _compile_sampling(self, name: str, value: Any) -> None:
        sampling = value if isinstance(value, dict) else {}
        allowed_keys = {"temperature", "top_p", "top_k", "max_tokens", "seed"}
        compiled = {
            key: sampling[key]
            for key in allowed_keys
            if key in sampling and sampling[key] is not None
        }
        if not compiled:
            self._drop_control(name, value, "empty_sampling")
            return
        self._extra_body.update(compiled)
        self._effective_controls[name] = value

    def _control_allowed(self, name: str) -> tuple[bool, str]:
        denied_controls = set(self._capability_profile.get("denied_controls") or [])
        if name in denied_controls:
            return False, "denied_by_capability_profile"

        allowed_controls = set(self._capability_profile.get("allowed_controls") or [])
        if allowed_controls and name not in allowed_controls:
            return False, "not_allowed_by_capability_profile"
        return True, "allowed"

    def _drop_control(self, name: str, value: Any, reason: str) -> None:
        self._dropped_controls[name] = {
            "value": value,
            "reason": reason,
            "capability_profile": self._capability_profile.get("id"),
        }

    def _validate_provider_requirements(self) -> None:
        requirements = self._capability_profile.get("provider_requirements") or {}
        required_provider = requirements.get("provider")
        if required_provider and required_provider != self._provider:
            raise ValueError(
                "capability profile provider requirement does not match endpoint provider: "
                f"{required_provider} != {self._provider}"
            )
        self._metadata["control_compilation"]["provider_requirements_satisfied"] = True


def _control_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value.get("value")
    return value
