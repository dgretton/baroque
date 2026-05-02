import pytest

from baroque.builder.query_builder import QueryBuilder


def test_query_builder_drops_denied_controls_and_records_reasons() -> None:
    request = (
        QueryBuilder(
            endpoint_id="local_ollama",
            provider="ollama_openai",
            base_url="http://localhost:11434/v1",
            model="gemma4:e2b",
            api_key="ollama",
            capability_profile={
                "id": "prompt_only_ollama",
                "allowed_controls": ["persona_text", "plain_output_instructions"],
                "denied_controls": ["sampling"],
                "provider_requirements": {"provider": "ollama_openai"},
            },
        )
        .with_controls(
            {
                "persona_text": {"value": "Be precise."},
                "plain_output_instructions": {"value": "Return JSON."},
                "sampling": {"temperature": 0.8},
            }
        )
        .with_user("Question?")
        .build()
    )

    assert request.requested_controls["sampling"] == {"temperature": 0.8}
    assert request.effective_controls["persona_text"] == {"value": "Be precise."}
    assert request.effective_controls["plain_output_instructions"] == {
        "value": "Return JSON."
    }
    assert request.dropped_controls["sampling"]["reason"] == "denied_by_capability_profile"
    assert request.extra_body == {}
    assert [message.role for message in request.messages] == ["system", "system", "user"]


def test_query_builder_compiles_sampling_when_profile_allows_it() -> None:
    request = (
        QueryBuilder(
            endpoint_id="local_ollama",
            provider="ollama_openai",
            base_url="http://localhost:11434/v1",
            model="gemma4:e2b",
            capability_profile={
                "id": "ollama_prompt_params",
                "allowed_controls": ["sampling", "seed"],
                "denied_controls": [],
                "provider_requirements": {"provider": "ollama_openai"},
            },
        )
        .with_controls(
            {
                "sampling": {"temperature": 0.4, "top_p": 0.9, "unused": "ignored"},
                "seed": {"value": 123},
            }
        )
        .with_user("Question?")
        .build()
    )

    assert request.extra_body == {"temperature": 0.4, "top_p": 0.9, "seed": 123}
    assert request.effective_controls["sampling"] == {
        "temperature": 0.4,
        "top_p": 0.9,
        "unused": "ignored",
    }
    assert request.effective_controls["seed"] == {"value": 123}
    assert request.dropped_controls == {}


def test_query_builder_rejects_provider_requirement_mismatch() -> None:
    with pytest.raises(ValueError, match="provider requirement"):
        QueryBuilder(
            endpoint_id="remote",
            provider="openai",
            base_url="https://example.invalid/v1",
            model="gpt-example",
            capability_profile={
                "id": "prompt_only_ollama",
                "allowed_controls": ["persona_text"],
                "denied_controls": [],
                "provider_requirements": {"provider": "ollama_openai"},
            },
        )
