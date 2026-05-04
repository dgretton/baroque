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


def test_query_builder_compiles_response_format_into_extra_body() -> None:
    schema = {
        "type": "object",
        "required": ["disclosure_points", "overall_rating"],
        "properties": {
            "disclosure_points": {"type": "array"},
            "overall_rating": {"type": "number"},
        },
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "grader_assessment", "schema": schema, "strict": True},
    }
    request = (
        QueryBuilder(
            endpoint_id="local_ollama",
            provider="ollama_openai",
            base_url="http://localhost:11434/v1",
            model="gemma4:e2b",
            capability_profile={
                "id": "prompt_only_ollama_structured",
                "allowed_controls": ["response_format"],
                "denied_controls": [],
                "provider_requirements": {"provider": "ollama_openai"},
            },
        )
        .with_controls({"response_format": response_format})
        .with_user("evaluate")
        .build()
    )

    assert request.extra_body == {"response_format": response_format}
    assert request.effective_controls["response_format"] == response_format
    assert request.dropped_controls == {}


def test_query_builder_drops_response_format_without_type_key() -> None:
    request = (
        QueryBuilder(
            endpoint_id="local_ollama",
            provider="ollama_openai",
            base_url="http://localhost:11434/v1",
            model="gemma4:e2b",
            capability_profile={
                "id": "prompt_only_ollama_structured",
                "allowed_controls": ["response_format"],
                "denied_controls": [],
                "provider_requirements": {"provider": "ollama_openai"},
            },
        )
        .with_controls({"response_format": {"json_schema": {"name": "x"}}})
        .with_user("evaluate")
        .build()
    )

    assert request.extra_body == {}
    assert "response_format" not in request.effective_controls
    assert request.dropped_controls["response_format"]["reason"] == "invalid_response_format"


def test_query_builder_drops_response_format_when_capability_profile_denies_it() -> None:
    request = (
        QueryBuilder(
            endpoint_id="local_ollama",
            provider="ollama_openai",
            base_url="http://localhost:11434/v1",
            model="gemma4:e2b",
            capability_profile={
                "id": "prompt_only_ollama",
                "allowed_controls": ["persona_text"],
                "denied_controls": [],
                "provider_requirements": {"provider": "ollama_openai"},
            },
        )
        .with_controls(
            {"response_format": {"type": "json_object"}}
        )
        .with_user("evaluate")
        .build()
    )

    assert request.extra_body == {}
    assert (
        request.dropped_controls["response_format"]["reason"]
        == "not_allowed_by_capability_profile"
    )


def test_role_contract_overrides_genome_when_layered_after() -> None:
    """Last-applied control wins. The runtime layers the role contract after
    the genome `control_requests`, so a contract response_format overrides any
    response_format the genome tried to set."""

    genome_format = {"type": "json_object"}
    contract_format = {
        "type": "json_schema",
        "json_schema": {"name": "grader_assessment", "schema": {"type": "object"}, "strict": True},
    }
    builder = QueryBuilder(
        endpoint_id="local_ollama",
        provider="ollama_openai",
        base_url="http://localhost:11434/v1",
        model="gemma4:e2b",
        capability_profile={
            "id": "prompt_only_ollama_structured",
            "allowed_controls": ["response_format"],
            "denied_controls": [],
            "provider_requirements": {"provider": "ollama_openai"},
        },
    )
    builder.with_controls({"response_format": genome_format})
    builder.with_controls({"response_format": contract_format})
    request = builder.with_user("evaluate").build()

    assert request.extra_body == {"response_format": contract_format}
    assert request.effective_controls["response_format"] == contract_format


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
