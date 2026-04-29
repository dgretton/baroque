"""Prompt-only stage handlers for the first vertical slice."""

from __future__ import annotations

from typing import Any

from baroque.builder.query_builder import QueryBuilder
from baroque.core.hashing import canonical_json
from baroque.core.models import ArtifactRef, ProviderRequest, ProviderResponse, StageRecord
from baroque.orchestration.handlers import StageContext, StageExecutionError, StageResult


async def actor_theater_conversation_handler(
    stage: StageRecord,
    context: StageContext,
) -> StageResult:
    """Run a one-turn prompt-only Actor-Theater conversation."""

    request = _provider_request_from_stage(stage, user_content=_actor_user_content(stage))
    response = await _send(context, request)
    artifact = await _write_json_artifact(
        context,
        {
            "kind": "actor_theater_conversation",
            "stage": _stage_summary(stage),
            "request": request.model_dump(mode="json", exclude={"api_key"}),
            "response": response.model_dump(mode="json"),
        },
        suffix=".conversation.json",
    )
    return StageResult(
        artifacts=[artifact],
        attributes={"response_text_chars": len(response.text or "")},
    )


async def grader_eval_handler(stage: StageRecord, context: StageContext) -> StageResult:
    """Run a prompt-only Grader evaluation.

    The first vertical slice stores parent artifact references on the completed
    parent stage, but the runner does not yet hydrate them into child handlers.
    Until that is added, the Grader request includes the conversation hash and
    scenario. This keeps the stage runnable while preserving the causal link.
    """

    request = _provider_request_from_stage(stage, user_content=_grader_user_content(stage))
    response = await _send(context, request)
    artifact = await _write_json_artifact(
        context,
        {
            "kind": "grader_eval",
            "stage": _stage_summary(stage),
            "request": request.model_dump(mode="json", exclude={"api_key"}),
            "response": response.model_dump(mode="json"),
        },
        suffix=".grader.json",
    )
    return StageResult(
        artifacts=[artifact],
        attributes={"response_text_chars": len(response.text or "")},
    )


def prompt_only_handlers() -> dict[str, Any]:
    return {
        "actor_theater_conversation": actor_theater_conversation_handler,
        "grader_eval": grader_eval_handler,
    }


def _provider_request_from_stage(stage: StageRecord, *, user_content: str) -> ProviderRequest:
    model_config = stage.metadata.get("model_config") or {}
    endpoint = model_config.get("endpoint") or {}
    model_name = model_config.get("model") or "gemma4:e2b"
    provider = endpoint.get("provider") or "ollama_openai"
    endpoint_id = endpoint.get("id") or "local_ollama"
    base_url = endpoint.get("base_url") or "http://localhost:11434/v1"
    api_key = endpoint.get("api_key")

    persona = _control_value(stage.metadata.get("requested_controls", {}), "persona_text")
    builder = QueryBuilder(
        endpoint_id=endpoint_id,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model_name,
    )
    if persona:
        builder.with_persona(persona)
    builder.with_user(user_content)
    builder.with_metadata("stage_id", stage.stage_id)
    builder.with_metadata("content_hash", stage.content_hash)
    return builder.build()


async def _send(context: StageContext, request: ProviderRequest) -> ProviderResponse:
    if context.inference_gateway is None:
        raise StageExecutionError(
            "stage requires an inference gateway",
            retryable=False,
            error_type="missing_inference_gateway",
        )
    return await context.inference_gateway.send(request)


async def _write_json_artifact(
    context: StageContext,
    payload: dict[str, Any],
    *,
    suffix: str,
) -> ArtifactRef:
    if context.artifact_store is None:
        raise StageExecutionError(
            "stage requires an artifact store",
            retryable=False,
            error_type="missing_artifact_store",
        )
    return await context.artifact_store.put_bytes(
        canonical_json(payload).encode("utf-8"),
        media_type="application/json",
        suffix=suffix,
    )


def _actor_user_content(stage: StageRecord) -> str:
    scenario = _scenario(stage)
    objectives = scenario.get("objectives") or []
    objective_text = "\n".join(f"- {objective}" for objective in objectives)
    return (
        "You are the Actor. Interrogate the Theater for the scenario below.\n\n"
        f"Scenario:\n{scenario.get('prompt', '')}\n\n"
        f"Objectives:\n{objective_text}"
    )


def _grader_user_content(stage: StageRecord) -> str:
    scenario = _scenario(stage)
    conversation_hash = stage.metadata.get("conversation_hash")
    return (
        "You are the Grader. Evaluate the Actor-Theater conversation associated with "
        f"content hash {conversation_hash}.\n\n"
        f"Scenario:\n{scenario.get('prompt', '')}\n\n"
        "Return JSON with rating and justification fields."
    )


def _scenario(stage: StageRecord) -> dict[str, Any]:
    config_snapshot = stage.metadata.get("config_snapshot") or {}
    scenario = config_snapshot.get("scenario") or {}
    if not isinstance(scenario, dict):
        raise StageExecutionError(
            "stage config snapshot does not contain a scenario mapping",
            retryable=False,
            error_type="invalid_stage_config",
        )
    return scenario


def _control_value(controls: dict[str, Any], name: str) -> str | None:
    value = controls.get(name)
    if isinstance(value, dict):
        inner = value.get("value")
        return str(inner) if inner is not None else None
    return str(value) if value is not None else None


def _stage_summary(stage: StageRecord) -> dict[str, Any]:
    return {
        "stage_id": stage.stage_id,
        "content_hash": stage.content_hash,
        "run_id": stage.run_id,
        "sample_id": stage.sample_id,
        "stage_type": stage.stage_type,
        "attempt": stage.attempt,
    }
