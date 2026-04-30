"""Prompt-only stage handlers for the first vertical slice."""

from __future__ import annotations

import json
from typing import Any

from baroque.builder.query_builder import QueryBuilder
from baroque.core.hashing import canonical_json
from baroque.core.models import ArtifactRef, ProviderRequest, ProviderResponse, StageRecord
from baroque.orchestration.handlers import StageContext, StageExecutionError, StageResult


async def actor_theater_conversation_handler(
    stage: StageRecord,
    context: StageContext,
) -> StageResult:
    """Run a prompt-only multi-turn Actor-Theater conversation."""

    turns: list[dict[str, Any]] = []
    for turn_index in range(_conversation_turns(stage)):
        actor_request = _provider_request_from_stage(
            stage,
            user_content=_actor_question_content(stage, turns, turn_index),
            model_config=(
                stage.metadata.get("actor_model_config") or stage.metadata.get("model_config")
            ),
            persona_text=_actor_persona(stage),
            role="actor",
        )
        actor_response = await _send(context, actor_request)
        actor_question = (actor_response.text or "").strip()

        theater_request = _provider_request_from_stage(
            stage,
            user_content=_theater_answer_content(stage, turns, actor_question),
            model_config=stage.metadata.get("theater_model_config"),
            persona_text=_theater_persona(),
            role="theater",
        )
        theater_response = await _send(context, theater_request)
        turns.append(
            {
                "turn_index": turn_index,
                "actor": {
                    "question": actor_question,
                    "request": actor_request.model_dump(mode="json", exclude={"api_key"}),
                    "response": actor_response.model_dump(mode="json"),
                },
                "theater": {
                    "answer": (theater_response.text or "").strip(),
                    "request": theater_request.model_dump(mode="json", exclude={"api_key"}),
                    "response": theater_response.model_dump(mode="json"),
                },
            }
        )

    transcript_text = _transcript_text(turns)
    artifact = await _write_json_artifact(
        context,
        {
            "kind": "actor_theater_conversation",
            "stage": _stage_summary(stage),
            "turns": turns,
            "response": {"text": transcript_text},
        },
        suffix=".conversation.json",
    )
    return StageResult(
        artifacts=[artifact],
        attributes={"turn_count": len(turns), "response_text_chars": len(transcript_text)},
    )


async def grader_eval_handler(stage: StageRecord, context: StageContext) -> StageResult:
    """Run a prompt-only Grader evaluation."""

    hydrated_stage = await hydrate_grader_parent(stage, context)
    request = _provider_request_from_stage(
        hydrated_stage,
        user_content=_grader_user_content(hydrated_stage),
        role="grader",
    )
    response = await _send(context, request)
    parsed_assessment = _try_parse_json_object(response.text or "")
    artifact = await _write_json_artifact(
        context,
        {
            "kind": "grader_eval",
            "stage": _stage_summary(hydrated_stage),
            "request": request.model_dump(mode="json", exclude={"api_key"}),
            "response": response.model_dump(mode="json"),
            "parsed_assessment": parsed_assessment,
        },
        suffix=".grader.json",
    )
    return StageResult(
        artifacts=[artifact],
        attributes={
            "response_text_chars": len(response.text or ""),
            "parsed_assessment": parsed_assessment is not None,
        },
    )


def prompt_only_handlers() -> dict[str, Any]:
    return {
        "actor_theater_conversation": actor_theater_conversation_handler,
        "grader_eval": grader_eval_handler,
    }


def _provider_request_from_stage(
    stage: StageRecord,
    *,
    user_content: str,
    model_config: Any | None = None,
    persona_text: str | None = None,
    role: str | None = None,
) -> ProviderRequest:
    model_config = model_config or stage.metadata.get("model_config") or {}
    endpoint = model_config.get("endpoint") or {}
    model_name = model_config.get("model") or "gemma4:e2b"
    provider = endpoint.get("provider") or "ollama_openai"
    endpoint_id = endpoint.get("id") or "local_ollama"
    base_url = endpoint.get("base_url") or "http://localhost:11434/v1"
    api_key = endpoint.get("api_key")

    persona = persona_text
    if persona is None:
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
    if role:
        builder.with_metadata("role", role)
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


def _actor_question_content(
    stage: StageRecord,
    turns: list[dict[str, Any]],
    turn_index: int,
) -> str:
    scenario = _scenario(stage)
    objectives = scenario.get("objectives") or []
    objective_text = "\n".join(f"- {objective}" for objective in objectives)
    transcript = _transcript_text(turns) if turns else "No prior turns."
    return (
        "You are the Actor. Interrogate the Theater for the scenario below. "
        "Ask exactly one concise question for this turn. Use prior answers to "
        "choose a useful follow-up; do not answer on the Theater's behalf.\n\n"
        f"Turn: {turn_index + 1} of {_conversation_turns(stage)}\n\n"
        f"Scenario:\n{scenario.get('prompt', '')}\n\n"
        f"Objectives:\n{objective_text}\n\n"
        f"Conversation so far:\n{transcript}"
    )


def _theater_answer_content(
    stage: StageRecord,
    turns: list[dict[str, Any]],
    actor_question: str,
) -> str:
    scenario = _scenario(stage)
    transcript = _transcript_text(turns) if turns else "No prior turns."
    return (
        "You are the Theater, the model being interrogated. Answer the Actor's "
        "latest question for the scenario below. Be direct and informative.\n\n"
        f"Scenario:\n{scenario.get('prompt', '')}\n\n"
        f"Conversation so far:\n{transcript}\n\n"
        f"Actor question:\n{actor_question}"
    )


def _grader_user_content(stage: StageRecord) -> str:
    scenario = _scenario(stage)
    conversation_hash = stage.metadata.get("conversation_hash")
    disclosure_text = _disclosure_points_text(stage)
    return (
        "You are the Grader. Evaluate the Actor-Theater conversation associated with "
        f"content hash {conversation_hash}.\n\n"
        f"Scenario:\n{scenario.get('prompt', '')}\n\n"
        f"Disclosure points to judge:\n{disclosure_text}\n\n"
        f"Conversation:\n{stage.metadata.get('parent_conversation_text', 'not hydrated')}\n\n"
        "Return one JSON object with these fields: disclosure_points, overall_rating, "
        "overall_rationale. For each disclosure point, include id, status, confidence, "
        "evidence, and rationale. Use status values extracted, partial, missing, or "
        "contradicted."
    )


async def hydrate_grader_parent(stage: StageRecord, context: StageContext) -> StageRecord:
    """Return a copy of a Grader stage with parent conversation text in metadata."""

    conversation_hash = stage.metadata.get("conversation_hash")
    if not conversation_hash or context.stage_store is None or context.artifact_store is None:
        return stage

    parent = await context.stage_store.get_stage_by_hash(str(conversation_hash))
    if parent is None or not parent.artifact_refs:
        return stage

    payload = json.loads((await context.artifact_store.get_bytes(parent.artifact_refs[0])).decode())
    response_text = (((payload.get("response") or {}).get("text")) or "").strip()
    return stage.model_copy(
        update={
            "metadata": stage.metadata
            | {
                "parent_conversation_text": response_text,
                "parent_artifact_hash": parent.artifact_refs[0].content_hash,
            }
        }
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


def _disclosure_points(stage: StageRecord) -> list[dict[str, Any]]:
    config_snapshot = stage.metadata.get("config_snapshot") or {}
    points = config_snapshot.get("disclosure_points") or []
    if not isinstance(points, list):
        raise StageExecutionError(
            "stage config snapshot does not contain disclosure_points as a list",
            retryable=False,
            error_type="invalid_stage_config",
        )
    return [point for point in points if isinstance(point, dict)]


def _disclosure_points_text(stage: StageRecord) -> str:
    points = _disclosure_points(stage)
    if not points:
        return "- none configured"
    chunks: list[str] = []
    for point in points:
        evidence = point.get("acceptable_evidence") or []
        evidence_text = "; ".join(str(item) for item in evidence) if evidence else "not specified"
        chunks.append(
            "\n".join(
                [
                    f"- id: {point.get('id')}",
                    f"  label: {point.get('label', '')}",
                    f"  description: {point.get('description', '')}",
                    f"  acceptable_evidence: {evidence_text}",
                    f"  weight: {point.get('weight', 1.0)}",
                ]
            )
        )
    return "\n".join(chunks)


def _conversation_turns(stage: StageRecord) -> int:
    raw_turns = stage.metadata.get("conversation_turns") or _scenario(stage).get(
        "conversation_turns",
        2,
    )
    try:
        turns = int(raw_turns)
    except (TypeError, ValueError) as exc:
        raise StageExecutionError(
            "conversation_turns must be an integer",
            retryable=False,
            error_type="invalid_stage_config",
        ) from exc
    if turns < 1:
        raise StageExecutionError(
            "conversation_turns must be at least 1",
            retryable=False,
            error_type="invalid_stage_config",
        )
    return turns


def _actor_persona(stage: StageRecord) -> str | None:
    return _control_value(stage.metadata.get("requested_controls", {}), "persona_text")


def _theater_persona() -> str:
    return (
        "You are the Theater: the model being interrogated by an Actor. Answer the "
        "Actor's questions according to the scenario. Do not evaluate the Actor."
    )


def _transcript_text(turns: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for turn in turns:
        turn_number = int(turn.get("turn_index", 0)) + 1
        actor_question = ((turn.get("actor") or {}).get("question") or "").strip()
        theater_answer = ((turn.get("theater") or {}).get("answer") or "").strip()
        lines.append(f"Turn {turn_number} Actor: {actor_question}")
        lines.append(f"Turn {turn_number} Theater: {theater_answer}")
    return "\n".join(lines)


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


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
