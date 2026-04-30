"""Prompt-only stage handlers for the first vertical slice."""

from __future__ import annotations

import json
from typing import Any

from baroque.builder.query_builder import QueryBuilder
from baroque.core.hashing import canonical_json
from baroque.core.models import ArtifactRef, ProviderRequest, ProviderResponse, StageRecord
from baroque.evolution.mutations import (
    GenomePatchOp,
    GenomePatchOperation,
    MutationOperatorKind,
    MutationProposal,
    apply_mutation_proposal,
)
from baroque.orchestration.handlers import StageContext, StageExecutionError, StageResult
from baroque.ranking.assessments import (
    aggregate_assessments,
    assessment_record_from_grader_artifact,
)


async def actor_turn_handler(
    stage: StageRecord,
    context: StageContext,
) -> StageResult:
    """Run one prompt-only Actor question stage."""

    previous_transcript = await _previous_transcript(stage, context, "actor turn")
    turn_index = _turn_index(stage)
    actor_request = _provider_request_from_stage(
        stage,
        user_content=_actor_question_content(stage, previous_transcript, turn_index),
        model_config=stage.metadata.get("actor_model_config") or stage.metadata.get("model_config"),
        persona_text=_actor_persona(stage),
        role="actor",
    )
    actor_response = await _send(context, actor_request)
    actor_question = (actor_response.text or "").strip()
    artifact = await _write_json_artifact(
        context,
        {
            "kind": "actor_turn",
            "stage": _stage_summary(stage),
            "turn_index": turn_index,
            "transcript_before": previous_transcript,
            "question": actor_question,
            "request": actor_request.model_dump(mode="json", exclude={"api_key"}),
            "response": actor_response.model_dump(mode="json"),
        },
        suffix=".actor_turn.json",
    )
    return StageResult(
        artifacts=[artifact],
        attributes={
            "turn_index": turn_index,
            "question_text_chars": len(actor_question),
            "response_text_chars": len(actor_response.text or ""),
        },
    )


async def theater_turn_handler(stage: StageRecord, context: StageContext) -> StageResult:
    """Run one prompt-only Theater answer stage."""

    actor_payload = await _load_single_parent_payload(stage, context, "theater turn")
    turn_index = _turn_index(stage)
    previous_transcript = str(actor_payload.get("transcript_before") or "")
    actor_question = str(actor_payload.get("question") or "").strip()
    theater_request = _provider_request_from_stage(
        stage,
        user_content=_theater_answer_content(stage, previous_transcript, actor_question),
        model_config=(
            stage.metadata.get("theater_model_config") or stage.metadata.get("model_config")
        ),
        persona_text=_theater_persona(),
        role="theater",
    )
    theater_response = await _send(context, theater_request)
    theater_answer = (theater_response.text or "").strip()
    transcript_text = _append_turn_transcript(
        previous_transcript,
        turn_index,
        actor_question,
        theater_answer,
    )
    artifact = await _write_json_artifact(
        context,
        {
            "kind": "theater_turn",
            "stage": _stage_summary(stage),
            "turn_index": turn_index,
            "transcript_before": previous_transcript,
            "transcript_text": transcript_text,
            "actor_artifact": actor_payload.get("_artifact_ref"),
            "actor_question": actor_question,
            "answer": theater_answer,
            "request": theater_request.model_dump(mode="json", exclude={"api_key"}),
            "response": theater_response.model_dump(mode="json"),
        },
        suffix=".theater_turn.json",
    )
    return StageResult(
        artifacts=[artifact],
        attributes={
            "turn_index": turn_index,
            "answer_text_chars": len(theater_answer),
            "transcript_text_chars": len(transcript_text),
        },
    )


async def conversation_transcript_handler(
    stage: StageRecord,
    context: StageContext,
) -> StageResult:
    """Assemble one Actor-Theater transcript from completed per-call stages."""

    parent_payloads = [
        await _load_parent_payload(parent_hash, context, "conversation transcript")
        for parent_hash in stage.parent_hashes
    ]
    turns = _turns_from_parent_payloads(parent_payloads)
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
    assessment_record = (
        assessment_record_from_grader_artifact(
            hydrated_stage,
            {"parsed_assessment": parsed_assessment},
        )
        if parsed_assessment is not None
        else None
    )
    artifact = await _write_json_artifact(
        context,
        {
            "kind": "grader_eval",
            "stage": _stage_summary(hydrated_stage),
            "request": request.model_dump(mode="json", exclude={"api_key"}),
            "response": response.model_dump(mode="json"),
            "parsed_assessment": parsed_assessment,
            "assessment_record": (
                assessment_record.model_dump(mode="json") if assessment_record is not None else None
            ),
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


async def assessment_aggregate_handler(stage: StageRecord, context: StageContext) -> StageResult:
    """Aggregate one rollout's plural Grader assessments."""

    if context.stage_store is None or context.artifact_store is None:
        raise StageExecutionError(
            "assessment aggregation requires stage and artifact stores",
            retryable=False,
            error_type="missing_runtime_store",
        )

    records = []
    parent_artifacts: list[dict[str, Any]] = []
    for parent_hash in stage.parent_hashes:
        parent = await context.stage_store.get_stage_by_hash(parent_hash)
        if parent is None or not parent.artifact_refs:
            raise StageExecutionError(
                f"assessment parent artifact is unavailable: {parent_hash}",
                retryable=True,
                error_type="missing_parent_artifact",
            )
        artifact = parent.artifact_refs[0]
        payload = json.loads((await context.artifact_store.get_bytes(artifact)).decode())
        payload["artifact_hash"] = artifact.content_hash
        records.append(assessment_record_from_grader_artifact(parent, payload))
        parent_artifacts.append(
            {
                "stage_hash": parent.content_hash,
                "artifact_hash": artifact.content_hash,
            }
        )

    aggregate = aggregate_assessments(records)
    artifact = await _write_json_artifact(
        context,
        {
            "kind": "assessment_aggregate",
            "stage": _stage_summary(stage),
            "parent_artifacts": parent_artifacts,
            "assessment_records": [record.model_dump(mode="json") for record in records],
            "aggregate": aggregate.model_dump(mode="json"),
        },
        suffix=".assessment_aggregate.json",
    )
    return StageResult(
        artifacts=[artifact],
        attributes={
            "assessment_count": aggregate.assessment_count,
            "disclosure_point_count": len(aggregate.disclosure_points),
            "overall_rating_mean": aggregate.overall_rating_mean,
        },
    )


async def mutation_proposal_handler(stage: StageRecord, context: StageContext) -> StageResult:
    """Propose a deterministic prompt-only mutation from assessment evidence."""

    aggregate_payload = await _load_single_parent_payload(stage, context, "mutation proposal")
    aggregate = aggregate_payload.get("aggregate") or {}
    actor_genome = _actor_genome(stage)
    actor_genome_id = _actor_genome_id(stage)
    proposal = _deterministic_prompt_mutation_proposal(
        stage,
        aggregate,
        actor_genome,
        actor_genome_id,
    )
    artifact = await _write_json_artifact(
        context,
        {
            "kind": "mutation_proposal",
            "stage": _stage_summary(stage),
            "parent_artifact": aggregate_payload.get("_artifact_ref"),
            "aggregate": aggregate,
            "proposal": proposal.model_dump(mode="json"),
            "proposal_hash": proposal.deterministic_hash(),
        },
        suffix=".mutation_proposal.json",
    )
    return StageResult(
        artifacts=[artifact],
        attributes={
            "operation_count": len(proposal.operations),
            "proposal_hash": proposal.deterministic_hash(),
        },
    )


async def mutation_application_handler(stage: StageRecord, context: StageContext) -> StageResult:
    """Apply a mutation proposal to the configured parent Actor genome."""

    proposal_payload = await _load_single_parent_payload(stage, context, "mutation application")
    proposal_data = proposal_payload.get("proposal") or {}
    if not isinstance(proposal_data, dict):
        raise StageExecutionError(
            "mutation proposal artifact does not contain a proposal object",
            retryable=False,
            error_type="invalid_parent_artifact",
        )
    proposal = MutationProposal.model_validate(proposal_data)
    application = apply_mutation_proposal(
        proposal,
        _actor_genome(stage),
        child_genome_id=stage.metadata.get("child_genome_id"),
    )
    artifact = await _write_json_artifact(
        context,
        {
            "kind": "mutation_application",
            "stage": _stage_summary(stage),
            "parent_artifact": proposal_payload.get("_artifact_ref"),
            "proposal": proposal.model_dump(mode="json"),
            "application": application.model_dump(mode="json"),
            "application_hash": application.deterministic_hash(),
        },
        suffix=".mutation_application.json",
    )
    return StageResult(
        artifacts=[artifact],
        attributes={
            "applied": application.applied,
            "child_genome_id": application.child_genome_id,
            "error_count": len(application.errors),
        },
    )


def prompt_only_handlers() -> dict[str, Any]:
    return {
        "actor_turn": actor_turn_handler,
        "theater_turn": theater_turn_handler,
        "conversation_transcript": conversation_transcript_handler,
        "grader_eval": grader_eval_handler,
        "assessment_aggregate": assessment_aggregate_handler,
        "mutation_proposal": mutation_proposal_handler,
        "mutation_application": mutation_application_handler,
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
    transcript_text: str,
    turn_index: int,
) -> str:
    scenario = _scenario(stage)
    objectives = scenario.get("objectives") or []
    objective_text = "\n".join(f"- {objective}" for objective in objectives)
    transcript = transcript_text if transcript_text else "No prior turns."
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
    transcript_text: str,
    actor_question: str,
) -> str:
    scenario = _scenario(stage)
    transcript = transcript_text if transcript_text else "No prior turns."
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


async def _load_single_parent_payload(
    stage: StageRecord,
    context: StageContext,
    label: str,
) -> dict[str, Any]:
    if len(stage.parent_hashes) != 1:
        raise StageExecutionError(
            f"{label} requires exactly one parent",
            retryable=False,
            error_type="invalid_stage_config",
        )
    return await _load_parent_payload(stage.parent_hashes[0], context, label)


async def _load_parent_payload(
    parent_hash: str,
    context: StageContext,
    label: str,
) -> dict[str, Any]:
    if context.stage_store is None or context.artifact_store is None:
        raise StageExecutionError(
            f"{label} requires stage and artifact stores",
            retryable=False,
            error_type="missing_runtime_store",
        )

    parent = await context.stage_store.get_stage_by_hash(parent_hash)
    if parent is None or not parent.artifact_refs:
        raise StageExecutionError(
            f"{label} parent artifact is unavailable: {parent_hash}",
            retryable=True,
            error_type="missing_parent_artifact",
        )
    artifact = parent.artifact_refs[0]
    payload = json.loads((await context.artifact_store.get_bytes(artifact)).decode())
    if not isinstance(payload, dict):
        raise StageExecutionError(
            f"{label} parent artifact is not a JSON object",
            retryable=False,
            error_type="invalid_parent_artifact",
        )
    payload["_artifact_ref"] = artifact.model_dump(mode="json")
    return payload


async def _previous_transcript(
    stage: StageRecord,
    context: StageContext,
    label: str,
) -> str:
    if not stage.parent_hashes:
        return ""
    payload = await _load_single_parent_payload(stage, context, label)
    response_text = ((payload.get("response") or {}).get("text")) or ""
    return str(payload.get("transcript_text") or response_text)


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


def _actor_genome(stage: StageRecord) -> dict[str, Any]:
    config_snapshot = stage.metadata.get("config_snapshot") or {}
    genome = config_snapshot.get("actor_genome") or {}
    if not isinstance(genome, dict):
        raise StageExecutionError(
            "stage config snapshot does not contain actor_genome as a mapping",
            retryable=False,
            error_type="invalid_stage_config",
        )
    return genome


def _actor_genome_id(stage: StageRecord) -> str:
    config_snapshot = stage.metadata.get("config_snapshot") or {}
    genome_id = config_snapshot.get("actor_genome_id")
    if not genome_id:
        raise StageExecutionError(
            "stage config snapshot does not contain actor_genome_id",
            retryable=False,
            error_type="invalid_stage_config",
        )
    return str(genome_id)


def _actor_id(stage: StageRecord) -> str | None:
    config_snapshot = stage.metadata.get("config_snapshot") or {}
    actor_id = config_snapshot.get("actor_id")
    return str(actor_id) if actor_id is not None else None


def _deterministic_prompt_mutation_proposal(
    stage: StageRecord,
    aggregate: dict[str, Any],
    actor_genome: dict[str, Any],
    actor_genome_id: str,
) -> MutationProposal:
    current_persona = _genome_persona(actor_genome)
    weakest_points = _weakest_disclosure_points(aggregate)
    focus_text = _mutation_focus_text(weakest_points)
    mutated_persona = (
        f"{current_persona.strip()}\n\n"
        "Mutation note: In the next run, ask one concise follow-up at a time and "
        f"prioritize extracting: {focus_text}."
    )
    return MutationProposal(
        parent_genome_id=actor_genome_id,
        target_agent_id=_actor_id(stage),
        operator=MutationOperatorKind.HAND_AUTHORED,
        operations=[
            GenomePatchOperation(
                op=GenomePatchOp.REPLACE,
                path="/control_requests/persona_text/value",
                value=mutated_persona,
            )
        ],
        rationale=(
            "Deterministic prompt-only baseline mutation from disclosure aggregate. "
            f"Weakest disclosure targets: {focus_text}."
        ),
        assessment_refs=[stage.parent_hashes[0]] if stage.parent_hashes else [],
        author={"kind": "deterministic_baseline", "stage_type": stage.stage_type},
        metadata={
            "rollout_index": stage.metadata.get("rollout_index"),
            "conversation_hash": stage.metadata.get("conversation_hash"),
        },
    )


def _genome_persona(actor_genome: dict[str, Any]) -> str:
    value = ((actor_genome.get("control_requests") or {}).get("persona_text") or {}).get("value")
    return str(value or "You are an Actor who asks careful, specific follow-up questions.")


def _weakest_disclosure_points(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    points = aggregate.get("disclosure_points") or []
    if not isinstance(points, list):
        return []
    valid_points = [point for point in points if isinstance(point, dict)]
    return sorted(
        valid_points,
        key=lambda point: (
            float(point.get("mean_score", 0.0)),
            float(point.get("partial_or_extracted_rate", 0.0)),
            str(point.get("point_id", "")),
        ),
    )[:2]


def _mutation_focus_text(points: list[dict[str, Any]]) -> str:
    if not points:
        return "the configured disclosure points"
    labels = [str(point.get("label") or point.get("point_id")) for point in points]
    return ", ".join(labels)


def _turn_index(stage: StageRecord) -> int:
    raw_turn = stage.metadata.get("turn_index")
    if raw_turn is None:
        raise StageExecutionError(
            "turn stage metadata does not contain turn_index",
            retryable=False,
            error_type="invalid_stage_config",
        )
    try:
        turn_index = int(raw_turn)
    except (TypeError, ValueError) as exc:
        raise StageExecutionError(
            "turn_index must be an integer",
            retryable=False,
            error_type="invalid_stage_config",
        ) from exc
    if turn_index < 0:
        raise StageExecutionError(
            "turn_index must be nonnegative",
            retryable=False,
            error_type="invalid_stage_config",
        )
    return turn_index


def _append_turn_transcript(
    previous_transcript: str,
    turn_index: int,
    actor_question: str,
    theater_answer: str,
) -> str:
    prior = previous_transcript.strip()
    current = "\n".join(
        [
            f"Turn {turn_index + 1} Actor: {actor_question}",
            f"Turn {turn_index + 1} Theater: {theater_answer}",
        ]
    )
    return f"{prior}\n{current}" if prior else current


def _turns_from_parent_payloads(parent_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns_by_index: dict[int, dict[str, Any]] = {}
    for payload in parent_payloads:
        raw_turn_index = payload.get("turn_index")
        if raw_turn_index is None:
            continue
        turn_index = int(raw_turn_index)
        turn = turns_by_index.setdefault(turn_index, {"turn_index": turn_index})
        if payload.get("kind") == "actor_turn":
            turn["actor"] = {
                "question": str(payload.get("question") or ""),
                "artifact": payload.get("_artifact_ref"),
            }
        elif payload.get("kind") == "theater_turn":
            turn.setdefault("actor", {"question": str(payload.get("actor_question") or "")})
            turn["theater"] = {
                "answer": str(payload.get("answer") or ""),
                "artifact": payload.get("_artifact_ref"),
            }
    return [turns_by_index[index] for index in sorted(turns_by_index)]


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
