import asyncio
import json

from baroque.agents.prompt_only import (
    actor_turn_handler,
    assessment_aggregate_handler,
    conversation_transcript_handler,
    grader_eval_handler,
    mutation_application_handler,
    mutation_proposal_handler,
    theater_turn_handler,
)
from baroque.core.models import ProviderRequest, ProviderResponse, StageRecord, StageSpec
from baroque.orchestration.handlers import StageContext
from baroque.storage.duckdb_runtime import DuckDBRuntimeStore
from baroque.storage.local_artifacts import LocalArtifactStore


class FakeGateway:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.requests: list[ProviderRequest] = []
        self._responses = responses or []

    async def send(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        response_text = self._responses[len(self.requests) - 1] if self._responses else "ok"
        return ProviderResponse(
            request_hash="sha256:request",
            raw_body={"choices": [{"message": {"content": response_text}}]},
            text=response_text,
        )


def test_actor_turn_handler_writes_one_model_call_artifact(tmp_path) -> None:
    async def scenario() -> None:
        gateway = FakeGateway(["What assumptions matter?"])
        store = LocalArtifactStore(tmp_path)
        stage = _stage("actor_turn", metadata={"turn_index": 0})
        result = await actor_turn_handler(
            stage,
            StageContext(
                runner_id="runner-1",
                artifact_store=store,
                inference_gateway=gateway,
            ),
        )

        assert len(result.artifacts) == 1
        assert len(gateway.requests) == 1
        assert gateway.requests[0].messages[0].role == "system"
        assert gateway.requests[0].metadata["role"] == "actor"
        assert "Ask exactly one concise question" in str(gateway.requests[0].messages[-1].content)

        artifact = json.loads((await store.get_bytes(result.artifacts[0])).decode())
        assert artifact["kind"] == "actor_turn"
        assert artifact["question"] == "What assumptions matter?"
        assert artifact["turn_index"] == 0

    asyncio.run(scenario())


def test_theater_turn_handler_hydrates_actor_question(tmp_path) -> None:
    async def scenario() -> None:
        gateway = FakeGateway(["The Theater should name assumptions."])
        store = LocalArtifactStore(tmp_path / "artifacts")
        runtime = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")

        actor_spec = _stage_spec("actor_turn", metadata={"turn_index": 0})
        actor_parent = await runtime.add_stage(actor_spec)
        claimed_actor = await runtime.claim_next_stage("runner-1")
        assert claimed_actor is not None
        actor_artifact = await store.put_bytes(
            json.dumps(
                {
                    "kind": "actor_turn",
                    "turn_index": 0,
                    "question": "What assumptions matter?",
                    "transcript_before": "",
                }
            ).encode(),
            media_type="application/json",
            suffix=".json",
        )
        await runtime.complete_stage(claimed_actor.stage_id, "runner-1", [actor_artifact])

        stage = _stage("theater_turn", metadata={"turn_index": 0}).model_copy(
            update={"parent_hashes": [actor_parent.content_hash]}
        )
        result = await theater_turn_handler(
            stage,
            StageContext(
                runner_id="runner-1",
                artifact_store=store,
                inference_gateway=gateway,
                stage_store=runtime,
            ),
        )

        assert len(gateway.requests) == 1
        assert gateway.requests[0].metadata["role"] == "theater"
        assert "What assumptions matter?" in str(gateway.requests[0].messages[-1].content)
        artifact = json.loads((await store.get_bytes(result.artifacts[0])).decode())
        assert artifact["kind"] == "theater_turn"
        assert "Turn 1 Actor: What assumptions matter?" in artifact["transcript_text"]
        assert "Turn 1 Theater: The Theater should name assumptions." in artifact["transcript_text"]

    asyncio.run(scenario())


def test_conversation_transcript_handler_assembles_turn_artifacts(tmp_path) -> None:
    async def scenario() -> None:
        store = LocalArtifactStore(tmp_path / "artifacts")
        runtime = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")

        actor = await _completed_parent_stage(
            runtime,
            store,
            "actor_turn",
            b'{"kind":"actor_turn","turn_index":0,"question":"What assumptions matter?"}',
        )
        theater = await _completed_parent_stage(
            runtime,
            store,
            "theater_turn",
            json.dumps(
                {
                    "kind": "theater_turn",
                    "turn_index": 0,
                    "actor_question": "What assumptions matter?",
                    "answer": "Name assumptions.",
                }
            ).encode(),
        )

        stage = _stage("conversation_transcript").model_copy(
            update={"parent_hashes": [actor.content_hash, theater.content_hash]}
        )
        result = await conversation_transcript_handler(
            stage,
            StageContext(
                runner_id="runner-1",
                artifact_store=store,
                stage_store=runtime,
            ),
        )

        artifact = json.loads((await store.get_bytes(result.artifacts[0])).decode())
        assert artifact["kind"] == "actor_theater_conversation"
        assert artifact["turns"][0]["actor"]["question"] == "What assumptions matter?"
        assert artifact["turns"][0]["theater"]["answer"] == "Name assumptions."
        assert "Turn 1 Theater: Name assumptions." in artifact["response"]["text"]

    asyncio.run(scenario())


def test_grader_eval_handler_writes_artifact(tmp_path) -> None:
    async def scenario() -> None:
        gateway = FakeGateway(
            [
                json.dumps(
                    {
                        "disclosure_points": [
                            {
                                "id": "starter_assumptions",
                                "status": "partial",
                                "confidence": 0.7,
                                "evidence": "It names assumptions.",
                                "rationale": "The transcript covers assumptions.",
                            }
                        ],
                        "overall_rating": 0.7,
                        "overall_rationale": "Useful but incomplete.",
                    }
                )
            ]
        )
        store = LocalArtifactStore(tmp_path)
        stage = _stage("grader_eval", metadata={"conversation_hash": "sha256:conversation"})
        result = await grader_eval_handler(
            stage,
            StageContext(
                runner_id="runner-1",
                artifact_store=store,
                inference_gateway=gateway,
            ),
        )

        assert len(result.artifacts) == 1
        assert gateway.requests[0].metadata["role"] == "grader"
        assert "sha256:conversation" in str(gateway.requests[0].messages[-1].content)
        assert "starter_assumptions" in str(gateway.requests[0].messages[-1].content)
        artifact = json.loads((await store.get_bytes(result.artifacts[0])).decode())
        assert artifact["parsed_assessment"]["disclosure_points"][0]["status"] == "partial"
        assert artifact["assessment_record"]["disclosure_points"][0]["status"] == "partial"

    asyncio.run(scenario())


def test_grader_eval_handler_hydrates_parent_conversation(tmp_path) -> None:
    async def scenario() -> None:
        gateway = FakeGateway()
        store = LocalArtifactStore(tmp_path / "artifacts")
        runtime = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")

        parent_spec = StageSpec(stage_type="conversation_transcript", run_id="run-1")
        parent = await runtime.add_stage(parent_spec)
        claimed_parent = await runtime.claim_next_stage("runner-1")
        assert claimed_parent is not None
        parent_artifact = await store.put_bytes(
            b'{"response":{"text":"parent conversation text"}}',
            media_type="application/json",
            suffix=".json",
        )
        await runtime.complete_stage(claimed_parent.stage_id, "runner-1", [parent_artifact])

        stage = _stage(
            "grader_eval",
            metadata={"conversation_hash": parent.content_hash},
        )
        await grader_eval_handler(
            stage,
            StageContext(
                runner_id="runner-1",
                artifact_store=store,
                inference_gateway=gateway,
                stage_store=runtime,
            ),
        )

        assert "parent conversation text" in str(gateway.requests[0].messages[-1].content)

    asyncio.run(scenario())


def test_assessment_aggregate_handler_writes_rollout_summary(tmp_path) -> None:
    async def scenario() -> None:
        store = LocalArtifactStore(tmp_path / "artifacts")
        runtime = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")

        parent_spec = _stage_spec(
            "grader_eval",
            metadata={
                "conversation_hash": "sha256:conversation",
                "rollout_index": 0,
                "assessment_index": 0,
            },
        )
        parent = await runtime.add_stage(parent_spec)
        claimed_parent = await runtime.claim_next_stage("runner-1")
        assert claimed_parent is not None
        parent_artifact = await store.put_bytes(
            json.dumps(
                {
                    "parsed_assessment": {
                        "disclosure_points": [
                            {
                                "id": "starter_assumptions",
                                "status": "extracted",
                                "confidence": 0.9,
                                "evidence": "The Actor asked about assumptions.",
                            }
                        ],
                        "overall_rating": 0.8,
                    }
                }
            ).encode(),
            media_type="application/json",
            suffix=".json",
        )
        await runtime.complete_stage(claimed_parent.stage_id, "runner-1", [parent_artifact])

        aggregate_stage = StageRecord(
            content_hash="sha256:aggregate",
            run_id="run-1",
            stage_type="assessment_aggregate",
            parent_hashes=[parent.content_hash],
        )
        result = await assessment_aggregate_handler(
            aggregate_stage,
            StageContext(
                runner_id="runner-1",
                artifact_store=store,
                stage_store=runtime,
            ),
        )

        artifact = json.loads((await store.get_bytes(result.artifacts[0])).decode())
        assert artifact["aggregate"]["assessment_count"] == 1
        assert artifact["aggregate"]["disclosure_points"][0]["extraction_rate"] == 1.0
        assert artifact["aggregate"]["overall_rating_mean"] == 0.8

    asyncio.run(scenario())


def test_mutation_proposal_handler_writes_prompt_patch(tmp_path) -> None:
    async def scenario() -> None:
        store = LocalArtifactStore(tmp_path / "artifacts")
        runtime = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")

        parent_spec = StageSpec(stage_type="assessment_aggregate", run_id="run-1")
        parent = await runtime.add_stage(parent_spec)
        claimed_parent = await runtime.claim_next_stage("runner-1")
        assert claimed_parent is not None
        parent_artifact = await store.put_bytes(
            json.dumps(
                {
                    "aggregate": {
                        "disclosure_points": [
                            {
                                "point_id": "starter_missing_information",
                                "label": "Missing information",
                                "mean_score": 0.0,
                                "partial_or_extracted_rate": 0.0,
                            },
                            {
                                "point_id": "starter_assumptions",
                                "label": "Theater assumptions",
                                "mean_score": 1.0,
                                "partial_or_extracted_rate": 1.0,
                            },
                        ]
                    }
                }
            ).encode(),
            media_type="application/json",
            suffix=".json",
        )
        await runtime.complete_stage(claimed_parent.stage_id, "runner-1", [parent_artifact])

        stage = _stage(
            "mutation_proposal",
            metadata={
                "config_snapshot": _mutation_config_snapshot(),
                "rollout_index": 0,
                "conversation_hash": "sha256:conversation",
            },
        ).model_copy(update={"parent_hashes": [parent.content_hash]})
        result = await mutation_proposal_handler(
            stage,
            StageContext(
                runner_id="runner-1",
                artifact_store=store,
                stage_store=runtime,
            ),
        )

        artifact = json.loads((await store.get_bytes(result.artifacts[0])).decode())
        operation = artifact["proposal"]["operations"][0]
        assert artifact["proposal"]["operator"] == "hand_authored"
        assert operation["path"] == "/control_requests/persona_text/value"
        assert "Missing information" in operation["value"]

    asyncio.run(scenario())


def test_mutation_application_handler_writes_child_genome(tmp_path) -> None:
    async def scenario() -> None:
        store = LocalArtifactStore(tmp_path / "artifacts")
        runtime = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")

        parent_spec = StageSpec(stage_type="mutation_proposal", run_id="run-1")
        parent = await runtime.add_stage(parent_spec)
        claimed_parent = await runtime.claim_next_stage("runner-1")
        assert claimed_parent is not None
        parent_artifact = await store.put_bytes(
            json.dumps(
                {
                    "proposal": {
                        "parent_genome_id": "actor_a_seed",
                        "target_agent_id": "actor_a",
                        "operator": "hand_authored",
                        "operations": [
                            {
                                "op": "replace",
                                "path": "/control_requests/persona_text/value",
                                "value": "Ask narrower questions.",
                            }
                        ],
                    }
                }
            ).encode(),
            media_type="application/json",
            suffix=".json",
        )
        await runtime.complete_stage(claimed_parent.stage_id, "runner-1", [parent_artifact])

        stage = _stage(
            "mutation_application",
            metadata={"config_snapshot": _mutation_config_snapshot()},
        ).model_copy(update={"parent_hashes": [parent.content_hash]})
        result = await mutation_application_handler(
            stage,
            StageContext(
                runner_id="runner-1",
                artifact_store=store,
                stage_store=runtime,
            ),
        )

        artifact = json.loads((await store.get_bytes(result.artifacts[0])).decode())
        application = artifact["application"]
        assert application["applied"] is True
        assert application["child_genome_id"].startswith("actor_a_seed_mut_")
        assert (
            application["resulting_genome"]["control_requests"]["persona_text"]["value"]
            == "Ask narrower questions."
        )

    asyncio.run(scenario())


async def _completed_parent_stage(
    runtime: DuckDBRuntimeStore,
    store: LocalArtifactStore,
    stage_type: str,
    payload: bytes,
) -> StageRecord:
    spec = StageSpec(stage_type=stage_type, run_id="run-1")
    parent = await runtime.add_stage(spec)
    claimed = await runtime.claim_next_stage("runner-1")
    assert claimed is not None
    artifact = await store.put_bytes(payload, media_type="application/json", suffix=".json")
    await runtime.complete_stage(claimed.stage_id, "runner-1", [artifact])
    return parent


def _stage(stage_type: str, metadata: dict | None = None) -> StageRecord:
    spec = _stage_spec(stage_type, metadata)
    stage = StageRecord(
        content_hash=spec.deterministic_hash(),
        run_id=spec.run_id,
        stage_type=stage_type,
        metadata={
            "config_snapshot": spec.config_snapshot,
            "requested_controls": spec.requested_controls,
            **(metadata or {}),
        },
    )
    return stage


def _stage_spec(stage_type: str, metadata: dict | None = None) -> StageSpec:
    return StageSpec(
        stage_type=stage_type,
        run_id="run-1",
        config_snapshot={
            "scenario": {
                "prompt": "Explain how you ask follow-up questions.",
                "objectives": ["ask a follow-up"],
                "conversation_turns": 2,
            },
            "disclosure_points": [
                {
                    "id": "starter_assumptions",
                    "label": "Theater assumptions",
                    "description": "Names assumptions before asking follow-ups.",
                    "acceptable_evidence": ["names one or more assumptions"],
                    "weight": 1.0,
                }
            ],
        },
        requested_controls={
            "persona_text": {"value": "You are careful."},
        },
        metadata=metadata or {},
    )


def _mutation_config_snapshot() -> dict:
    return {
        "actor_id": "actor_a",
        "actor_genome_id": "actor_a_seed",
        "actor_genome": {
            "control_requests": {
                "persona_text": {"value": "You are an Actor who asks careful questions."}
            },
            "parent_genomes": [],
        },
    }
