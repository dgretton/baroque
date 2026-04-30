import asyncio
import json

from baroque.agents.prompt_only import actor_theater_conversation_handler, grader_eval_handler
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


def test_actor_theater_conversation_handler_writes_artifact(tmp_path) -> None:
    async def scenario() -> None:
        gateway = FakeGateway()
        store = LocalArtifactStore(tmp_path)
        stage = _stage("actor_theater_conversation")
        result = await actor_theater_conversation_handler(
            stage,
            StageContext(
                runner_id="runner-1",
                artifact_store=store,
                inference_gateway=gateway,
            ),
        )

        assert len(result.artifacts) == 1
        assert len(gateway.requests) == 4
        assert gateway.requests[0].messages[0].role == "system"
        assert gateway.requests[0].metadata["role"] == "actor"
        assert gateway.requests[1].metadata["role"] == "theater"
        assert "Ask exactly one concise question" in str(gateway.requests[0].messages[-1].content)

        artifact = json.loads((await store.get_bytes(result.artifacts[0])).decode())
        assert len(artifact["turns"]) == 2
        assert "Turn 1 Actor" in artifact["response"]["text"]

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

    asyncio.run(scenario())


def test_grader_eval_handler_hydrates_parent_conversation(tmp_path) -> None:
    async def scenario() -> None:
        gateway = FakeGateway()
        store = LocalArtifactStore(tmp_path / "artifacts")
        runtime = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")

        parent_spec = StageSpec(stage_type="actor_theater_conversation", run_id="run-1")
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


def _stage(stage_type: str, metadata: dict | None = None) -> StageRecord:
    spec = StageSpec(
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
