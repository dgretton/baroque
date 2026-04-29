import asyncio

from baroque.agents.prompt_only import actor_theater_conversation_handler, grader_eval_handler
from baroque.core.models import ProviderRequest, ProviderResponse, StageRecord, StageSpec
from baroque.orchestration.handlers import StageContext
from baroque.storage.duckdb_runtime import DuckDBRuntimeStore
from baroque.storage.local_artifacts import LocalArtifactStore


class FakeGateway:
    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    async def send(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            request_hash="sha256:request",
            raw_body={"choices": [{"message": {"content": "ok"}}]},
            text="ok",
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
        assert gateway.requests[0].messages[0].role == "system"
        assert "Interrogate the Theater" in str(gateway.requests[0].messages[-1].content)

    asyncio.run(scenario())


def test_grader_eval_handler_writes_artifact(tmp_path) -> None:
    async def scenario() -> None:
        gateway = FakeGateway()
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
        assert "sha256:conversation" in str(gateway.requests[0].messages[-1].content)

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
            }
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
