import asyncio

from baroque.core.models import EventRecord, StageRecord, StageSpec, StageStatus
from baroque.orchestration import (
    AsyncStageRunner,
    RunnerConfig,
    StageContext,
    StageExecutionError,
    StageResult,
)
from baroque.storage.duckdb_runtime import DuckDBRuntimeStore
from baroque.storage.local_artifacts import LocalArtifactStore


class MemoryEventSink:
    def __init__(self) -> None:
        self.events: list[EventRecord] = []

    async def emit(self, event: EventRecord) -> None:
        self.events.append(event)


async def successful_handler(stage: StageRecord, context: StageContext) -> StageResult:
    assert context.artifact_store is not None
    ref = await context.artifact_store.put_bytes(
        f"stage:{stage.stage_id}".encode(),
        media_type="text/plain",
        suffix=".txt",
    )
    return StageResult(artifacts=[ref], attributes={"handled": True})


async def terminal_failure_handler(stage: StageRecord, context: StageContext) -> StageResult:
    raise StageExecutionError("bad stage", retryable=False, error_type="bad_stage")


async def retryable_failure_handler(stage: StageRecord, context: StageContext) -> StageResult:
    raise StageExecutionError("try again", retryable=True, error_type="temporary_error")


def test_runner_completes_successful_stage(tmp_path) -> None:
    async def scenario() -> None:
        runtime = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        events = MemoryEventSink()
        inserted = await runtime.add_stage(StageSpec(stage_type="echo", run_id="run-1"))

        runner = AsyncStageRunner(
            lease_store=runtime,
            handlers={"echo": successful_handler},
            config=RunnerConfig(runner_id="runner-1", heartbeat_interval_s=0.01),
            artifact_store=artifacts,
            event_sink=events,
        )
        stats = await runner.run_until_idle()

        completed = await runtime.get_stage(inserted.stage_id)
        assert stats.succeeded == 1
        assert completed is not None
        assert completed.status == StageStatus.SUCCEEDED
        assert len(completed.artifact_refs) == 1
        assert [event.event for event in events.events] == ["stage.claimed", "stage.succeeded"]

    asyncio.run(scenario())


def test_runner_marks_missing_handler_terminal(tmp_path) -> None:
    async def scenario() -> None:
        runtime = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")
        inserted = await runtime.add_stage(StageSpec(stage_type="unknown", run_id="run-1"))

        runner = AsyncStageRunner(
            lease_store=runtime,
            handlers={},
            config=RunnerConfig(runner_id="runner-1"),
        )
        stats = await runner.run_until_idle()

        failed = await runtime.get_stage(inserted.stage_id)
        assert stats.failed_terminal == 1
        assert failed is not None
        assert failed.status == StageStatus.FAILED_TERMINAL
        assert failed.error is not None
        assert failed.error["type"] == "missing_stage_handler"

    asyncio.run(scenario())


def test_runner_respects_max_claims_for_retryable_failures(tmp_path) -> None:
    async def scenario() -> None:
        runtime = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")
        inserted = await runtime.add_stage(StageSpec(stage_type="retryable", run_id="run-1"))

        runner = AsyncStageRunner(
            lease_store=runtime,
            handlers={"retryable": retryable_failure_handler},
            config=RunnerConfig(runner_id="runner-1"),
        )
        stats = await runner.run_until_idle(max_claims=1)

        failed = await runtime.get_stage(inserted.stage_id)
        assert stats.failed_retryable == 1
        assert failed is not None
        assert failed.status == StageStatus.FAILED_RETRYABLE
        assert failed.error is not None
        assert failed.error["type"] == "temporary_error"

    asyncio.run(scenario())
