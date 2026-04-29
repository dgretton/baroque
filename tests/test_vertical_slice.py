import asyncio

from baroque.agents import prompt_only_handlers
from baroque.config.loader import load_project_config_dir
from baroque.core.models import ProviderRequest, ProviderResponse, StageStatus
from baroque.orchestration import AsyncStageRunner, RunnerConfig, StaticRunPlanner
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


def test_prompt_only_vertical_slice_runs_seeded_stages(tmp_path) -> None:
    async def scenario() -> None:
        config = load_project_config_dir("configs")
        runtime = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        gateway = FakeGateway()
        planner = StaticRunPlanner(config)

        seeded = await planner.seed_run(runtime, "baseline_prompt_only")
        runner = AsyncStageRunner(
            lease_store=runtime,
            handlers=prompt_only_handlers(),
            config=RunnerConfig(runner_id="runner-1", idle_sleep_s=0.01),
            artifact_store=artifacts,
            inference_gateway=gateway,
        )
        stats = await runner.run_until_idle()

        completed = [await runtime.get_stage(record.stage_id) for record in seeded]
        assert stats.succeeded == 2
        assert len(gateway.requests) == 2
        assert all(record is not None for record in completed)
        assert all(record.status == StageStatus.SUCCEEDED for record in completed if record)

    asyncio.run(scenario())

