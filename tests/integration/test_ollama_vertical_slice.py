import asyncio
import os

import pytest

from baroque.agents import prompt_only_handlers
from baroque.config.loader import load_project_config_dir
from baroque.core.models import StageStatus
from baroque.gateways import OpenAICompatibleGateway
from baroque.orchestration import AsyncStageRunner, RunnerConfig, StaticRunPlanner
from baroque.storage.duckdb_runtime import DuckDBRuntimeStore
from baroque.storage.local_artifacts import LocalArtifactStore

RUN_OLLAMA = os.environ.get("BAROQUE_RUN_OLLAMA_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_OLLAMA,
        reason="set BAROQUE_RUN_OLLAMA_INTEGRATION=1 to run Ollama integration tests",
    ),
]


def test_ollama_prompt_only_vertical_slice(tmp_path) -> None:
    async def scenario() -> None:
        config = load_project_config_dir("configs")
        test_model = os.environ.get("BAROQUE_TEST_MODEL")
        if test_model:
            config.models["gemma4_e2b"].model = test_model

        runtime = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        gateway = OpenAICompatibleGateway(
            timeout_s=float(os.environ.get("BAROQUE_INTEGRATION_TIMEOUT_S", "600"))
        )
        planner = StaticRunPlanner(config)

        seeded = await planner.seed_run(runtime, "baseline_prompt_only")
        runner = AsyncStageRunner(
            lease_store=runtime,
            handlers=prompt_only_handlers(),
            config=RunnerConfig(runner_id="ollama-integration", idle_sleep_s=0.05),
            artifact_store=artifacts,
            inference_gateway=gateway,
            stage_store=runtime,
        )
        stats = await runner.run_until_idle()

        completed = [await runtime.get_stage(record.stage_id) for record in seeded]
        assert stats.succeeded == 3
        assert all(record is not None for record in completed)
        assert all(record.status == StageStatus.SUCCEEDED for record in completed if record)
        assert all(record.artifact_refs for record in completed if record)

    asyncio.run(scenario())
