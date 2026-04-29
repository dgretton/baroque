import asyncio
from datetime import timedelta

from baroque.core.models import ArtifactRef, StageSpec, StageStatus
from baroque.storage.duckdb_runtime import DuckDBRuntimeStore


def test_duckdb_runtime_claims_and_completes_stage(tmp_path) -> None:
    async def scenario() -> None:
        store = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")
        spec = StageSpec(stage_type="grader_eval", run_id="run-1", sample_id="sample-1")

        inserted = await store.add_stage(spec)
        duplicate = await store.add_stage(spec)
        claimed = await store.claim_next_stage("runner-1")

        assert duplicate.stage_id == inserted.stage_id
        assert claimed is not None
        assert claimed.stage_id == inserted.stage_id
        assert claimed.status == StageStatus.RUNNING
        assert claimed.attempt == 1

        artifact = ArtifactRef(
            uri="memory://artifact",
            content_hash="sha256:abc",
            media_type="application/json",
            size_bytes=2,
        )
        await store.complete_stage(claimed.stage_id, "runner-1", [artifact])

        completed = await store.get_stage(claimed.stage_id)
        assert completed is not None
        assert completed.status == StageStatus.SUCCEEDED
        assert completed.artifact_refs[0].content_hash == "sha256:abc"

    asyncio.run(scenario())


def test_duckdb_runtime_retries_retryable_failures(tmp_path) -> None:
    async def scenario() -> None:
        store = DuckDBRuntimeStore(tmp_path / "runtime.duckdb", retry_delay=timedelta(seconds=0))
        inserted = await store.add_stage(StageSpec(stage_type="actor_turn", run_id="run-1"))
        claimed = await store.claim_next_stage("runner-1")
        assert claimed is not None

        await store.fail_stage(
            claimed.stage_id,
            "runner-1",
            retryable=True,
            error={"type": "timeout"},
        )

        failed = await store.get_stage(inserted.stage_id)
        assert failed is not None
        assert failed.status == StageStatus.FAILED_RETRYABLE
        assert failed.error == {"type": "timeout"}

        reclaimed = await store.claim_next_stage("runner-2")
        assert reclaimed is not None
        assert reclaimed.attempt == 2

    asyncio.run(scenario())


def test_duckdb_runtime_reclaims_expired_leases(tmp_path) -> None:
    async def scenario() -> None:
        store = DuckDBRuntimeStore(tmp_path / "runtime.duckdb", lease_ttl=timedelta(seconds=-1))
        await store.add_stage(StageSpec(stage_type="actor_turn", run_id="run-1"))
        claimed = await store.claim_next_stage("runner-1")
        assert claimed is not None

        reclaimed_count = await store.reclaim_expired_leases()
        abandoned = await store.get_stage(claimed.stage_id)

        assert reclaimed_count == 1
        assert abandoned is not None
        assert abandoned.status == StageStatus.ABANDONED
        assert abandoned.lease_owner is None

    asyncio.run(scenario())
