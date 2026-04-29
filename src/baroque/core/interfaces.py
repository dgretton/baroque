"""System boundary protocols.

These protocols are the contracts that should stay stable while implementations
move from local DuckDB/filesystem storage to Postgres, object storage, Ray,
Temporal, cloud workers, or other execution backends.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from baroque.core.models import (
    ArtifactRef,
    EventRecord,
    ProviderRequest,
    ProviderResponse,
    StageRecord,
    StageSpec,
)


class ConfigStore(Protocol):
    async def load_run_config(self, run_id: str) -> Mapping[str, Any]:
        """Load the resolved immutable config snapshot for a run."""


class Planner(Protocol):
    async def plan_missing_stages(self, run_id: str) -> Sequence[StageSpec]:
        """Return stages that are required but not completed."""


class LeaseStore(Protocol):
    async def claim_next_stage(self, runner_id: str) -> StageRecord | None:
        """Claim a runnable stage for a runner, returning None if no work exists."""

    async def reclaim_expired_leases(self) -> int:
        """Return expired running stages to the runnable pool."""

    async def heartbeat(self, stage_id: str, runner_id: str) -> None:
        """Refresh the lease for a running stage."""

    async def complete_stage(
        self,
        stage_id: str,
        runner_id: str,
        artifacts: Sequence[ArtifactRef],
    ) -> None:
        """Atomically mark a stage complete after its artifacts are durable."""

    async def fail_stage(
        self,
        stage_id: str,
        runner_id: str,
        *,
        retryable: bool,
        error: Mapping[str, Any],
    ) -> None:
        """Record a failed attempt and optionally schedule a retry."""


class ArtifactStore(Protocol):
    async def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str,
        suffix: str = "",
    ) -> ArtifactRef:
        """Persist bytes and return a content-addressed reference."""

    async def get_bytes(self, ref: ArtifactRef) -> bytes:
        """Load bytes by artifact reference."""

    async def exists(self, ref: ArtifactRef) -> bool:
        """Return whether the artifact still exists."""


class InferenceGateway(Protocol):
    async def send(self, request: ProviderRequest) -> ProviderResponse:
        """Send a provider request to a local or remote inference endpoint."""


class EventSink(Protocol):
    async def emit(self, event: EventRecord) -> None:
        """Emit a structured event to logs, telemetry, or both."""


class AnalysisExport(Protocol):
    async def export_run(self, run_id: str, destination: str) -> None:
        """Export analysis-ready data for a run."""
