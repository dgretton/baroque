"""Async stage runner."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from baroque.core.interfaces import ArtifactStore, EventSink, InferenceGateway, LeaseStore
from baroque.core.models import EventRecord, StageRecord
from baroque.orchestration.handlers import (
    StageContext,
    StageExecutionError,
    StageHandler,
    StageResult,
)


@dataclass(frozen=True)
class RunnerConfig:
    runner_id: str
    max_concurrent_stages: int = 1
    idle_sleep_s: float = 1.0
    heartbeat_interval_s: float = 30.0
    reclaim_interval_s: float = 30.0
    context_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunnerStats:
    claimed: int = 0
    succeeded: int = 0
    failed_retryable: int = 0
    failed_terminal: int = 0
    reclaimed: int = 0

    def plus(self, other: RunnerStats) -> RunnerStats:
        return RunnerStats(
            claimed=self.claimed + other.claimed,
            succeeded=self.succeeded + other.succeeded,
            failed_retryable=self.failed_retryable + other.failed_retryable,
            failed_terminal=self.failed_terminal + other.failed_terminal,
            reclaimed=self.reclaimed + other.reclaimed,
        )


class AsyncStageRunner:
    """Claim stages from a lease store and dispatch them to registered handlers."""

    def __init__(
        self,
        *,
        lease_store: LeaseStore,
        handlers: Mapping[str, StageHandler],
        config: RunnerConfig,
        artifact_store: ArtifactStore | None = None,
        inference_gateway: InferenceGateway | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        if config.max_concurrent_stages < 1:
            raise ValueError("max_concurrent_stages must be at least 1")
        self._lease_store = lease_store
        self._handlers = dict(handlers)
        self._config = config
        self._artifact_store = artifact_store
        self._inference_gateway = inference_gateway
        self._event_sink = event_sink
        self._stop_event = asyncio.Event()

    def stop(self) -> None:
        self._stop_event.set()

    async def run_until_idle(self, *, max_claims: int | None = None) -> RunnerStats:
        """Run claimed work until no runnable stages remain."""

        stats = RunnerStats()
        in_flight: set[asyncio.Task[RunnerStats]] = set()
        next_reclaim_after = 0.0

        while not self._stop_event.is_set():
            now = asyncio.get_running_loop().time()
            if now >= next_reclaim_after:
                reclaimed = await self._lease_store.reclaim_expired_leases()
                if reclaimed:
                    stats = stats.plus(RunnerStats(reclaimed=reclaimed))
                    await self._emit(EventRecord(event="runner.reclaimed_leases", attributes={
                        "count": reclaimed
                    }))
                next_reclaim_after = now + self._config.reclaim_interval_s

            while len(in_flight) < self._config.max_concurrent_stages:
                if max_claims is not None and stats.claimed >= max_claims:
                    break
                stage = await self._lease_store.claim_next_stage(self._config.runner_id)
                if stage is None:
                    break
                stats = stats.plus(RunnerStats(claimed=1))
                task = asyncio.create_task(self._execute_stage(stage))
                in_flight.add(task)

            if not in_flight:
                break

            done, pending = await asyncio.wait(
                in_flight,
                timeout=self._config.idle_sleep_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            in_flight = pending
            for task in done:
                stats = stats.plus(task.result())

            if max_claims is not None and stats.claimed >= max_claims and not in_flight:
                break

        if in_flight:
            for task in in_flight:
                task.cancel()
            results = await asyncio.gather(*in_flight, return_exceptions=True)
            for result in results:
                if isinstance(result, RunnerStats):
                    stats = stats.plus(result)
        return stats

    async def run_forever(self) -> None:
        """Run continuously until `stop` is called or the task is cancelled."""

        await self._emit(EventRecord(event="runner.started"))
        try:
            while not self._stop_event.is_set():
                stats = await self.run_until_idle()
                await self._emit(
                    EventRecord(
                        event="runner.idle",
                        attributes=stats.__dict__,
                    )
                )
                await asyncio.sleep(self._config.idle_sleep_s)
        except asyncio.CancelledError:
            self.stop()
            await self._emit(EventRecord(event="runner.cancelled", level="warning"))
            raise
        finally:
            await self._emit(EventRecord(event="runner.stopped"))

    async def _execute_stage(self, stage: StageRecord) -> RunnerStats:
        await self._emit(self._stage_event("stage.claimed", stage))
        done = asyncio.Event()
        heartbeat_task = asyncio.create_task(self._heartbeat_until_done(stage, done))
        try:
            handler = self._handlers.get(stage.stage_type)
            if handler is None:
                raise StageExecutionError(
                    f"no handler registered for stage type: {stage.stage_type}",
                    retryable=False,
                    error_type="missing_stage_handler",
                )

            context = StageContext(
                runner_id=self._config.runner_id,
                artifact_store=self._artifact_store,
                inference_gateway=self._inference_gateway,
                event_sink=self._event_sink,
                metadata=self._config.context_metadata,
            )
            result = await handler(stage, context)
            await self._complete_stage(stage, result)
            return RunnerStats(succeeded=1)
        except StageExecutionError as error:
            await self._fail_stage(stage, error.retryable, error.as_error_payload())
            return RunnerStats(
                failed_retryable=1 if error.retryable else 0,
                failed_terminal=0 if error.retryable else 1,
            )
        except Exception as error:
            payload = {
                "type": error.__class__.__name__,
                "message": str(error),
                "retryable": True,
            }
            await self._fail_stage(stage, True, payload)
            return RunnerStats(failed_retryable=1)
        finally:
            done.set()
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)

    async def _complete_stage(self, stage: StageRecord, result: StageResult) -> None:
        await self._lease_store.complete_stage(
            stage.stage_id,
            self._config.runner_id,
            result.artifacts,
        )
        await self._emit(
            self._stage_event(
                "stage.succeeded",
                stage,
                attributes={"artifact_count": len(result.artifacts), **result.attributes},
            )
        )

    async def _fail_stage(
        self,
        stage: StageRecord,
        retryable: bool,
        error: Mapping[str, Any],
    ) -> None:
        await self._lease_store.fail_stage(
            stage.stage_id,
            self._config.runner_id,
            retryable=retryable,
            error=error,
        )
        await self._emit(
            self._stage_event(
                "stage.failed",
                stage,
                level="warning" if retryable else "error",
                error_type=str(error.get("type") or "stage_error"),
                attributes=dict(error),
            )
        )

    async def _heartbeat_until_done(self, stage: StageRecord, done: asyncio.Event) -> None:
        while not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), timeout=self._config.heartbeat_interval_s)
            except TimeoutError:
                try:
                    await self._lease_store.heartbeat(stage.stage_id, self._config.runner_id)
                    await self._emit(self._stage_event("stage.heartbeat", stage))
                except Exception as error:
                    await self._emit(
                        self._stage_event(
                            "stage.heartbeat_failed",
                            stage,
                            level="warning",
                            error_type=error.__class__.__name__,
                            attributes={"message": str(error)},
                        )
                    )

    async def _emit(self, event: EventRecord) -> None:
        if self._event_sink is None:
            return
        if event.runner_id is None:
            event.runner_id = self._config.runner_id
        await self._event_sink.emit(event)

    @staticmethod
    def _stage_event(
        event: str,
        stage: StageRecord,
        *,
        level: str = "info",
        error_type: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> EventRecord:
        return EventRecord(
            event=event,
            level=level,
            run_id=stage.run_id,
            iteration_id=stage.iteration_id,
            sample_id=stage.sample_id,
            stage_id=stage.stage_id,
            content_hash=stage.content_hash,
            attempt=stage.attempt,
            error_type=error_type,
            attributes=attributes or {},
        )
