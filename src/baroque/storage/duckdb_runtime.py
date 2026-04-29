"""DuckDB-backed local runtime store.

This implementation is intentionally local-first. It is suitable for one runner
process and for exercising the lease/state model before introducing Postgres or
another distributed control plane.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import duckdb

from baroque.core.hashing import canonical_json
from baroque.core.models import ArtifactRef, StageRecord, StageSpec, StageStatus, utc_now


class DuckDBRuntimeStore:
    """Local DuckDB runtime store with stage leases and attempts."""

    def __init__(
        self,
        path: str | Path,
        *,
        lease_ttl: timedelta | None = None,
        retry_delay: timedelta | None = None,
    ) -> None:
        self._path = Path(path)
        self._lease_ttl = lease_ttl if lease_ttl is not None else timedelta(minutes=10)
        self._retry_delay = retry_delay if retry_delay is not None else timedelta(seconds=30)
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    async def add_stage(self, spec: StageSpec) -> StageRecord:
        """Insert a pending stage if its content hash is new, otherwise return the existing row."""

        return await asyncio.to_thread(self._add_stage_sync, spec)

    async def get_stage(self, stage_id: str) -> StageRecord | None:
        return await asyncio.to_thread(self._get_stage_sync, stage_id)

    async def get_stage_by_hash(self, hash_value: str) -> StageRecord | None:
        return await asyncio.to_thread(self._get_stage_by_hash_sync, hash_value)

    async def claim_next_stage(self, runner_id: str) -> StageRecord | None:
        return await asyncio.to_thread(self._claim_next_stage_sync, runner_id)

    async def heartbeat(self, stage_id: str, runner_id: str) -> None:
        await asyncio.to_thread(self._heartbeat_sync, stage_id, runner_id)

    async def complete_stage(
        self,
        stage_id: str,
        runner_id: str,
        artifacts: Sequence[ArtifactRef],
    ) -> None:
        await asyncio.to_thread(self._complete_stage_sync, stage_id, runner_id, artifacts)

    async def fail_stage(
        self,
        stage_id: str,
        runner_id: str,
        *,
        retryable: bool,
        error: Mapping[str, Any],
    ) -> None:
        await asyncio.to_thread(
            self._fail_stage_sync,
            stage_id,
            runner_id,
            retryable,
            dict(error),
        )

    async def reclaim_expired_leases(self) -> int:
        return await asyncio.to_thread(self._reclaim_expired_leases_sync)

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self._path))

    def _initialize_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stages (
                    stage_id VARCHAR PRIMARY KEY,
                    content_hash VARCHAR UNIQUE NOT NULL,
                    run_id VARCHAR NOT NULL,
                    stage_type VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    iteration_id VARCHAR,
                    sample_id VARCHAR,
                    parent_hashes_json VARCHAR NOT NULL,
                    attempt INTEGER NOT NULL,
                    lease_owner VARCHAR,
                    leased_until TIMESTAMPTZ,
                    heartbeat_at TIMESTAMPTZ,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    artifact_refs_json VARCHAR NOT NULL,
                    error_json VARCHAR,
                    metadata_json VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stage_attempts (
                    stage_id VARCHAR NOT NULL,
                    attempt INTEGER NOT NULL,
                    runner_id VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    started_at TIMESTAMPTZ NOT NULL,
                    completed_at TIMESTAMPTZ,
                    artifact_refs_json VARCHAR NOT NULL,
                    error_json VARCHAR,
                    PRIMARY KEY (stage_id, attempt)
                )
                """
            )

    def _add_stage_sync(self, spec: StageSpec) -> StageRecord:
        content_hash = spec.deterministic_hash()
        existing = self._get_stage_by_hash_sync(content_hash)
        if existing is not None:
            return existing

        now = utc_now()
        record = StageRecord(
            content_hash=content_hash,
            run_id=spec.run_id,
            stage_type=spec.stage_type,
            iteration_id=spec.iteration_id,
            sample_id=spec.sample_id,
            parent_hashes=spec.parent_hashes,
            metadata=spec.metadata
            | {
                "config_snapshot": spec.config_snapshot,
                "requested_controls": spec.requested_controls,
            },
        )

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO stages (
                    stage_id, content_hash, run_id, stage_type, status,
                    iteration_id, sample_id, parent_hashes_json, attempt,
                    lease_owner, leased_until, heartbeat_at, started_at, completed_at,
                    artifact_refs_json, error_json, metadata_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._stage_params(record, created_at=now, updated_at=now),
            )
        return record

    def _get_stage_sync(self, stage_id: str) -> StageRecord | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM stages WHERE stage_id = ?",
                [stage_id],
            ).fetchone()
        return self._row_to_stage(row) if row else None

    def _get_stage_by_hash_sync(self, hash_value: str) -> StageRecord | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM stages WHERE content_hash = ?",
                [hash_value],
            ).fetchone()
        return self._row_to_stage(row) if row else None

    def _claim_next_stage_sync(self, runner_id: str) -> StageRecord | None:
        now = utc_now()
        leased_until = now + self._lease_ttl
        retry_cutoff = now - self._retry_delay

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM stages
                WHERE status IN (?, ?)
                   OR (status = ? AND completed_at <= ?)
                ORDER BY created_at, stage_id
                LIMIT 100
                """,
                [
                    StageStatus.PENDING.value,
                    StageStatus.ABANDONED.value,
                    StageStatus.FAILED_RETRYABLE.value,
                    retry_cutoff,
                ],
            ).fetchall()
            record = self._first_runnable_stage(conn, rows)
            if record is None:
                return None

            next_attempt = record.attempt + 1
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.execute(
                    """
                    UPDATE stages
                    SET status = ?, attempt = ?, lease_owner = ?, leased_until = ?,
                        heartbeat_at = ?, started_at = ?, completed_at = NULL,
                        error_json = NULL, updated_at = ?
                    WHERE stage_id = ?
                    """,
                    [
                        StageStatus.RUNNING.value,
                        next_attempt,
                        runner_id,
                        leased_until,
                        now,
                        now,
                        now,
                        record.stage_id,
                    ],
                )
                conn.execute(
                    """
                    INSERT INTO stage_attempts (
                        stage_id, attempt, runner_id, status, started_at, completed_at,
                        artifact_refs_json, error_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        record.stage_id,
                        next_attempt,
                        runner_id,
                        StageStatus.RUNNING.value,
                        now,
                        None,
                        "[]",
                        None,
                    ],
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        claimed = self._get_stage_sync(record.stage_id)
        if claimed is None:
            raise RuntimeError(f"claimed stage disappeared: {record.stage_id}")
        return claimed

    def _first_runnable_stage(
        self,
        conn: duckdb.DuckDBPyConnection,
        rows: Sequence[Sequence[Any]],
    ) -> StageRecord | None:
        for row in rows:
            record = self._row_to_stage(row)
            if self._parents_succeeded(conn, record.parent_hashes):
                return record
        return None

    def _parents_succeeded(
        self,
        conn: duckdb.DuckDBPyConnection,
        parent_hashes: Sequence[str],
    ) -> bool:
        for parent_hash in parent_hashes:
            row = conn.execute(
                "SELECT status FROM stages WHERE content_hash = ?",
                [parent_hash],
            ).fetchone()
            if row is None or row[0] != StageStatus.SUCCEEDED.value:
                return False
        return True

    def _heartbeat_sync(self, stage_id: str, runner_id: str) -> None:
        now = utc_now()
        leased_until = now + self._lease_ttl
        with self._lock, self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE stages
                SET heartbeat_at = ?, leased_until = ?, updated_at = ?
                WHERE stage_id = ? AND lease_owner = ? AND status = ?
                RETURNING stage_id
                """,
                [now, leased_until, now, stage_id, runner_id, StageStatus.RUNNING.value],
            ).fetchone()
        if updated is None:
            raise RuntimeError(f"stage is not leased by runner: {stage_id}")

    def _complete_stage_sync(
        self,
        stage_id: str,
        runner_id: str,
        artifacts: Sequence[ArtifactRef],
    ) -> None:
        now = utc_now()
        artifacts_json = self._to_json([artifact.model_dump(mode="json") for artifact in artifacts])
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                updated = conn.execute(
                    """
                    UPDATE stages
                    SET status = ?, completed_at = ?, artifact_refs_json = ?,
                        lease_owner = NULL, leased_until = NULL, heartbeat_at = NULL,
                        updated_at = ?
                    WHERE stage_id = ? AND lease_owner = ? AND status = ?
                    RETURNING attempt
                    """,
                    [
                        StageStatus.SUCCEEDED.value,
                        now,
                        artifacts_json,
                        now,
                        stage_id,
                        runner_id,
                        StageStatus.RUNNING.value,
                    ],
                ).fetchone()
                if updated is None:
                    raise RuntimeError(f"stage is not leased by runner: {stage_id}")
                attempt = int(updated[0])
                conn.execute(
                    """
                    UPDATE stage_attempts
                    SET status = ?, completed_at = ?, artifact_refs_json = ?
                    WHERE stage_id = ? AND attempt = ? AND runner_id = ?
                    """,
                    [
                        StageStatus.SUCCEEDED.value,
                        now,
                        artifacts_json,
                        stage_id,
                        attempt,
                        runner_id,
                    ],
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _fail_stage_sync(
        self,
        stage_id: str,
        runner_id: str,
        retryable: bool,
        error: dict[str, Any],
    ) -> None:
        now = utc_now()
        status = StageStatus.FAILED_RETRYABLE if retryable else StageStatus.FAILED_TERMINAL
        error_json = self._to_json(error)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN TRANSACTION")
            try:
                updated = conn.execute(
                    """
                    UPDATE stages
                    SET status = ?, completed_at = ?, error_json = ?,
                        lease_owner = NULL, leased_until = NULL, heartbeat_at = NULL,
                        updated_at = ?
                    WHERE stage_id = ? AND lease_owner = ? AND status = ?
                    RETURNING attempt
                    """,
                    [
                        status.value,
                        now,
                        error_json,
                        now,
                        stage_id,
                        runner_id,
                        StageStatus.RUNNING.value,
                    ],
                ).fetchone()
                if updated is None:
                    raise RuntimeError(f"stage is not leased by runner: {stage_id}")
                attempt = int(updated[0])
                conn.execute(
                    """
                    UPDATE stage_attempts
                    SET status = ?, completed_at = ?, error_json = ?
                    WHERE stage_id = ? AND attempt = ? AND runner_id = ?
                    """,
                    [status.value, now, error_json, stage_id, attempt, runner_id],
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _reclaim_expired_leases_sync(self) -> int:
        now = utc_now()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                UPDATE stages
                SET status = ?, lease_owner = NULL, leased_until = NULL,
                    heartbeat_at = NULL, updated_at = ?
                WHERE status = ? AND leased_until < ?
                RETURNING stage_id
                """,
                [StageStatus.ABANDONED.value, now, StageStatus.RUNNING.value, now],
            ).fetchall()
        return len(rows)

    def _stage_params(
        self,
        record: StageRecord,
        *,
        created_at: Any,
        updated_at: Any,
    ) -> list[Any]:
        return [
            record.stage_id,
            record.content_hash,
            record.run_id,
            record.stage_type,
            record.status.value,
            record.iteration_id,
            record.sample_id,
            self._to_json(record.parent_hashes),
            record.attempt,
            record.lease_owner,
            record.leased_until,
            record.heartbeat_at,
            record.started_at,
            record.completed_at,
            self._to_json([artifact.model_dump(mode="json") for artifact in record.artifact_refs]),
            self._to_json(record.error) if record.error is not None else None,
            self._to_json(record.metadata),
            created_at,
            updated_at,
        ]

    @staticmethod
    def _row_to_stage(row: Sequence[Any]) -> StageRecord:
        return StageRecord(
            stage_id=row[0],
            content_hash=row[1],
            run_id=row[2],
            stage_type=row[3],
            status=StageStatus(row[4]),
            iteration_id=row[5],
            sample_id=row[6],
            parent_hashes=json.loads(row[7]),
            attempt=row[8],
            lease_owner=row[9],
            leased_until=row[10],
            heartbeat_at=row[11],
            started_at=row[12],
            completed_at=row[13],
            artifact_refs=[ArtifactRef.model_validate(item) for item in json.loads(row[14])],
            error=json.loads(row[15]) if row[15] else None,
            metadata=json.loads(row[16]),
        )

    @staticmethod
    def _to_json(value: Any) -> str:
        return canonical_json(value)
