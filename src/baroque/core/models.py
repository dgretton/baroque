"""Core runtime and provider models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from baroque.core.hashing import content_hash


def utc_now() -> datetime:
    return datetime.now(UTC)


class StageStatus(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"


class Message(BaseModel):
    role: str
    content: str | list[dict[str, Any]]
    name: str | None = None


class ArtifactRef(BaseModel):
    uri: str
    content_hash: str
    media_type: str
    size_bytes: int
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderRequest(BaseModel):
    endpoint_id: str
    provider: str
    base_url: str
    model: str
    messages: list[Message]
    api_key: str | None = None
    requested_controls: dict[str, Any] = Field(default_factory=dict)
    effective_controls: dict[str, Any] = Field(default_factory=dict)
    dropped_controls: dict[str, Any] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def chat_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.model_dump(exclude_none=True) for message in self.messages],
        }
        payload.update(self.extra_body)
        return payload


class ProviderResponse(BaseModel):
    request_hash: str
    raw_body: dict[str, Any] | None = None
    text: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    received_at: datetime = Field(default_factory=utc_now)


class StageSpec(BaseModel):
    stage_type: str
    run_id: str
    iteration_id: str | None = None
    sample_id: str | None = None
    parent_hashes: list[str] = Field(default_factory=list)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    requested_controls: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def deterministic_hash(self) -> str:
        return content_hash(
            {
                "stage_type": self.stage_type,
                "run_id": self.run_id,
                "iteration_id": self.iteration_id,
                "sample_id": self.sample_id,
                "parent_hashes": self.parent_hashes,
                "config_snapshot": self.config_snapshot,
                "requested_controls": self.requested_controls,
                "metadata": self.metadata,
            }
        )


class StageRecord(BaseModel):
    stage_id: str = Field(default_factory=lambda: str(uuid4()))
    content_hash: str
    run_id: str
    stage_type: str
    status: StageStatus = StageStatus.PENDING
    iteration_id: str | None = None
    sample_id: str | None = None
    parent_hashes: list[str] = Field(default_factory=list)
    attempt: int = 0
    lease_owner: str | None = None
    leased_until: datetime | None = None
    heartbeat_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventRecord(BaseModel):
    event: str
    level: str = "info"
    timestamp: datetime = Field(default_factory=utc_now)
    run_id: str | None = None
    runner_id: str | None = None
    iteration_id: str | None = None
    sample_id: str | None = None
    stage_id: str | None = None
    content_hash: str | None = None
    attempt: int | None = None
    role: str | None = None
    agent_id: str | None = None
    model: str | None = None
    endpoint: str | None = None
    duration_ms: int | None = None
    error_type: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
