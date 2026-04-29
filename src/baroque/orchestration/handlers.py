"""Stage handler abstractions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from baroque.core.interfaces import ArtifactStore, EventSink, InferenceGateway
from baroque.core.models import ArtifactRef, StageRecord


@dataclass(frozen=True)
class StageContext:
    """Context available to stage handlers."""

    runner_id: str
    artifact_store: ArtifactStore | None = None
    inference_gateway: InferenceGateway | None = None
    event_sink: EventSink | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StageResult(BaseModel):
    """Successful stage handler result."""

    artifacts: list[ArtifactRef] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class StageExecutionError(Exception):
    """Exception type for expected stage failures."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        error_type: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.error_type = error_type or self.__class__.__name__
        self.attributes = dict(attributes or {})

    def as_error_payload(self) -> dict[str, Any]:
        return {
            "type": self.error_type,
            "message": str(self),
            "retryable": self.retryable,
            "attributes": self.attributes,
        }


StageHandler = Callable[[StageRecord, StageContext], Awaitable[StageResult]]
