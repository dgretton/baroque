"""Typed assessment records and aggregation helpers."""

from __future__ import annotations

import json
from enum import StrEnum
from statistics import mean, pstdev
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from baroque.core.hashing import content_hash
from baroque.core.models import StageRecord


class DisclosureStatus(StrEnum):
    EXTRACTED = "extracted"
    PARTIAL = "partial"
    MISSING = "missing"
    CONTRADICTED = "contradicted"


STATUS_SCORES: dict[DisclosureStatus, float] = {
    DisclosureStatus.EXTRACTED: 1.0,
    DisclosureStatus.PARTIAL: 0.5,
    DisclosureStatus.MISSING: 0.0,
    DisclosureStatus.CONTRADICTED: 0.0,
}


class DisclosurePointJudgment(BaseModel):
    """One evaluator's judgment for a configured disclosure point."""

    model_config = ConfigDict(populate_by_name=True)

    point_id: str = Field(alias="id")
    status: DisclosureStatus = DisclosureStatus.MISSING
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)
    rationale: str = ""
    label: str | None = None
    weight: float = 1.0

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in {status.value for status in DisclosureStatus}:
            return DisclosureStatus.MISSING.value
        return normalized

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, parsed))

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_evidence(cls, value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, list):
            return [str(item) for item in value if item not in (None, "")]
        return [str(value)]

    @property
    def score(self) -> float:
        return STATUS_SCORES[self.status]


class AssessmentRecord(BaseModel):
    """A typed Grader assessment for one rollout."""

    run_id: str
    sample_id: str | None = None
    rollout_index: int | None = None
    assessment_index: int | None = None
    conversation_hash: str | None = None
    stage_hash: str
    grader_id: str | None = None
    grader_genome_id: str | None = None
    disclosure_points: list[DisclosurePointJudgment] = Field(default_factory=list)
    overall_rating: float | None = None
    overall_rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("overall_rating", mode="before")
    @classmethod
    def _normalize_rating(cls, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, parsed))

    def deterministic_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class DisclosurePointAggregate(BaseModel):
    """Aggregated statistics for one disclosure point over plural assessments."""

    point_id: str
    label: str | None = None
    weight: float = 1.0
    assessment_count: int
    status_counts: dict[str, int]
    extraction_rate: float
    partial_or_extracted_rate: float
    mean_confidence: float | None = None
    mean_score: float
    weighted_score: float
    evidence_samples: list[str] = Field(default_factory=list)


class AssessmentAggregate(BaseModel):
    """Aggregated assessment statistics for a rollout."""

    run_id: str
    sample_id: str | None = None
    rollout_index: int | None = None
    conversation_hash: str | None = None
    assessment_count: int
    disclosure_points: list[DisclosurePointAggregate]
    overall_rating_mean: float | None = None
    overall_rating_stdev: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def deterministic_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


def assessment_record_from_grader_artifact(
    stage: StageRecord,
    artifact_payload: dict[str, Any],
) -> AssessmentRecord:
    """Build a typed assessment from a Grader stage and its artifact payload."""

    parsed = artifact_payload.get("parsed_assessment")
    if not isinstance(parsed, dict):
        response_text = ((artifact_payload.get("response") or {}).get("text")) or ""
        parsed = _try_parse_json_object(response_text)
    if not isinstance(parsed, dict):
        parsed = {}

    configured_points = _configured_points(stage)
    configured_by_id = {
        str(point.get("id")): point for point in configured_points if point.get("id") is not None
    }
    parsed_points = parsed.get("disclosure_points") or []
    judgments_by_id: dict[str, DisclosurePointJudgment] = {}
    if isinstance(parsed_points, list):
        for item in parsed_points:
            if isinstance(item, dict) and item.get("id") is not None:
                judgment = DisclosurePointJudgment.model_validate(item)
                judgments_by_id[judgment.point_id] = _with_configured_point(
                    judgment,
                    configured_by_id.get(judgment.point_id),
                )

    judgments: list[DisclosurePointJudgment] = []
    for point in configured_points:
        point_id = str(point.get("id"))
        if point_id in judgments_by_id:
            judgments.append(judgments_by_id.pop(point_id))
        else:
            judgments.append(
                DisclosurePointJudgment(
                    id=point_id,
                    status=DisclosureStatus.MISSING,
                    label=_optional_str(point.get("label")),
                    weight=float(point.get("weight", 1.0)),
                )
            )
    judgments.extend(judgments_by_id.values())

    return AssessmentRecord(
        run_id=stage.run_id,
        sample_id=stage.sample_id,
        rollout_index=_optional_int(stage.metadata.get("rollout_index")),
        assessment_index=_optional_int(stage.metadata.get("assessment_index")),
        conversation_hash=_optional_str(stage.metadata.get("conversation_hash")),
        stage_hash=stage.content_hash,
        grader_id=_optional_str(stage.metadata.get("agent_id")),
        grader_genome_id=_optional_str(stage.metadata.get("genome_id")),
        disclosure_points=judgments,
        overall_rating=parsed.get("overall_rating"),
        overall_rationale=str(parsed.get("overall_rationale") or ""),
        metadata={
            "stage_id": stage.stage_id,
            "artifact_hash": artifact_payload.get("artifact_hash"),
        },
    )


def aggregate_assessments(records: list[AssessmentRecord]) -> AssessmentAggregate:
    """Aggregate plural Grader assessments into rollout-level statistics."""

    if not records:
        raise ValueError("cannot aggregate zero assessments")

    first = records[0]
    point_ids: list[str] = []
    by_point: dict[str, list[DisclosurePointJudgment]] = {}
    for record in records:
        for judgment in record.disclosure_points:
            if judgment.point_id not in by_point:
                point_ids.append(judgment.point_id)
                by_point[judgment.point_id] = []
            by_point[judgment.point_id].append(judgment)

    ratings = [record.overall_rating for record in records if record.overall_rating is not None]
    return AssessmentAggregate(
        run_id=first.run_id,
        sample_id=first.sample_id,
        rollout_index=first.rollout_index,
        conversation_hash=first.conversation_hash,
        assessment_count=len(records),
        disclosure_points=[
            _aggregate_disclosure_point(point_id, by_point[point_id]) for point_id in point_ids
        ],
        overall_rating_mean=mean(ratings) if ratings else None,
        overall_rating_stdev=pstdev(ratings) if len(ratings) > 1 else None,
        metadata={"assessment_hashes": [record.deterministic_hash() for record in records]},
    )


def _aggregate_disclosure_point(
    point_id: str,
    judgments: list[DisclosurePointJudgment],
) -> DisclosurePointAggregate:
    status_counts = {status.value: 0 for status in DisclosureStatus}
    for judgment in judgments:
        status_counts[judgment.status.value] += 1

    confidences = [judgment.confidence for judgment in judgments if judgment.confidence is not None]
    scores = [judgment.score for judgment in judgments]
    first = judgments[0]
    evidence_samples: list[str] = []
    for judgment in judgments:
        for evidence in judgment.evidence:
            if evidence and evidence not in evidence_samples:
                evidence_samples.append(evidence)
            if len(evidence_samples) >= 5:
                break
        if len(evidence_samples) >= 5:
            break

    assessment_count = len(judgments)
    mean_score = mean(scores) if scores else 0.0
    return DisclosurePointAggregate(
        point_id=point_id,
        label=first.label,
        weight=first.weight,
        assessment_count=assessment_count,
        status_counts=status_counts,
        extraction_rate=status_counts[DisclosureStatus.EXTRACTED.value] / assessment_count,
        partial_or_extracted_rate=(
            status_counts[DisclosureStatus.EXTRACTED.value]
            + status_counts[DisclosureStatus.PARTIAL.value]
        )
        / assessment_count,
        mean_confidence=mean(confidences) if confidences else None,
        mean_score=mean_score,
        weighted_score=mean_score * first.weight,
        evidence_samples=evidence_samples,
    )


def _configured_points(stage: StageRecord) -> list[dict[str, Any]]:
    snapshot = stage.metadata.get("config_snapshot") or {}
    points = snapshot.get("disclosure_points") or []
    if not isinstance(points, list):
        return []
    return [point for point in points if isinstance(point, dict)]


def _with_configured_point(
    judgment: DisclosurePointJudgment,
    configured_point: dict[str, Any] | None,
) -> DisclosurePointJudgment:
    if configured_point is None:
        return judgment
    return judgment.model_copy(
        update={
            "label": judgment.label or _optional_str(configured_point.get("label")),
            "weight": float(configured_point.get("weight", judgment.weight)),
        }
    )


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
