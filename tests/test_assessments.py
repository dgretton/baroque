from baroque.core.models import StageRecord
from baroque.ranking import aggregate_assessments, assessment_record_from_grader_artifact


def test_assessment_record_from_grader_artifact_fills_missing_configured_points() -> None:
    stage = _grader_stage()
    record = assessment_record_from_grader_artifact(
        stage,
        {
            "parsed_assessment": {
                "disclosure_points": [
                    {
                        "id": "starter_assumptions",
                        "status": "extracted",
                        "confidence": 0.9,
                        "evidence": "It names assumptions.",
                    }
                ],
                "overall_rating": 0.8,
                "overall_rationale": "Strong extraction.",
            }
        },
    )

    assert record.overall_rating == 0.8
    assert [point.point_id for point in record.disclosure_points] == [
        "starter_assumptions",
        "starter_missing_information",
    ]
    assert record.disclosure_points[0].status == "extracted"
    assert record.disclosure_points[0].weight == 2.0
    assert record.disclosure_points[1].status == "missing"


def test_aggregate_assessments_computes_disclosure_statistics() -> None:
    stage = _grader_stage()
    first = assessment_record_from_grader_artifact(
        stage,
        {
            "parsed_assessment": {
                "disclosure_points": [
                    {"id": "starter_assumptions", "status": "extracted", "confidence": 1.0},
                    {"id": "starter_missing_information", "status": "partial", "confidence": 0.5},
                ],
                "overall_rating": 0.75,
            }
        },
    )
    second = assessment_record_from_grader_artifact(
        stage.model_copy(update={"metadata": stage.metadata | {"assessment_index": 1}}),
        {
            "parsed_assessment": {
                "disclosure_points": [
                    {"id": "starter_assumptions", "status": "partial", "confidence": 0.6},
                    {"id": "starter_missing_information", "status": "missing"},
                ],
                "overall_rating": 0.25,
            }
        },
    )

    aggregate = aggregate_assessments([first, second])

    assert aggregate.assessment_count == 2
    assert aggregate.overall_rating_mean == 0.5
    assumptions = aggregate.disclosure_points[0]
    assert assumptions.extraction_rate == 0.5
    assert assumptions.partial_or_extracted_rate == 1.0
    assert assumptions.mean_score == 0.75
    assert assumptions.weighted_score == 1.5


def _grader_stage() -> StageRecord:
    return StageRecord(
        content_hash="sha256:grader",
        run_id="run-1",
        stage_type="grader_eval",
        sample_id="sample-1",
        metadata={
            "agent_id": "grader-a",
            "genome_id": "grader-a-seed",
            "conversation_hash": "sha256:conversation",
            "rollout_index": 0,
            "assessment_index": 0,
            "config_snapshot": {
                "disclosure_points": [
                    {
                        "id": "starter_assumptions",
                        "label": "Theater assumptions",
                        "description": "Names assumptions.",
                        "weight": 2.0,
                    },
                    {
                        "id": "starter_missing_information",
                        "label": "Missing information",
                        "description": "Names what is missing.",
                        "weight": 1.0,
                    },
                ]
            },
        },
    )
