import asyncio
import json

from scripts.stacked_agents_poc import (
    DEFAULT_GOAL,
    MockChatClient,
    StackConfig,
    _json_object_or_fallback,
    format_transcript,
    run_stack,
    write_run_artifact,
)


def test_mock_stacked_agent_run_produces_three_level_artifact(tmp_path) -> None:
    async def scenario() -> None:
        config = StackConfig(
            goal=DEFAULT_GOAL,
            disclosure_points=[{"id": "assumptions", "label": "Assumptions"}],
            iterations=2,
            turns=3,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            actor_model="mock",
            theater_model="mock",
            assessor_model="mock",
            meta_assessor_model="mock",
            temperature=0.0,
            timeout_s=1.0,
        )

        result = await run_stack(config, MockChatClient())
        output = tmp_path / "stacked.json"
        write_run_artifact(result, output)

        saved = json.loads(output.read_text())
        assert len(saved["iterations"]) == 2
        assert len(saved["iterations"][0]["transcript"]) == 3
        assert saved["iterations"][0]["assessment"]["extracted"] == [
            "assumptions",
            "missing_information",
        ]
        assert saved["iterations"][0]["meta_assessment"]["assessment_quality"] == 0.7
        assert "Explicitly ask about tradeoffs" in saved["final_genomes"]["actor"]["persona"]
        assert "quoted transcript evidence" in saved["final_genomes"]["assessor"]["persona"]

    asyncio.run(scenario())


def test_json_fallback_preserves_raw_response() -> None:
    parsed = _json_object_or_fallback("not json", fallback_key="raw_assessment")

    assert parsed["raw_assessment"] == "not json"
    assert parsed["overall_score"] == 0.0
    assert parsed["revised_prompt"] is None


def test_format_transcript_handles_empty_and_turns() -> None:
    assert format_transcript([]) == "No prior turns."
