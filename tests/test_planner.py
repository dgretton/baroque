from baroque.config.loader import load_project_config_dir
from baroque.orchestration.planner import StaticRunPlanner
from baroque.storage.duckdb_runtime import DuckDBRuntimeStore


def test_static_run_planner_creates_conversation_then_grader_stage() -> None:
    config = load_project_config_dir("configs")
    planner = StaticRunPlanner(config)

    stages = planner.plan_missing_stages("baseline_prompt_only")

    assert [stage.stage_type for stage in stages] == [
        "actor_theater_conversation",
        "grader_eval",
    ]
    conversation, grader = stages
    assert grader.parent_hashes == [conversation.deterministic_hash()]
    assert conversation.sample_id == grader.sample_id
    assert conversation.metadata["role"] == "actor"
    assert grader.metadata["role"] == "grader"
    assert conversation.metadata["model_config"]["model"] == "gemma4:e2b"


def test_static_run_planner_is_deterministic() -> None:
    config = load_project_config_dir("configs")
    planner = StaticRunPlanner(config)

    first = planner.plan_missing_stages("baseline_prompt_only")
    second = planner.plan_missing_stages("baseline_prompt_only")

    assert [stage.deterministic_hash() for stage in first] == [
        stage.deterministic_hash() for stage in second
    ]


def test_static_run_planner_seeds_runtime_store(tmp_path) -> None:
    import asyncio

    async def scenario() -> None:
        config = load_project_config_dir("configs")
        planner = StaticRunPlanner(config)
        store = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")

        records = await planner.seed_run(store, "baseline_prompt_only")
        duplicate_records = await planner.seed_run(store, "baseline_prompt_only")

        assert len(records) == 2
        assert [record.stage_id for record in records] == [
            record.stage_id for record in duplicate_records
        ]

    asyncio.run(scenario())
