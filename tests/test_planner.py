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
        "assessment_aggregate",
        "mutation_proposal",
        "mutation_application",
    ]
    conversation, grader, aggregate, mutation_proposal, mutation_application = stages
    assert grader.parent_hashes == [conversation.deterministic_hash()]
    assert aggregate.parent_hashes == [grader.deterministic_hash()]
    assert mutation_proposal.parent_hashes == [aggregate.deterministic_hash()]
    assert mutation_application.parent_hashes == [mutation_proposal.deterministic_hash()]
    assert conversation.sample_id == grader.sample_id
    assert conversation.sample_id == aggregate.sample_id
    assert conversation.sample_id == mutation_proposal.sample_id
    assert conversation.sample_id == mutation_application.sample_id
    assert conversation.metadata["role"] == "actor"
    assert grader.metadata["role"] == "grader"
    assert aggregate.metadata["role"] == "assessment_aggregator"
    assert mutation_proposal.metadata["role"] == "mutator"
    assert mutation_application.metadata["role"] == "mutation_applicator"
    assert conversation.metadata["model_config"]["model"] == "gemma4:e2b"
    assert conversation.metadata["theater_model_config"]["model"] == "gemma4:e2b"
    assert conversation.metadata["conversation_turns"] == 2
    assert grader.metadata["assessment_index"] == 0
    assert grader.config_snapshot["disclosure_points"][0]["id"] == "starter_assumptions"


def test_static_run_planner_is_deterministic() -> None:
    config = load_project_config_dir("configs")
    planner = StaticRunPlanner(config)

    first = planner.plan_missing_stages("baseline_prompt_only")
    second = planner.plan_missing_stages("baseline_prompt_only")

    assert [stage.deterministic_hash() for stage in first] == [
        stage.deterministic_hash() for stage in second
    ]


def test_static_run_planner_honors_replicate_counts() -> None:
    config = load_project_config_dir("configs")
    config.runs["baseline_prompt_only"].metadata["rollout_replicates"] = 2
    config.runs["baseline_prompt_only"].metadata["assessment_replicates"] = 2
    planner = StaticRunPlanner(config)

    stages = planner.plan_missing_stages("baseline_prompt_only")

    assert [stage.stage_type for stage in stages] == [
        "actor_theater_conversation",
        "grader_eval",
        "grader_eval",
        "assessment_aggregate",
        "mutation_proposal",
        "mutation_application",
        "actor_theater_conversation",
        "grader_eval",
        "grader_eval",
        "assessment_aggregate",
        "mutation_proposal",
        "mutation_application",
    ]
    assert [stage.metadata["rollout_index"] for stage in stages] == [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert [stage.metadata.get("assessment_index") for stage in stages] == [
        None,
        0,
        1,
        None,
        None,
        None,
        None,
        0,
        1,
        None,
        None,
        None,
    ]
    assert stages[3].parent_hashes == [
        stages[1].deterministic_hash(),
        stages[2].deterministic_hash(),
    ]
    assert stages[4].parent_hashes == [stages[3].deterministic_hash()]
    assert stages[5].parent_hashes == [stages[4].deterministic_hash()]


def test_static_run_planner_seeds_runtime_store(tmp_path) -> None:
    import asyncio

    async def scenario() -> None:
        config = load_project_config_dir("configs")
        planner = StaticRunPlanner(config)
        store = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")

        records = await planner.seed_run(store, "baseline_prompt_only")
        duplicate_records = await planner.seed_run(store, "baseline_prompt_only")

        assert len(records) == 5
        assert [record.stage_id for record in records] == [
            record.stage_id for record in duplicate_records
        ]

    asyncio.run(scenario())
