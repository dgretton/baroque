from baroque.config.loader import load_project_config_dir
from baroque.orchestration.planner import StaticRunPlanner
from baroque.storage.duckdb_runtime import DuckDBRuntimeStore


def test_static_run_planner_creates_conversation_then_grader_stage() -> None:
    config = load_project_config_dir("configs")
    planner = StaticRunPlanner(config)

    stages = planner.plan_missing_stages("baseline_prompt_only")

    assert [stage.stage_type for stage in stages] == [
        "actor_turn",
        "theater_turn",
        "actor_turn",
        "theater_turn",
        "conversation_transcript",
        "grader_eval",
        "assessment_aggregate",
        "mutation_proposal",
        "mutation_application",
    ]
    (
        actor_0,
        theater_0,
        actor_1,
        theater_1,
        conversation,
        grader,
        aggregate,
        mutation_proposal,
        mutation_application,
    ) = stages
    assert actor_0.parent_hashes == []
    assert theater_0.parent_hashes == [actor_0.deterministic_hash()]
    assert actor_1.parent_hashes == [theater_0.deterministic_hash()]
    assert theater_1.parent_hashes == [actor_1.deterministic_hash()]
    assert conversation.parent_hashes == [
        actor_0.deterministic_hash(),
        theater_0.deterministic_hash(),
        actor_1.deterministic_hash(),
        theater_1.deterministic_hash(),
    ]
    assert grader.parent_hashes == [conversation.deterministic_hash()]
    assert aggregate.parent_hashes == [grader.deterministic_hash()]
    assert mutation_proposal.parent_hashes == [aggregate.deterministic_hash()]
    assert mutation_application.parent_hashes == [mutation_proposal.deterministic_hash()]
    assert conversation.sample_id == grader.sample_id
    assert conversation.sample_id == aggregate.sample_id
    assert conversation.sample_id == mutation_proposal.sample_id
    assert conversation.sample_id == mutation_application.sample_id
    assert actor_0.metadata["role"] == "actor"
    assert theater_0.metadata["role"] == "theater"
    assert conversation.metadata["role"] == "conversation_builder"
    assert grader.metadata["role"] == "grader"
    assert aggregate.metadata["role"] == "assessment_aggregator"
    assert mutation_proposal.metadata["role"] == "mutator"
    assert mutation_application.metadata["role"] == "mutation_applicator"
    assert actor_0.metadata["model_config"]["model"] == "gemma4:e2b"
    assert theater_0.metadata["theater_model_config"]["model"] == "gemma4:e2b"
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

    per_rollout = [
        "actor_turn",
        "theater_turn",
        "actor_turn",
        "theater_turn",
        "conversation_transcript",
        "grader_eval",
        "grader_eval",
        "assessment_aggregate",
        "mutation_proposal",
        "mutation_application",
    ]
    assert [stage.stage_type for stage in stages] == per_rollout + per_rollout
    assert [stage.metadata["rollout_index"] for stage in stages] == [0] * 10 + [1] * 10
    assert [stage.metadata.get("assessment_index") for stage in stages[:10]] == [
        None,
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
    assert stages[7].parent_hashes == [
        stages[5].deterministic_hash(),
        stages[6].deterministic_hash(),
    ]
    assert stages[8].parent_hashes == [stages[7].deterministic_hash()]
    assert stages[9].parent_hashes == [stages[8].deterministic_hash()]


def test_static_run_planner_seeds_runtime_store(tmp_path) -> None:
    import asyncio

    async def scenario() -> None:
        config = load_project_config_dir("configs")
        planner = StaticRunPlanner(config)
        store = DuckDBRuntimeStore(tmp_path / "runtime.duckdb")

        records = await planner.seed_run(store, "baseline_prompt_only")
        duplicate_records = await planner.seed_run(store, "baseline_prompt_only")

        assert len(records) == 9
        assert [record.stage_id for record in records] == [
            record.stage_id for record in duplicate_records
        ]

    asyncio.run(scenario())
