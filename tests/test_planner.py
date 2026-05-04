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
    assert mutation_proposal.metadata["mutation_operator_id"] == "deterministic_prompt_baseline"
    assert mutation_proposal.metadata["operator"] == "hand_authored"

    grader_contract = grader.metadata["role_output_contract"]
    response_format = grader_contract["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "grader_assessment"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == ["disclosure_points", "overall_rating", "overall_rationale"]
    point_props = schema["properties"]["disclosure_points"]["items"]["properties"]
    assert point_props["status"]["enum"] == [
        "extracted",
        "partial",
        "missing",
        "contradicted",
    ]
    assert actor_0.metadata["role_output_contract"] == {}
    assert theater_0.metadata["role_output_contract"] == {}
    assert mutation_proposal.metadata["operator_implementation"] == "deterministic_prompt_baseline"
    mutation_operator_config = mutation_proposal.config_snapshot["mutation_operator"]["config"]
    assert mutation_operator_config["focus_point_count"] == 2
    assert mutation_application.metadata["mutation_operator_id"] == "deterministic_prompt_baseline"
    model_pool = config.model_pools["small_local_gemma"]
    actor_model_config = actor_0.metadata["model_config"]
    theater_model_config = theater_0.metadata["theater_model_config"]
    assert actor_model_config["model_id"] in model_pool.models
    assert actor_model_config["model"] in {"gemma4:e2b", "gemma4:e4b"}
    assert actor_model_config["model_selection"]["candidate_ids"] == model_pool.models
    assert actor_model_config["model_selection"]["selected_id"] == actor_model_config["model_id"]
    assert actor_model_config["model_selection"]["selector_context"]["stage_type"] == "actor_turn"
    assert theater_model_config["model_id"] in model_pool.models
    assert theater_model_config["model_selection"]["selector_context"]["stage_type"] == (
        "theater_turn"
    )
    assert theater_model_config["endpoint"]["id"] == "local_ollama"
    assert theater_model_config["endpoint"]["endpoint_selection"]["selected_id"] == "local_ollama"
    assert actor_0.config_snapshot["capability_profile"] == "prompt_only_ollama_structured"
    capability_snapshot = actor_0.config_snapshot["capability_profile_snapshot"]
    assert capability_snapshot["provider_requirements"] == {"provider": "ollama_openai"}
    # Structured profile inherits prompt_only_ollama and adds response_format
    assert "response_format" in capability_snapshot["allowed_controls"]
    assert "persona_text" in capability_snapshot["allowed_controls"]
    assert actor_0.requested_controls["transcript_policy"] == "actor_running_window"
    assert actor_0.requested_controls["sampling"] == {"temperature": 0.8}
    assert grader.requested_controls["plain_output_instructions"] == {
        "value": "Return JSON with rating and justification fields."
    }
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


def test_static_run_planner_fans_out_active_mutation_operators() -> None:
    config = load_project_config_dir("configs")
    config.mutation_operators["deterministic_prompt_variant"] = config.mutation_operators[
        "deterministic_prompt_baseline"
    ].model_copy(update={"config": {"focus_point_count": 1}})
    config.runs["baseline_prompt_only"].active_mutation_operators = [
        "deterministic_prompt_baseline",
        "deterministic_prompt_variant",
    ]
    planner = StaticRunPlanner(config)

    stages = planner.plan_missing_stages("baseline_prompt_only")

    mutation_proposals = [
        stage for stage in stages if stage.stage_type == "mutation_proposal"
    ]
    assert [stage.metadata["mutation_operator_id"] for stage in mutation_proposals] == [
        "deterministic_prompt_baseline",
        "deterministic_prompt_variant",
    ]
    assert mutation_proposals[1].config_snapshot["mutation_operator"]["config"] == {
        "focus_point_count": 1
    }


def test_static_run_planner_honors_weighted_model_and_endpoint_selection() -> None:
    config = load_project_config_dir("configs")
    config.runtime_endpoints["backup_ollama"] = config.runtime_endpoints[
        "local_ollama"
    ].model_copy(update={"base_url": "http://backup.local:11434/v1"})
    config.models["gemma4_e4b"].endpoint_pool = ["local_ollama", "backup_ollama"]
    config.models["gemma4_e4b"].endpoint_selection = "weighted_random"
    config.models["gemma4_e4b"].endpoint_weights = {
        "local_ollama": 0,
        "backup_ollama": 1,
    }
    config.model_pools["small_local_gemma"].selection = "weighted_random"
    config.model_pools["small_local_gemma"].weights = {
        "gemma4_e2b": 0,
        "gemma4_e4b": 1,
    }
    planner = StaticRunPlanner(config)

    stages = planner.plan_missing_stages("baseline_prompt_only")

    for stage in _llm_stages(stages):
        model_config = stage.metadata["model_config"]
        assert model_config["model_id"] == "gemma4_e4b"
        assert model_config["model"] == "gemma4:e4b"
        assert model_config["model_selection"]["strategy"] == "weighted_random"
        assert model_config["model_selection"]["selected_id"] == "gemma4_e4b"
        assert model_config["model_selection"]["weights"] == {
            "gemma4_e2b": 0.0,
            "gemma4_e4b": 1.0,
        }
        assert model_config["endpoint"]["id"] == "backup_ollama"
        assert model_config["endpoint"]["endpoint_selection"]["selected_id"] == (
            "backup_ollama"
        )


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


def _llm_stages(stages):
    return [
        stage
        for stage in stages
        if stage.stage_type in {"actor_turn", "theater_turn", "grader_eval"}
    ]
