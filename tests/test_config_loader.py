from baroque.config.loader import load_project_config_dir


def test_default_configs_load() -> None:
    config = load_project_config_dir("configs")

    assert "baseline_prompt_only" in config.runs
    assert "prompt_only_ollama" in config.capability_profiles
    assert "gemma4_e2b" in config.models
    assert config.runs["baseline_prompt_only"].topology == "actor_theater_grader"

