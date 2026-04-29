from baroque.config.loader import expand_env_values, load_project_config_dir


def test_default_configs_load() -> None:
    config = load_project_config_dir("configs")

    assert "baseline_prompt_only" in config.runs
    assert "prompt_only_ollama" in config.capability_profiles
    assert "gemma4_e2b" in config.models
    assert "starter_questioning_strategy" in config.scenarios
    assert config.runs["baseline_prompt_only"].topology == "actor_theater_grader"
    assert config.runtime_endpoints["local_ollama"].base_url == "http://localhost:11434/v1"


def test_expand_env_values_uses_environment_and_defaults(monkeypatch) -> None:
    monkeypatch.setenv("BAROQUE_TEST_VALUE", "from-env")

    expanded = expand_env_values(
        {
            "present": "${BAROQUE_TEST_VALUE:-default}",
            "missing": "${BAROQUE_MISSING_TEST_VALUE:-default}",
            "embedded": "prefix-${BAROQUE_TEST_VALUE}-suffix",
        }
    )

    assert expanded == {
        "present": "from-env",
        "missing": "default",
        "embedded": "prefix-from-env-suffix",
    }
