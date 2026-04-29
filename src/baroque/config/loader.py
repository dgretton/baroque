"""YAML configuration loading helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from baroque.config.models import ProjectConfig


def iter_config_files(root: str | Path) -> list[Path]:
    """Return all YAML config files below a root in deterministic order."""

    root_path = Path(root)
    return sorted([*root_path.rglob("*.yaml"), *root_path.rglob("*.yml")])


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load one YAML file as a mapping."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping: {path}")
    return data


def load_project_config(paths: Iterable[str | Path]) -> ProjectConfig:
    """Load and merge config files into a ProjectConfig."""

    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, load_yaml_file(path))
    return ProjectConfig.model_validate(merged)


def load_project_config_dir(root: str | Path) -> ProjectConfig:
    """Load all YAML files under a config root."""

    return load_project_config(iter_config_files(root))


def deep_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings, with right-side scalars/lists winning."""

    merged = dict(left)
    for key, right_value in right.items():
        left_value = merged.get(key)
        if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
            merged[key] = deep_merge(left_value, right_value)
        else:
            merged[key] = right_value
    return merged

