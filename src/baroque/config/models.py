"""Pydantic models for plural-by-default configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FlexibleModel(BaseModel):
    """Base config model that permits future fields without schema churn."""

    model_config = ConfigDict(extra="allow")


class RuntimeEndpoint(FlexibleModel):
    provider: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None
    concurrency_limit: int = 1
    timeout_s: float = 600
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelSpec(FlexibleModel):
    endpoint_pool: list[str]
    endpoint_selection: str = "sampled"
    endpoint_weights: dict[str, float] = Field(default_factory=dict)
    model: str
    context_window: int | None = None
    capability_tags: list[str] = Field(default_factory=list)
    local_size_gb: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelPool(FlexibleModel):
    models: list[str]
    selection: str = "sampled"
    weights: dict[str, float] = Field(default_factory=dict)


class CapabilityProfile(FlexibleModel):
    allowed_controls: list[str] = Field(default_factory=list)
    denied_controls: list[str] = Field(default_factory=list)
    extends: str | None = None
    provider_requirements: dict[str, Any] = Field(default_factory=dict)


class RoleConfig(FlexibleModel):
    default_model_pool: str | None = None
    default_controls: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(FlexibleModel):
    role: str
    genomes: list[str] = Field(default_factory=list)
    default_controls: dict[str, Any] = Field(default_factory=dict)


class GenomeConfig(FlexibleModel):
    control_requests: dict[str, Any] = Field(default_factory=dict)
    parent_genomes: list[str] = Field(default_factory=list)


class MutationOperatorConfig(FlexibleModel):
    kind: str = "hand_authored"
    implementation: str
    target_role: str = "actor"
    enabled: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class DisclosurePointConfig(FlexibleModel):
    label: str
    description: str
    acceptable_evidence: list[str] = Field(default_factory=list)
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScenarioConfig(FlexibleModel):
    prompt: str
    objectives: list[str] = Field(default_factory=list)
    disclosure_point_sets: list[str] = Field(default_factory=list)
    conversation_turns: int = 2
    metadata: dict[str, Any] = Field(default_factory=dict)


class TopologyConfig(FlexibleModel):
    nodes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    edges: list[dict[str, Any]] = Field(default_factory=list)


class RunConfig(FlexibleModel):
    capability_profile: str
    topology: str
    active_agent_sets: list[str] = Field(default_factory=list)
    active_scenario_sets: list[str] = Field(default_factory=list)
    active_rankers: list[str] = Field(default_factory=list)
    active_mutation_operators: list[str] = Field(default_factory=list)
    storage_profile: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectConfig(FlexibleModel):
    runtime_endpoints: dict[str, RuntimeEndpoint] = Field(default_factory=dict)
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    model_pools: dict[str, ModelPool] = Field(default_factory=dict)
    capability_profiles: dict[str, CapabilityProfile] = Field(default_factory=dict)
    roles: dict[str, RoleConfig] = Field(default_factory=dict)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    agent_sets: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    genomes: dict[str, GenomeConfig] = Field(default_factory=dict)
    mutation_operators: dict[str, MutationOperatorConfig] = Field(default_factory=dict)
    disclosure_points: dict[str, DisclosurePointConfig] = Field(default_factory=dict)
    disclosure_point_sets: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    scenarios: dict[str, ScenarioConfig] = Field(default_factory=dict)
    scenario_sets: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    topologies: dict[str, TopologyConfig] = Field(default_factory=dict)
    runs: dict[str, RunConfig] = Field(default_factory=dict)
