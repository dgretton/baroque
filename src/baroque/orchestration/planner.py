"""Static run planner for the first vertical slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from baroque.config.models import AgentConfig, ProjectConfig, TopologyConfig
from baroque.core.hashing import content_hash
from baroque.core.interfaces import StageStore
from baroque.core.models import StageRecord, StageSpec


@dataclass(frozen=True)
class TopologyGraph:
    """Validated role topology.

    This graph describes role communication/evaluation structure, not strict
    stage dependencies. Interaction graphs may contain cycles; planned stage
    dependencies must still be acyclic through `parent_hashes`.
    """

    topology_id: str
    nodes: dict[str, dict[str, Any]]
    edges: list[dict[str, Any]]

    @classmethod
    def from_config(cls, topology_id: str, config: TopologyConfig) -> TopologyGraph:
        graph = cls(topology_id=topology_id, nodes=config.nodes, edges=config.edges)
        graph.validate()
        return graph

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError(f"topology has no nodes: {self.topology_id}")
        for edge in self.edges:
            source = edge.get("from")
            target = edge.get("to")
            if source not in self.nodes:
                raise ValueError(f"topology edge references unknown source node: {source}")
            if target not in self.nodes:
                raise ValueError(f"topology edge references unknown target node: {target}")

    def role_nodes(self, role: str) -> list[str]:
        return [node_id for node_id, node in self.nodes.items() if node.get("role") == role]


class StaticRunPlanner:
    """Plan deterministic seed stages from config.

    The first useful vertical slice is one Actor-Theater conversation followed by
    one Grader evaluation per configured Actor/Grader/scenario combination.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self._config = config

    async def seed_run(self, stage_store: StageStore, run_id: str) -> list[StageRecord]:
        """Plan and insert seed stages for a run."""

        records: list[StageRecord] = []
        for stage in self.plan_missing_stages(run_id):
            records.append(await stage_store.add_stage(stage))
        return records

    def plan_missing_stages(self, run_id: str) -> list[StageSpec]:
        run = self._require_key(self._config.runs, run_id, "run")
        topology_config = self._require_key(self._config.topologies, run.topology, "topology")
        topology = TopologyGraph.from_config(run.topology, topology_config)
        self._require_role(topology, "actor")
        self._require_role(topology, "theater")
        self._require_role(topology, "grader")

        actor_agents = self._agents_for_role(run.active_agent_sets, "actor")
        grader_agents = self._agents_for_role(run.active_agent_sets, "grader")
        scenario_ids = self._scenario_ids(run.active_scenario_sets)
        rollout_replicates = self._positive_int(
            run.metadata.get("rollout_replicates", 1),
            "rollout_replicates",
        )
        assessment_replicates = self._positive_int(
            run.metadata.get("assessment_replicates", 1),
            "assessment_replicates",
        )
        mutation_replicates = self._nonnegative_int(
            run.metadata.get("mutation_replicates", 1),
            "mutation_replicates",
        )
        capability_profile_snapshot = self._capability_profile_snapshot(run.capability_profile)

        stages: list[StageSpec] = []
        for scenario_id in scenario_ids:
            scenario = self._require_key(self._config.scenarios, scenario_id, "scenario")
            disclosure_points = self._disclosure_points_for_scenario(scenario)
            for actor_id, actor in actor_agents.items():
                actor_genome_id = self._first_genome_id(actor_id, actor)
                actor_genome = self._require_key(self._config.genomes, actor_genome_id, "genome")
                actor_controls = self._controls_for_agent(actor, actor_genome)
                for grader_id, grader in grader_agents.items():
                    grader_genome_id = self._first_genome_id(grader_id, grader)
                    grader_genome = self._require_key(
                        self._config.genomes,
                        grader_genome_id,
                        "genome",
                    )
                    grader_controls = self._controls_for_agent(grader, grader_genome)
                    for rollout_index in range(rollout_replicates):
                        sample_id = self._sample_id(
                            run_id,
                            scenario_id,
                            actor_id,
                            grader_id,
                            rollout_index,
                        )

                        turn_hashes: list[str] = []
                        previous_theater_hash: str | None = None
                        for turn_index in range(scenario.conversation_turns):
                            actor_model_config = self._model_config_for_agent(
                                actor,
                                selector_context={
                                    "run_id": run_id,
                                    "sample_id": sample_id,
                                    "stage_type": "actor_turn",
                                    "role": "actor",
                                    "agent_id": actor_id,
                                    "genome_id": actor_genome_id,
                                    "scenario_id": scenario_id,
                                    "rollout_index": rollout_index,
                                    "turn_index": turn_index,
                                },
                            )
                            actor_turn = StageSpec(
                                stage_type="actor_turn",
                                run_id=run_id,
                                sample_id=sample_id,
                                parent_hashes=(
                                    [previous_theater_hash] if previous_theater_hash else []
                                ),
                                config_snapshot={
                                    "capability_profile": run.capability_profile,
                                    "capability_profile_snapshot": capability_profile_snapshot,
                                    "topology": run.topology,
                                    "scenario_id": scenario_id,
                                    "scenario": scenario.model_dump(mode="json"),
                                    "actor_id": actor_id,
                                    "actor": actor.model_dump(mode="json"),
                                    "actor_genome_id": actor_genome_id,
                                    "actor_genome": actor_genome.model_dump(mode="json"),
                                },
                                requested_controls=actor_controls,
                                metadata={
                                    "role": "actor",
                                    "agent_id": actor_id,
                                    "genome_id": actor_genome_id,
                                    "scenario_id": scenario_id,
                                    "rollout_index": rollout_index,
                                    "turn_index": turn_index,
                                    "conversation_turns": scenario.conversation_turns,
                                    "model_config": actor_model_config,
                                    "actor_model_config": actor_model_config,
                                },
                            )
                            stages.append(actor_turn)
                            turn_hashes.append(actor_turn.deterministic_hash())

                            theater_model_config = self._model_config_for_role(
                                "theater",
                                selector_context={
                                    "run_id": run_id,
                                    "sample_id": sample_id,
                                    "stage_type": "theater_turn",
                                    "role": "theater",
                                    "actor_id": actor_id,
                                    "grader_id": grader_id,
                                    "scenario_id": scenario_id,
                                    "rollout_index": rollout_index,
                                    "turn_index": turn_index,
                                },
                            )
                            theater_turn = StageSpec(
                                stage_type="theater_turn",
                                run_id=run_id,
                                sample_id=sample_id,
                                parent_hashes=[actor_turn.deterministic_hash()],
                                config_snapshot={
                                    "capability_profile": run.capability_profile,
                                    "capability_profile_snapshot": capability_profile_snapshot,
                                    "topology": run.topology,
                                    "scenario_id": scenario_id,
                                    "scenario": scenario.model_dump(mode="json"),
                                },
                                metadata={
                                    "role": "theater",
                                    "scenario_id": scenario_id,
                                    "rollout_index": rollout_index,
                                    "turn_index": turn_index,
                                    "conversation_turns": scenario.conversation_turns,
                                    "actor_turn_hash": actor_turn.deterministic_hash(),
                                    "model_config": theater_model_config,
                                    "theater_model_config": theater_model_config,
                                },
                            )
                            stages.append(theater_turn)
                            turn_hashes.append(theater_turn.deterministic_hash())
                            previous_theater_hash = theater_turn.deterministic_hash()

                        conversation_transcript = StageSpec(
                            stage_type="conversation_transcript",
                            run_id=run_id,
                            sample_id=sample_id,
                            parent_hashes=turn_hashes,
                            config_snapshot={
                                "capability_profile": run.capability_profile,
                                "capability_profile_snapshot": capability_profile_snapshot,
                                "topology": run.topology,
                                "scenario_id": scenario_id,
                                "scenario": scenario.model_dump(mode="json"),
                                "actor_id": actor_id,
                                "actor": actor.model_dump(mode="json"),
                                "actor_genome_id": actor_genome_id,
                                "actor_genome": actor_genome.model_dump(mode="json"),
                            },
                            metadata={
                                "role": "conversation_builder",
                                "agent_id": actor_id,
                                "genome_id": actor_genome_id,
                                "scenario_id": scenario_id,
                                "rollout_index": rollout_index,
                                "conversation_turns": scenario.conversation_turns,
                            },
                        )
                        stages.append(conversation_transcript)
                        conversation_hash = conversation_transcript.deterministic_hash()
                        grader_hashes: list[str] = []
                        for assessment_index in range(assessment_replicates):
                            grader_model_config = self._model_config_for_agent(
                                grader,
                                selector_context={
                                    "run_id": run_id,
                                    "sample_id": sample_id,
                                    "stage_type": "grader_eval",
                                    "role": "grader",
                                    "agent_id": grader_id,
                                    "genome_id": grader_genome_id,
                                    "scenario_id": scenario_id,
                                    "rollout_index": rollout_index,
                                    "assessment_index": assessment_index,
                                },
                            )
                            grader_eval = StageSpec(
                                stage_type="grader_eval",
                                run_id=run_id,
                                sample_id=sample_id,
                                parent_hashes=[conversation_hash],
                                config_snapshot={
                                    "capability_profile": run.capability_profile,
                                    "capability_profile_snapshot": capability_profile_snapshot,
                                    "topology": run.topology,
                                    "scenario_id": scenario_id,
                                    "scenario": scenario.model_dump(mode="json"),
                                    "disclosure_points": disclosure_points,
                                    "grader_id": grader_id,
                                    "grader": grader.model_dump(mode="json"),
                                    "grader_genome_id": grader_genome_id,
                                    "grader_genome": grader_genome.model_dump(mode="json"),
                                },
                                requested_controls=grader_controls,
                                metadata={
                                    "role": "grader",
                                    "agent_id": grader_id,
                                    "genome_id": grader_genome_id,
                                    "scenario_id": scenario_id,
                                    "rollout_index": rollout_index,
                                    "assessment_index": assessment_index,
                                    "conversation_hash": conversation_hash,
                                    "model_config": grader_model_config,
                                },
                            )
                            stages.append(grader_eval)
                            grader_hashes.append(grader_eval.deterministic_hash())

                        assessment_aggregate = StageSpec(
                            stage_type="assessment_aggregate",
                            run_id=run_id,
                            sample_id=sample_id,
                            parent_hashes=grader_hashes,
                            config_snapshot={
                                "capability_profile": run.capability_profile,
                                "capability_profile_snapshot": capability_profile_snapshot,
                                "topology": run.topology,
                                "scenario_id": scenario_id,
                                "scenario": scenario.model_dump(mode="json"),
                                "disclosure_points": disclosure_points,
                                "actor_id": actor_id,
                                "actor": actor.model_dump(mode="json"),
                                "actor_genome_id": actor_genome_id,
                                "actor_genome": actor_genome.model_dump(mode="json"),
                                "grader_id": grader_id,
                                "grader": grader.model_dump(mode="json"),
                                "grader_genome_id": grader_genome_id,
                                "grader_genome": grader_genome.model_dump(mode="json"),
                            },
                            metadata={
                                "role": "assessment_aggregator",
                                "scenario_id": scenario_id,
                                "rollout_index": rollout_index,
                                "assessment_count": assessment_replicates,
                                "conversation_hash": conversation_hash,
                            },
                        )
                        stages.append(assessment_aggregate)
                        for mutation_index in range(mutation_replicates):
                            mutation_proposal = StageSpec(
                                stage_type="mutation_proposal",
                                run_id=run_id,
                                sample_id=sample_id,
                                parent_hashes=[assessment_aggregate.deterministic_hash()],
                                config_snapshot={
                                    "capability_profile": run.capability_profile,
                                    "capability_profile_snapshot": capability_profile_snapshot,
                                    "topology": run.topology,
                                    "scenario_id": scenario_id,
                                    "scenario": scenario.model_dump(mode="json"),
                                    "disclosure_points": disclosure_points,
                                    "actor_id": actor_id,
                                    "actor": actor.model_dump(mode="json"),
                                    "actor_genome_id": actor_genome_id,
                                    "actor_genome": actor_genome.model_dump(mode="json"),
                                    "grader_id": grader_id,
                                    "grader": grader.model_dump(mode="json"),
                                    "grader_genome_id": grader_genome_id,
                                    "grader_genome": grader_genome.model_dump(mode="json"),
                                },
                                metadata={
                                    "role": "mutator",
                                    "scenario_id": scenario_id,
                                    "rollout_index": rollout_index,
                                    "mutation_index": mutation_index,
                                    "conversation_hash": conversation_hash,
                                    "assessment_aggregate_hash": (
                                        assessment_aggregate.deterministic_hash()
                                    ),
                                    "operator": "deterministic_prompt_baseline",
                                },
                            )
                            stages.append(mutation_proposal)
                            mutation_application = StageSpec(
                                stage_type="mutation_application",
                                run_id=run_id,
                                sample_id=sample_id,
                                parent_hashes=[mutation_proposal.deterministic_hash()],
                                config_snapshot={
                                    "capability_profile": run.capability_profile,
                                    "capability_profile_snapshot": capability_profile_snapshot,
                                    "topology": run.topology,
                                    "scenario_id": scenario_id,
                                    "scenario": scenario.model_dump(mode="json"),
                                    "disclosure_points": disclosure_points,
                                    "actor_id": actor_id,
                                    "actor": actor.model_dump(mode="json"),
                                    "actor_genome_id": actor_genome_id,
                                    "actor_genome": actor_genome.model_dump(mode="json"),
                                },
                                metadata={
                                    "role": "mutation_applicator",
                                    "scenario_id": scenario_id,
                                    "rollout_index": rollout_index,
                                    "mutation_index": mutation_index,
                                    "conversation_hash": conversation_hash,
                                    "mutation_proposal_hash": (
                                        mutation_proposal.deterministic_hash()
                                    ),
                                },
                            )
                            stages.append(mutation_application)
        return stages

    def _agents_for_role(
        self,
        agent_set_ids: list[str],
        role: str,
    ) -> dict[str, AgentConfig]:
        selected: dict[str, AgentConfig] = {}
        for agent_set_id in agent_set_ids:
            agent_set = self._require_key(self._config.agent_sets, agent_set_id, "agent set")
            for agent_id in agent_set.get("agents", []):
                agent = self._require_key(self._config.agents, agent_id, "agent")
                if agent.role == role:
                    selected[agent_id] = agent
        if not selected:
            raise ValueError(f"run has no agents for role: {role}")
        return selected

    def _scenario_ids(self, scenario_set_ids: list[str]) -> list[str]:
        scenario_ids: list[str] = []
        for scenario_set_id in scenario_set_ids:
            scenario_set = self._require_key(
                self._config.scenario_sets,
                scenario_set_id,
                "scenario set",
            )
            scenario_ids.extend(scenario_set.get("scenarios", []))
        if not scenario_ids:
            raise ValueError("run has no scenarios")
        return scenario_ids

    def _capability_profile_snapshot(
        self,
        profile_id: str,
        seen: set[str] | None = None,
    ) -> dict[str, Any]:
        seen = seen or set()
        if profile_id in seen:
            raise ValueError(f"capability profile extends cycle: {profile_id}")
        seen.add(profile_id)

        profile = self._require_key(
            self._config.capability_profiles,
            profile_id,
            "capability profile",
        )
        if profile.extends:
            parent = self._capability_profile_snapshot(profile.extends, seen)
            child_allowed = list(profile.allowed_controls)
            denied_controls = _unique(
                [
                    *[
                        control
                        for control in parent["denied_controls"]
                        if control not in set(child_allowed)
                    ],
                    *profile.denied_controls,
                ]
            )
            allowed_controls = _unique([*parent["allowed_controls"], *child_allowed])
            allowed_controls = [
                control for control in allowed_controls if control not in set(denied_controls)
            ]
            provider_requirements = _deep_merge(
                parent["provider_requirements"],
                profile.provider_requirements,
            )
        else:
            allowed_controls = _unique(profile.allowed_controls)
            denied_controls = _unique(profile.denied_controls)
            provider_requirements = dict(profile.provider_requirements)

        return {
            "id": profile_id,
            "extends": profile.extends,
            "allowed_controls": allowed_controls,
            "denied_controls": denied_controls,
            "provider_requirements": provider_requirements,
        }

    def _controls_for_agent(self, agent: AgentConfig, genome: Any) -> dict[str, Any]:
        role_config = self._require_key(self._config.roles, agent.role, "role")
        return _deep_merge(
            role_config.default_controls,
            agent.default_controls,
            genome.control_requests,
        )

    def _model_config_for_agent(
        self,
        agent: AgentConfig,
        *,
        selector_context: dict[str, Any],
    ) -> dict[str, Any]:
        return self._model_config_for_role(agent.role, selector_context=selector_context)

    def _model_config_for_role(
        self,
        role: str,
        *,
        selector_context: dict[str, Any],
    ) -> dict[str, Any]:
        role_config = self._require_key(self._config.roles, role, "role")
        model_pool_id = role_config.default_model_pool
        if model_pool_id is None:
            raise ValueError(f"role has no default model pool: {role}")
        model_pool = self._require_key(self._config.model_pools, model_pool_id, "model pool")
        if not model_pool.models:
            raise ValueError(f"model pool has no models: {model_pool_id}")
        model_id, model_selection = self._select_from_pool(
            model_pool.models,
            selection=model_pool.selection,
            weights=model_pool.weights,
            selector_context=selector_context | {"pool_id": model_pool_id},
            label="model",
        )
        model = self._require_key(self._config.models, model_id, "model")
        if not model.endpoint_pool:
            raise ValueError(f"model has no endpoint pool: {model_id}")
        endpoint_id, endpoint_selection = self._select_from_pool(
            model.endpoint_pool,
            selection=model.endpoint_selection,
            weights=model.endpoint_weights,
            selector_context=selector_context | {"model_id": model_id},
            label="endpoint",
        )
        endpoint = self._require_key(self._config.runtime_endpoints, endpoint_id, "endpoint")
        return {
            "model_id": model_id,
            "model": model.model,
            "model_pool_id": model_pool_id,
            "model_selection": model_selection,
            "context_window": model.context_window,
            "capability_tags": model.capability_tags,
            "local_size_gb": model.local_size_gb,
            "endpoint": {
                "id": endpoint_id,
                "provider": endpoint.provider,
                "base_url": endpoint.base_url,
                "api_key": endpoint.api_key,
                "concurrency_limit": endpoint.concurrency_limit,
                "timeout_s": endpoint.timeout_s,
                "endpoint_selection": endpoint_selection,
            },
        }

    def _select_from_pool(
        self,
        candidates: list[str],
        *,
        selection: str,
        weights: dict[str, float],
        selector_context: dict[str, Any],
        label: str,
    ) -> tuple[str, dict[str, Any]]:
        if not candidates:
            raise ValueError(f"{label} pool has no candidates")

        strategy = selection.lower().replace("-", "_")
        draw_hash = content_hash(
            {
                "label": label,
                "strategy": strategy,
                "candidates": candidates,
                "selector_context": selector_context,
            }
        )
        draw_fraction = self._hash_fraction(draw_hash)

        if strategy in {"first", "fixed", "ordered", "primary"}:
            selected_index = 0
            effective_weights: dict[str, float] | None = None
        elif strategy in {"sampled", "uniform", "uniform_random", "random"}:
            selected_index = min(int(draw_fraction * len(candidates)), len(candidates) - 1)
            effective_weights = None
        elif strategy in {"weighted", "weighted_random", "weighted_sampled"}:
            effective_weights = self._effective_weights(candidates, weights, label)
            selected_index = self._weighted_index(candidates, effective_weights, draw_fraction)
        else:
            raise ValueError(f"unsupported {label} selection strategy: {selection}")

        selected_id = candidates[selected_index]
        record: dict[str, Any] = {
            "strategy": selection,
            "normalized_strategy": strategy,
            "candidate_ids": list(candidates),
            "selected_id": selected_id,
            "selected_index": selected_index,
            "draw_hash": draw_hash,
            "draw_fraction": draw_fraction,
            "selector_context": selector_context,
        }
        if effective_weights is not None:
            record["weights"] = effective_weights
        return selected_id, record

    @staticmethod
    def _effective_weights(
        candidates: list[str],
        weights: dict[str, float],
        label: str,
    ) -> dict[str, float]:
        effective_weights: dict[str, float] = {}
        for candidate in candidates:
            weight = float(weights.get(candidate, 1.0))
            if weight < 0:
                raise ValueError(f"{label} selection weight cannot be negative: {candidate}")
            effective_weights[candidate] = weight
        if sum(effective_weights.values()) <= 0:
            raise ValueError(f"{label} selection weights must include a positive value")
        return effective_weights

    @staticmethod
    def _weighted_index(
        candidates: list[str],
        weights: dict[str, float],
        draw_fraction: float,
    ) -> int:
        total_weight = sum(weights[candidate] for candidate in candidates)
        threshold = draw_fraction * total_weight
        cumulative = 0.0
        for index, candidate in enumerate(candidates):
            cumulative += weights[candidate]
            if threshold < cumulative:
                return index
        return len(candidates) - 1

    @staticmethod
    def _hash_fraction(hash_value: str) -> float:
        digest = hash_value.split(":", maxsplit=1)[1]
        return int(digest[:16], 16) / 16**16

    def _disclosure_points_for_scenario(self, scenario: Any) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        for point_set_id in scenario.disclosure_point_sets:
            point_set = self._require_key(
                self._config.disclosure_point_sets,
                point_set_id,
                "disclosure point set",
            )
            for point_id in point_set.get("disclosure_points", []):
                point = self._require_key(
                    self._config.disclosure_points,
                    point_id,
                    "disclosure point",
                )
                points.append({"id": point_id, **point.model_dump(mode="json")})
        return points

    @staticmethod
    def _first_genome_id(agent_id: str, agent: AgentConfig) -> str:
        if not agent.genomes:
            raise ValueError(f"agent has no genomes: {agent_id}")
        return agent.genomes[0]

    @staticmethod
    def _sample_id(
        run_id: str,
        scenario_id: str,
        actor_id: str,
        grader_id: str,
        rollout_index: int,
    ) -> str:
        hash_value = content_hash(
            {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "actor_id": actor_id,
                "grader_id": grader_id,
                "rollout_index": rollout_index,
            }
        )
        return "sample-" + hash_value.split(":", maxsplit=1)[1][:16]

    @staticmethod
    def _positive_int(value: Any, label: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise ValueError(f"{label} must be at least 1")
        return parsed

    @staticmethod
    def _nonnegative_int(value: Any, label: str) -> int:
        parsed = int(value)
        if parsed < 0:
            raise ValueError(f"{label} must be at least 0")
        return parsed

    @staticmethod
    def _require_role(topology: TopologyGraph, role: str) -> None:
        if not topology.role_nodes(role):
            raise ValueError(f"topology has no node for role: {role}")

    @staticmethod
    def _require_key(mapping: dict[str, Any], key: str, label: str) -> Any:
        try:
            return mapping[key]
        except KeyError as exc:
            raise ValueError(f"unknown {label}: {key}") from exc


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def _deep_merge(*mappings: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            existing = merged.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                merged[key] = _deep_merge(existing, value)
            else:
                merged[key] = value
    return merged
