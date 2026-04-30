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

        stages: list[StageSpec] = []
        for scenario_id in scenario_ids:
            scenario = self._require_key(self._config.scenarios, scenario_id, "scenario")
            disclosure_points = self._disclosure_points_for_scenario(scenario)
            theater_model_config = self._model_config_for_role("theater")
            for actor_id, actor in actor_agents.items():
                actor_genome_id = self._first_genome_id(actor_id, actor)
                actor_genome = self._require_key(self._config.genomes, actor_genome_id, "genome")
                actor_model_config = self._model_config_for_agent(actor)
                for grader_id, grader in grader_agents.items():
                    grader_genome_id = self._first_genome_id(grader_id, grader)
                    grader_genome = self._require_key(
                        self._config.genomes,
                        grader_genome_id,
                        "genome",
                    )
                    grader_model_config = self._model_config_for_agent(grader)
                    for rollout_index in range(rollout_replicates):
                        sample_id = self._sample_id(
                            run_id,
                            scenario_id,
                            actor_id,
                            grader_id,
                            rollout_index,
                        )

                        conversation = StageSpec(
                            stage_type="actor_theater_conversation",
                            run_id=run_id,
                            sample_id=sample_id,
                            config_snapshot={
                                "capability_profile": run.capability_profile,
                                "topology": run.topology,
                                "scenario_id": scenario_id,
                                "scenario": scenario.model_dump(mode="json"),
                                "actor_id": actor_id,
                                "actor": actor.model_dump(mode="json"),
                                "actor_genome_id": actor_genome_id,
                                "actor_genome": actor_genome.model_dump(mode="json"),
                            },
                            requested_controls=actor_genome.control_requests,
                            metadata={
                                "role": "actor",
                                "agent_id": actor_id,
                                "genome_id": actor_genome_id,
                                "scenario_id": scenario_id,
                                "rollout_index": rollout_index,
                                "conversation_turns": scenario.conversation_turns,
                                "model_config": actor_model_config,
                                "actor_model_config": actor_model_config,
                                "theater_model_config": theater_model_config,
                            },
                        )
                        stages.append(conversation)
                        grader_hashes: list[str] = []
                        for assessment_index in range(assessment_replicates):
                            grader_eval = StageSpec(
                                stage_type="grader_eval",
                                run_id=run_id,
                                sample_id=sample_id,
                                parent_hashes=[conversation.deterministic_hash()],
                                config_snapshot={
                                    "capability_profile": run.capability_profile,
                                    "topology": run.topology,
                                    "scenario_id": scenario_id,
                                    "scenario": scenario.model_dump(mode="json"),
                                    "disclosure_points": disclosure_points,
                                    "grader_id": grader_id,
                                    "grader": grader.model_dump(mode="json"),
                                    "grader_genome_id": grader_genome_id,
                                    "grader_genome": grader_genome.model_dump(mode="json"),
                                },
                                requested_controls=grader_genome.control_requests,
                                metadata={
                                    "role": "grader",
                                    "agent_id": grader_id,
                                    "genome_id": grader_genome_id,
                                    "scenario_id": scenario_id,
                                    "rollout_index": rollout_index,
                                    "assessment_index": assessment_index,
                                    "conversation_hash": conversation.deterministic_hash(),
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
                                "conversation_hash": conversation.deterministic_hash(),
                            },
                        )
                        stages.append(assessment_aggregate)
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

    def _model_config_for_agent(self, agent: AgentConfig) -> dict[str, Any]:
        return self._model_config_for_role(agent.role)

    def _model_config_for_role(self, role: str) -> dict[str, Any]:
        role_config = self._require_key(self._config.roles, role, "role")
        model_pool_id = role_config.default_model_pool
        if model_pool_id is None:
            raise ValueError(f"role has no default model pool: {role}")
        model_pool = self._require_key(self._config.model_pools, model_pool_id, "model pool")
        if not model_pool.models:
            raise ValueError(f"model pool has no models: {model_pool_id}")
        model_id = model_pool.models[0]
        model = self._require_key(self._config.models, model_id, "model")
        if not model.endpoint_pool:
            raise ValueError(f"model has no endpoint pool: {model_id}")
        endpoint_id = model.endpoint_pool[0]
        endpoint = self._require_key(self._config.runtime_endpoints, endpoint_id, "endpoint")
        return {
            "model_id": model_id,
            "model": model.model,
            "model_pool_id": model_pool_id,
            "endpoint": {
                "id": endpoint_id,
                "provider": endpoint.provider,
                "base_url": endpoint.base_url,
                "api_key": endpoint.api_key,
                "timeout_s": endpoint.timeout_s,
            },
        }

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
    def _require_role(topology: TopologyGraph, role: str) -> None:
        if not topology.role_nodes(role):
            raise ValueError(f"topology has no node for role: {role}")

    @staticmethod
    def _require_key(mapping: dict[str, Any], key: str, label: str) -> Any:
        try:
            return mapping[key]
        except KeyError as exc:
            raise ValueError(f"unknown {label}: {key}") from exc
