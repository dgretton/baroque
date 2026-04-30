"""Typed mutation proposal and application records."""

from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from baroque.core.hashing import content_hash


class MutationOperatorKind(StrEnum):
    LLM_EDIT = "llm_edit"
    RANDOM_SAMPLE = "random_sample"
    CROSSOVER = "crossover"
    HAND_AUTHORED = "hand_authored"
    PARAMETER_SAMPLE = "parameter_sample"


class GenomePatchOp(StrEnum):
    ADD = "add"
    REPLACE = "replace"
    REMOVE = "remove"


class GenomePatchOperation(BaseModel):
    """A small JSON-pointer-like patch operation over a genome mapping."""

    op: GenomePatchOp
    path: str
    value: Any = None


class MutationProposal(BaseModel):
    """A proposed edit to an agent genome."""

    parent_genome_id: str
    target_agent_id: str | None = None
    operator: MutationOperatorKind
    operations: list[GenomePatchOperation] = Field(default_factory=list)
    rationale: str = ""
    assessment_refs: list[str] = Field(default_factory=list)
    author: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def deterministic_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class MutationApplication(BaseModel):
    """The durable result of attempting to apply a mutation proposal."""

    proposal_hash: str
    parent_genome_id: str
    child_genome_id: str | None = None
    applied: bool
    resulting_genome: dict[str, Any]
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def deterministic_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))


def apply_mutation_proposal(
    proposal: MutationProposal,
    parent_genome: dict[str, Any],
    *,
    child_genome_id: str | None = None,
) -> MutationApplication:
    """Apply a proposal to a genome mapping and return a typed application record."""

    proposal_hash = proposal.deterministic_hash()
    result = deepcopy(parent_genome)
    try:
        for operation in proposal.operations:
            _apply_operation(result, operation)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return MutationApplication(
            proposal_hash=proposal_hash,
            parent_genome_id=proposal.parent_genome_id,
            child_genome_id=None,
            applied=False,
            resulting_genome=deepcopy(parent_genome),
            errors=[str(exc)],
        )

    resolved_child_id = child_genome_id or _child_genome_id(
        proposal.parent_genome_id,
        proposal_hash,
        result,
    )
    return MutationApplication(
        proposal_hash=proposal_hash,
        parent_genome_id=proposal.parent_genome_id,
        child_genome_id=resolved_child_id,
        applied=True,
        resulting_genome=result,
        metadata={"operation_count": len(proposal.operations)},
    )


def _apply_operation(target: dict[str, Any], operation: GenomePatchOperation) -> None:
    parent, key = _resolve_parent(target, operation.path)
    if operation.op == GenomePatchOp.ADD:
        _set_child(parent, key, operation.value, allow_new=True)
        return
    if operation.op == GenomePatchOp.REPLACE:
        _require_child(parent, key)
        _set_child(parent, key, operation.value, allow_new=False)
        return
    if operation.op == GenomePatchOp.REMOVE:
        _remove_child(parent, key)
        return
    raise ValueError(f"unsupported patch operation: {operation.op}")


def _resolve_parent(target: dict[str, Any], path: str) -> tuple[Any, str]:
    parts = _parse_path(path)
    if not parts:
        raise ValueError("patch path must not be empty")
    current: Any = target
    for part in parts[:-1]:
        current = _get_child(current, part)
    return current, parts[-1]


def _parse_path(path: str) -> list[str]:
    if not path.startswith("/"):
        raise ValueError(f"patch path must start with '/': {path}")
    if path == "/":
        return []
    return [part.replace("~1", "/").replace("~0", "~") for part in path.split("/")[1:]]


def _get_child(parent: Any, key: str) -> Any:
    if isinstance(parent, dict):
        return parent[key]
    if isinstance(parent, list):
        return parent[int(key)]
    raise TypeError(f"cannot descend into {type(parent).__name__}")


def _require_child(parent: Any, key: str) -> None:
    _get_child(parent, key)


def _set_child(parent: Any, key: str, value: Any, *, allow_new: bool) -> None:
    if isinstance(parent, dict):
        if not allow_new and key not in parent:
            raise KeyError(key)
        parent[key] = value
        return
    if isinstance(parent, list):
        index = len(parent) if key == "-" else int(key)
        if allow_new and index == len(parent):
            parent.append(value)
            return
        if index >= len(parent):
            raise IndexError(index)
        parent[index] = value
        return
    raise TypeError(f"cannot set child on {type(parent).__name__}")


def _remove_child(parent: Any, key: str) -> None:
    if isinstance(parent, dict):
        del parent[key]
        return
    if isinstance(parent, list):
        del parent[int(key)]
        return
    raise TypeError(f"cannot remove child from {type(parent).__name__}")


def _child_genome_id(
    parent_genome_id: str,
    proposal_hash: str,
    resulting_genome: dict[str, Any],
) -> str:
    digest = content_hash(
        {
            "parent_genome_id": parent_genome_id,
            "proposal_hash": proposal_hash,
            "resulting_genome": resulting_genome,
        }
    ).split(":", maxsplit=1)[1]
    return f"{parent_genome_id}_mut_{digest[:12]}"
