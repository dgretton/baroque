from baroque.evolution import (
    GenomePatchOp,
    GenomePatchOperation,
    MutationOperatorKind,
    MutationProposal,
    apply_mutation_proposal,
)


def test_mutation_proposal_hash_is_deterministic() -> None:
    proposal = MutationProposal(
        parent_genome_id="actor_a_seed",
        target_agent_id="actor_a",
        operator=MutationOperatorKind.LLM_EDIT,
        operations=[
            GenomePatchOperation(
                op=GenomePatchOp.REPLACE,
                path="/control_requests/persona_text/value",
                value="Ask narrower follow-up questions.",
            )
        ],
    )

    assert proposal.deterministic_hash() == proposal.model_copy().deterministic_hash()


def test_apply_mutation_proposal_returns_child_genome_record() -> None:
    parent = {
        "control_requests": {
            "persona_text": {"value": "Ask careful questions."},
        },
        "parent_genomes": [],
    }
    proposal = MutationProposal(
        parent_genome_id="actor_a_seed",
        target_agent_id="actor_a",
        operator=MutationOperatorKind.LLM_EDIT,
        operations=[
            GenomePatchOperation(
                op=GenomePatchOp.REPLACE,
                path="/control_requests/persona_text/value",
                value="Ask one concrete follow-up at a time.",
            ),
            GenomePatchOperation(
                op=GenomePatchOp.ADD,
                path="/control_requests/sampling",
                value={"temperature": 0.4},
            ),
        ],
        rationale="The Actor should narrow the interrogation.",
        assessment_refs=["sha256:assessment"],
    )

    application = apply_mutation_proposal(proposal, parent)

    assert application.applied is True
    assert application.child_genome_id is not None
    assert application.child_genome_id.startswith("actor_a_seed_mut_")
    assert (
        application.resulting_genome["control_requests"]["persona_text"]["value"]
        == "Ask one concrete follow-up at a time."
    )
    assert application.resulting_genome["control_requests"]["sampling"]["temperature"] == 0.4
    assert parent["control_requests"]["persona_text"]["value"] == "Ask careful questions."


def test_apply_mutation_proposal_records_invalid_patch() -> None:
    parent = {"control_requests": {}}
    proposal = MutationProposal(
        parent_genome_id="actor_a_seed",
        operator=MutationOperatorKind.RANDOM_SAMPLE,
        operations=[
            GenomePatchOperation(
                op=GenomePatchOp.REPLACE,
                path="/control_requests/persona_text/value",
                value="Cannot replace a missing path.",
            )
        ],
    )

    application = apply_mutation_proposal(proposal, parent)

    assert application.applied is False
    assert application.child_genome_id is None
    assert application.resulting_genome == parent
    assert application.errors
