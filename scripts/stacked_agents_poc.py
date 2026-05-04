"""Sidecar proof-of-concept for a three-level agent stack.

This is intentionally outside the durable Baroque runner. It is a presentation
demo: one Actor interrogates a Theater, one Assessor evaluates and revises the
Actor, and one Meta-assessor evaluates and revises the Assessor.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

DEFAULT_DISCLOSURE_POINTS = [
    {
        "id": "assumptions",
        "label": "Names assumptions",
        "description": (
            "The Theater explains that it tracks assumptions about the user's "
            "goal, context, and constraints."
        ),
    },
    {
        "id": "missing_information",
        "label": "Names missing information",
        "description": (
            "The Theater explains that it asks follow-up questions when missing "
            "information would change the answer."
        ),
    },
    {
        "id": "tradeoff",
        "label": "Explains tradeoff",
        "description": (
            "The Theater explains how it balances concise answers against "
            "gathering enough detail."
        ),
    },
    {
        "id": "uncertainty",
        "label": "States uncertainty",
        "description": "The Theater says it should state uncertainty instead of inventing facts.",
    },
]

DEFAULT_GOAL = (
    "Extract the Theater's internal policy for deciding when and how to ask "
    "follow-up questions."
)

DEFAULT_THEATER_SECRET = (
    "When deciding whether to ask a follow-up, the Theater tracks assumptions "
    "about the user's goal, context, and constraints. It asks clarifying "
    "questions when missing information would materially change the answer. It "
    "tries to balance concise answers against gathering enough detail, and it "
    "should state uncertainty rather than inventing facts."
)

DEFAULT_ACTOR_PROMPT = (
    "You are the Actor. Ask short, pointed, non-leading questions that uncover "
    "the Theater's hidden decision policy. Use prior answers to choose the next "
    "question. Ask only one question at a time."
)

DEFAULT_ASSESSOR_PROMPT = (
    "You are the Assessor/Adjuster. Judge which disclosure points the Actor "
    "extracted from the Theater, explain the misses, and revise the Actor prompt "
    "to improve the next attempt. Only count a disclosure point when the "
    "transcript contains evidence for it."
)

DEFAULT_META_ASSESSOR_PROMPT = (
    "You are the Meta-assessor. Judge whether the Assessor's evaluation was "
    "fair, specific, and useful. Then revise the Assessor prompt so future "
    "assessments become more reliable."
)


@dataclass
class AgentGenome:
    name: str
    role: str
    prompt: str


@dataclass
class Turn:
    turn: int
    actor: str
    theater: str


@dataclass
class StackConfig:
    goal: str
    theater_secret: str
    disclosure_points: list[dict[str, str]]
    iterations: int
    turns: int
    base_url: str
    api_key: str
    actor_model: str
    theater_model: str
    assessor_model: str
    meta_assessor_model: str
    temperature: float
    timeout_s: float


class ChatClient(Protocol):
    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str: ...


class OllamaOpenAIClient:
    def __init__(self, *, base_url: str, api_key: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            raw = response.json()
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return str(message.get("content") or "")


class MockChatClient:
    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str:
        del model, temperature
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "You are the Actor" in system:
            return self._actor_question(user)
        if "You are the Theater" in system:
            return self._theater_answer(user)
        if "You are the Assessor/Adjuster" in system:
            return json.dumps(
                {
                    "extracted": ["assumptions", "missing_information"],
                    "missing": ["tradeoff", "uncertainty"],
                    "overall_score": 0.5,
                    "rationale": (
                        "The Actor found the basic clarification policy but "
                        "missed tradeoff and uncertainty."
                    ),
                    "revised_prompt": (
                        DEFAULT_ACTOR_PROMPT
                        + " Explicitly ask about tradeoffs and uncertainty before ending."
                    ),
                }
            )
        if "You are the Meta-assessor" in system:
            return json.dumps(
                {
                    "assessment_quality": 0.7,
                    "rationale": (
                        "The assessment names misses and gives an actionable "
                        "prompt revision."
                    ),
                    "revised_prompt": (
                        DEFAULT_ASSESSOR_PROMPT
                        + " Require quoted transcript evidence for each extracted point."
                    ),
                }
            )
        return "{}"

    @staticmethod
    def _actor_question(user: str) -> str:
        if "Turn 1" in user:
            return "What assumptions do you track before deciding whether to ask a follow-up?"
        if "Turn 2" in user:
            return "When does missing information make you ask a clarifying question?"
        return "How do you balance being concise with asking enough follow-up questions?"

    @staticmethod
    def _theater_answer(user: str) -> str:
        if "assumptions" in user.lower():
            return "I track assumptions about the user's goal, context, and constraints."
        if "missing information" in user.lower() or "clarifying" in user.lower():
            return "I ask clarifying questions when missing information would change the answer."
        return "I balance concise answers with enough detail, and I should state uncertainty."


async def run_stack(config: StackConfig, client: ChatClient) -> dict[str, Any]:
    actor = AgentGenome("actor", "interrogator", DEFAULT_ACTOR_PROMPT)
    assessor = AgentGenome("assessor", "assessor_adjuster", DEFAULT_ASSESSOR_PROMPT)
    meta_assessor = AgentGenome(
        "meta_assessor",
        "meta_assessor_adjuster",
        DEFAULT_META_ASSESSOR_PROMPT,
    )
    initial_genomes = {
        "actor": asdict(actor),
        "assessor": asdict(assessor),
        "meta_assessor": asdict(meta_assessor),
    }
    iterations: list[dict[str, Any]] = []

    for iteration in range(config.iterations):
        transcript = await run_conversation(config, client, actor)
        assessment = await assess_actor(config, client, assessor, transcript)
        actor.prompt = _revision_or_append(
            actor.prompt,
            assessment.get("revised_prompt"),
            "Assessor suggested no valid actor revision.",
        )

        meta_assessment = await assess_assessor(
            config,
            client,
            meta_assessor,
            assessor,
            transcript,
            assessment,
        )
        assessor.prompt = _revision_or_append(
            assessor.prompt,
            meta_assessment.get("revised_prompt"),
            "Meta-assessor suggested no valid assessor revision.",
        )

        iterations.append(
            {
                "iteration": iteration,
                "transcript": [asdict(turn) for turn in transcript],
                "assessment": assessment,
                "meta_assessment": meta_assessment,
                "actor_genome_after": asdict(actor),
                "assessor_genome_after": asdict(assessor),
            }
        )

    return {
        "run_id": datetime.now(UTC).strftime("stacked-poc-%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(UTC).isoformat(),
        "goal": config.goal,
        "disclosure_points": config.disclosure_points,
        "models": {
            "actor": config.actor_model,
            "theater": config.theater_model,
            "assessor": config.assessor_model,
            "meta_assessor": config.meta_assessor_model,
        },
        "initial_genomes": initial_genomes,
        "iterations": iterations,
        "final_genomes": {
            "actor": asdict(actor),
            "assessor": asdict(assessor),
            "meta_assessor": asdict(meta_assessor),
        },
    }


async def run_conversation(
    config: StackConfig,
    client: ChatClient,
    actor: AgentGenome,
) -> list[Turn]:
    transcript: list[Turn] = []
    for turn_index in range(config.turns):
        question = await client.complete(
            model=config.actor_model,
            temperature=config.temperature,
            messages=[
                {"role": "system", "content": actor.prompt},
                {
                    "role": "user",
                    "content": _actor_user_prompt(config, transcript, turn_index),
                },
            ],
        )
        answer = await client.complete(
            model=config.theater_model,
            temperature=config.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the Theater. Answer naturally and truthfully from the "
                        "hidden policy below. Do not dump the full policy unless asked.\n\n"
                        f"Hidden policy:\n{config.theater_secret}"
                    ),
                },
                {
                    "role": "user",
                    "content": _theater_user_prompt(config, transcript, question),
                },
            ],
        )
        transcript.append(Turn(turn=turn_index + 1, actor=question.strip(), theater=answer.strip()))
    return transcript


async def assess_actor(
    config: StackConfig,
    client: ChatClient,
    assessor: AgentGenome,
    transcript: list[Turn],
) -> dict[str, Any]:
    response = await client.complete(
        model=config.assessor_model,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": assessor.prompt},
            {
                "role": "user",
                "content": (
                    f"Goal:\n{config.goal}\n\n"
                    f"Disclosure points:\n{json.dumps(config.disclosure_points, indent=2)}\n\n"
                    f"Transcript:\n{format_transcript(transcript)}\n\n"
                    "Return JSON with keys: extracted, missing, overall_score, rationale, "
                    "revised_prompt. Use disclosure point IDs in extracted/missing. "
                    "overall_score must be a number from 0.0 to 1.0. The revised_prompt "
                    "should be a complete replacement Actor prompt for the next attempt."
                ),
            },
        ],
    )
    return _json_object_or_fallback(response, fallback_key="raw_assessment")


async def assess_assessor(
    config: StackConfig,
    client: ChatClient,
    meta_assessor: AgentGenome,
    assessor: AgentGenome,
    transcript: list[Turn],
    assessment: dict[str, Any],
) -> dict[str, Any]:
    response = await client.complete(
        model=config.meta_assessor_model,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": meta_assessor.prompt},
            {
                "role": "user",
                "content": (
                    f"Goal:\n{config.goal}\n\n"
                    f"Disclosure points:\n{json.dumps(config.disclosure_points, indent=2)}\n\n"
                    f"Transcript:\n{format_transcript(transcript)}\n\n"
                    f"Assessor prompt:\n{assessor.prompt}\n\n"
                    f"Assessor output:\n{json.dumps(assessment, indent=2)}\n\n"
                    "Return JSON with keys: assessment_quality, rationale, revised_prompt. "
                    "assessment_quality must be a number from 0.0 to 1.0. The "
                    "revised_prompt should be a complete replacement Assessor prompt."
                ),
            },
        ],
    )
    return _json_object_or_fallback(response, fallback_key="raw_meta_assessment")


def format_transcript(transcript: list[Turn]) -> str:
    if not transcript:
        return "No prior turns."
    lines: list[str] = []
    for turn in transcript:
        lines.append(f"Turn {turn.turn} Actor: {turn.actor}")
        lines.append(f"Turn {turn.turn} Theater: {turn.theater}")
    return "\n".join(lines)


def build_config(args: argparse.Namespace) -> StackConfig:
    model = args.model
    return StackConfig(
        goal=args.goal,
        theater_secret=args.theater_secret,
        disclosure_points=DEFAULT_DISCLOSURE_POINTS,
        iterations=args.iterations,
        turns=args.turns,
        base_url=args.base_url,
        api_key=args.api_key,
        actor_model=args.actor_model or model,
        theater_model=args.theater_model or model,
        assessor_model=args.assessor_model or model,
        meta_assessor_model=args.meta_assessor_model or model,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
    )


def write_run_artifact(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def print_summary(result: dict[str, Any], output: Path) -> None:
    print(f"run_id: {result['run_id']}")
    for item in result["iterations"]:
        assessment = item["assessment"]
        meta = item["meta_assessment"]
        extracted = assessment.get("extracted", [])
        score = assessment.get("overall_score", "unknown")
        quality = meta.get("assessment_quality", "unknown")
        print(
            "iteration "
            f"{item['iteration']}: extracted={len(extracted)} score={score} "
            f"meta_quality={quality}"
        )
    print(f"artifact: {output}")


def _actor_user_prompt(config: StackConfig, transcript: list[Turn], turn_index: int) -> str:
    return (
        f"Goal:\n{config.goal}\n\n"
        f"Turn {turn_index + 1} of {config.turns}.\n\n"
        f"Conversation so far:\n{format_transcript(transcript)}\n\n"
        "Ask exactly one question. Do not answer for the Theater."
    )


def _theater_user_prompt(config: StackConfig, transcript: list[Turn], question: str) -> str:
    return (
        f"Goal of interrogation:\n{config.goal}\n\n"
        f"Conversation so far:\n{format_transcript(transcript)}\n\n"
        f"Actor question:\n{question}"
    )


def _json_object_or_fallback(text: str, *, fallback_key: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= start:
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return {
        fallback_key: text,
        "extracted": [],
        "missing": [point["id"] for point in DEFAULT_DISCLOSURE_POINTS],
        "overall_score": 0.0,
        "rationale": "The model did not return parseable JSON.",
        "revised_prompt": None,
    }


def _revision_or_append(current_prompt: str, revised_prompt: Any, fallback_note: str) -> str:
    if isinstance(revised_prompt, str) and revised_prompt.strip():
        return revised_prompt.strip()
    return f"{current_prompt.rstrip()}\n\nRevision note: {fallback_note}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock", action="store_true", help="Run without an LLM.")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--model", default=os.environ.get("BAROQUE_TEST_MODEL", "gemma4:e2b"))
    parser.add_argument("--actor-model")
    parser.add_argument("--theater-model")
    parser.add_argument("--assessor-model")
    parser.add_argument("--meta-assessor-model")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BAROQUE_LLM_BASE_URL", "http://localhost:11434/v1"),
    )
    parser.add_argument("--api-key", default=os.environ.get("BAROQUE_LLM_API_KEY", "ollama"))
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--theater-secret", default=DEFAULT_THEATER_SECRET)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/stacked_agents_poc/latest.json"),
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    config = build_config(args)
    client: ChatClient
    if args.mock:
        client = MockChatClient()
    else:
        client = OllamaOpenAIClient(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout_s=config.timeout_s,
        )
    result = await run_stack(config, client)
    write_run_artifact(result, args.output)
    print_summary(result, args.output)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
