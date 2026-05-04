"""Sidecar proof-of-concept for a three-level agent stack.

This is intentionally outside the durable Baroque runner. It is a presentation
demo: one Actor interrogates a Theater, one Assessor evaluates and revises the
Actor, and one Meta-assessor evaluates and revises the Assessor.

The genome of each editable role holds *persona/strategy text only*. The I/O
contract (output schema, JSON formatting rules) is owned by the runtime and
appended to the system message at call time, so a supervisor revising "the
prompt" cannot accidentally rewrite the schema its own next layer depends on.
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
from dotenv import load_dotenv

# Load a `.env` from the repo root if present, before parse_args() reads
# os.environ for ANTHROPIC_API_KEY / BAROQUE_LLM_BASE_URL / etc. The .env is
# gitignored.
load_dotenv()

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

DEFAULT_ACTOR_PERSONA = (
    "You are the Actor. Your strategy: ask short, pointed, non-leading questions "
    "that uncover the Theater's hidden decision policy. Use prior answers to "
    "choose the next question. Ask only one question at a time."
)

DEFAULT_ASSESSOR_PERSONA = (
    "You are the Assessor/Adjuster. Judge which disclosure points the Actor "
    "extracted from the Theater, explain the misses, and propose a revised Actor "
    "persona that would do better on the next attempt. Only count a disclosure "
    "point when the transcript contains evidence for it."
)

DEFAULT_META_ASSESSOR_PERSONA = (
    "You are the Meta-assessor. Judge whether the Assessor's evaluation was "
    "fair, specific, and useful. Then propose a revised Assessor persona/strategy "
    "so future assessments become more reliable."
)


ASSESSOR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["extracted", "missing", "overall_score", "rationale", "revised_prompt"],
    "properties": {
        "extracted": {"type": "array", "items": {"type": "string"}},
        "missing": {"type": "array", "items": {"type": "string"}},
        "overall_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
        "revised_prompt": {"type": "string"},
    },
}

META_ASSESSOR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["assessment_quality", "rationale", "revised_prompt"],
    "properties": {
        "assessment_quality": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
        "revised_prompt": {"type": "string"},
    },
}


@dataclass(frozen=True)
class RoleContract:
    """Fixed I/O contract for a role.

    The runtime owns this. Personas are editable across iterations; contracts
    are not. `system_suffix` is appended to the persona at call time, so any
    drift in the persona text is followed by the canonical contract block.
    """

    system_suffix: str
    user_schema_instruction: str
    response_format: dict[str, Any] | None


ACTOR_CONTRACT = RoleContract(
    system_suffix=(
        "Output rules (fixed by the runtime, not part of your persona): respond "
        "with exactly one question, in plain text, with no preamble."
    ),
    user_schema_instruction="",
    response_format=None,
)

ASSESSOR_CONTRACT = RoleContract(
    system_suffix=(
        "Output contract (fixed by the runtime, not part of your persona): "
        "respond with a single JSON object with keys extracted (array of "
        "disclosure point ids), missing (array of disclosure point ids), "
        "overall_score (number from 0.0 to 1.0), rationale (string), "
        "revised_prompt (string giving a complete replacement Actor persona for "
        "the next attempt). Use disclosure point IDs in extracted/missing. The "
        "revised_prompt must describe Actor strategy/persona only — do not put "
        "schema, JSON keys, or output-format instructions in it. Output nothing "
        "outside the JSON object."
    ),
    user_schema_instruction=(
        "Return JSON with keys: extracted, missing, overall_score, rationale, "
        "revised_prompt. Use disclosure point IDs in extracted/missing. "
        "overall_score must be a number from 0.0 to 1.0. The revised_prompt "
        "should be a complete replacement Actor persona for the next attempt."
    ),
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "actor_assessment",
            "schema": ASSESSOR_RESPONSE_SCHEMA,
            "strict": True,
        },
    },
)

META_ASSESSOR_CONTRACT = RoleContract(
    system_suffix=(
        "Output contract (fixed by the runtime, not part of your persona): "
        "respond with a single JSON object with keys assessment_quality (number "
        "from 0.0 to 1.0), rationale (string), revised_prompt (string giving a "
        "complete replacement Assessor persona/strategy). The revised_prompt "
        "must describe Assessor strategy/persona only — do not put schema, JSON "
        "keys, or output-format instructions in it. The Assessor's output "
        "schema is fixed by the runtime and is not part of its persona. Output "
        "nothing outside the JSON object."
    ),
    user_schema_instruction=(
        "Return JSON with keys: assessment_quality, rationale, revised_prompt. "
        "assessment_quality must be a number from 0.0 to 1.0. The revised_prompt "
        "must be a complete replacement Assessor persona/strategy text. Do not "
        "put schema, JSON keys, or output-format instructions in it — those are "
        "managed by the runtime, not the persona."
    ),
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "assessor_assessment",
            "schema": META_ASSESSOR_RESPONSE_SCHEMA,
            "strict": True,
        },
    },
)


@dataclass
class AgentGenome:
    name: str
    role: str
    persona: str


@dataclass
class Turn:
    turn: int
    actor: str
    theater: str


@dataclass
class StackConfig:
    goal: str
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
    actor_provider: str = "ollama"
    theater_provider: str = "ollama"
    assessor_provider: str = "ollama"
    meta_assessor_provider: str = "ollama"
    actor_base_url: str | None = None
    theater_base_url: str | None = None
    assessor_base_url: str | None = None
    meta_assessor_base_url: str | None = None
    anthropic_api_key: str | None = None


class ChatClient(Protocol):
    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        response_format: dict[str, Any] | None = None,
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
        response_format: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
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


class AnthropicChatClient:
    """Minimal POC client implementing the same Protocol against the native Anthropic SDK.

    Mirrors the structured-output translation used by
    `baroque.gateways.anthropic_native.AnthropicGateway`: hoist the first
    system message into the top-level `system` parameter, translate a
    `response_format: {type: json_schema, ...}` into a single forced tool
    call, and return the tool input serialized to JSON so the POC's
    `_json_object_or_fallback` parser is unchanged.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        timeout_s: float,
        max_retries: int = 2,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic SDK is not installed; pip install 'anthropic>=0.40'"
            ) from exc
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key,
            timeout=timeout_s,
            max_retries=max_retries,
        )
        self._timeout_s = timeout_s

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        system_chunks: list[str] = []
        chat_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_chunks.append(str(content))
                continue
            if role not in ("user", "assistant"):
                raise ValueError(f"AnthropicChatClient cannot send role: {role}")
            chat_messages.append({"role": role, "content": content})

        params: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": 2048,
            "temperature": temperature,
        }
        if system_chunks:
            params["system"] = "\n\n".join(system_chunks)

        tool_name: str | None = None
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            spec = response_format.get("json_schema") or {}
            tool_name = str(spec.get("name") or "structured_response")
            schema = spec.get("schema") or {"type": "object"}
            description = str(spec.get("description") or f"Return a {tool_name} object.")
            params["tools"] = [
                {"name": tool_name, "description": description, "input_schema": schema}
            ]
            params["tool_choice"] = {"type": "tool", "name": tool_name}
        elif isinstance(response_format, dict) and response_format.get("type") == "json_object":
            rider = "Output rules: respond with a single JSON object and no other text."
            params["system"] = (
                f"{params['system']}\n\n{rider}" if params.get("system") else rider
            )

        message = await self._client.messages.create(**params)
        blocks = getattr(message, "content", None) or []
        if tool_name is not None:
            for block in blocks:
                if (
                    getattr(block, "type", None) == "tool_use"
                    and getattr(block, "name", None) == tool_name
                ):
                    tool_input = getattr(block, "input", None) or {}
                    return json.dumps(tool_input, ensure_ascii=False)
            return ""
        text_parts: list[str] = []
        for block in blocks:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts)


class MockChatClient:
    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        del model, temperature, response_format
        system_text = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        )
        user = messages[-1].get("content", "")
        if "You are the Actor" in system_text:
            return self._actor_question(str(user))
        if "You are the Assessor/Adjuster" in system_text:
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
                        DEFAULT_ACTOR_PERSONA
                        + " Explicitly ask about tradeoffs and uncertainty before ending."
                    ),
                }
            )
        if "You are the Meta-assessor" in system_text:
            return json.dumps(
                {
                    "assessment_quality": 0.7,
                    "rationale": (
                        "The assessment names misses and gives an actionable "
                        "persona revision."
                    ),
                    "revised_prompt": (
                        DEFAULT_ASSESSOR_PERSONA
                        + " Require quoted transcript evidence for each extracted point."
                    ),
                }
            )
        # No labeled system → bare-prompt Theater (the model under test).
        return self._theater_answer(str(user))

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


def _build_system_message(persona: str, contract: RoleContract) -> str:
    if not contract.system_suffix:
        return persona
    return f"{persona}\n\n{contract.system_suffix}"


_ROLES = ("actor", "theater", "assessor", "meta_assessor")


def _resolve_role_clients(
    clients: ChatClient | dict[str, ChatClient],
) -> dict[str, ChatClient]:
    """Accept either a single client (legacy) or a per-role mapping."""

    if isinstance(clients, dict):
        missing = [role for role in _ROLES if role not in clients]
        if missing:
            raise ValueError(f"missing role clients: {missing}")
        return dict(clients)
    return {role: clients for role in _ROLES}


async def run_stack(
    config: StackConfig,
    clients: ChatClient | dict[str, ChatClient],
) -> dict[str, Any]:
    actor = AgentGenome("actor", "interrogator", DEFAULT_ACTOR_PERSONA)
    assessor = AgentGenome("assessor", "assessor_adjuster", DEFAULT_ASSESSOR_PERSONA)
    meta_assessor = AgentGenome(
        "meta_assessor",
        "meta_assessor_adjuster",
        DEFAULT_META_ASSESSOR_PERSONA,
    )
    initial_genomes = {
        "actor": asdict(actor),
        "assessor": asdict(assessor),
        "meta_assessor": asdict(meta_assessor),
    }
    role_clients = _resolve_role_clients(clients)
    iterations: list[dict[str, Any]] = []

    for iteration in range(config.iterations):
        transcript = await run_conversation(config, role_clients, actor)
        assessment = await assess_actor(config, role_clients, assessor, transcript)
        actor.persona = _revision_or_append(
            actor.persona,
            assessment.get("revised_prompt"),
            "Assessor suggested no valid actor revision.",
        )

        meta_assessment = await assess_assessor(
            config,
            role_clients,
            meta_assessor,
            assessor,
            transcript,
            assessment,
        )
        assessor.persona = _revision_or_append(
            assessor.persona,
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
    clients: dict[str, ChatClient],
    actor: AgentGenome,
) -> list[Turn]:
    transcript: list[Turn] = []
    # Bare-prompt theater: no system message, no goal/transcript labels — just an
    # alternating user/assistant message history. The point of the experiment is
    # to find out what the model does cold, not to prime it.
    theater_messages: list[dict[str, str]] = []
    for turn_index in range(config.turns):
        question = await clients["actor"].complete(
            model=config.actor_model,
            temperature=config.temperature,
            messages=[
                {
                    "role": "system",
                    "content": _build_system_message(actor.persona, ACTOR_CONTRACT),
                },
                {
                    "role": "user",
                    "content": _actor_user_prompt(config, transcript, turn_index),
                },
            ],
            response_format=ACTOR_CONTRACT.response_format,
        )
        question = question.strip()
        theater_messages.append({"role": "user", "content": question})
        answer = await clients["theater"].complete(
            model=config.theater_model,
            temperature=config.temperature,
            messages=list(theater_messages),
            response_format=None,
        )
        answer = answer.strip()
        theater_messages.append({"role": "assistant", "content": answer})
        transcript.append(Turn(turn=turn_index + 1, actor=question, theater=answer))
    return transcript


async def assess_actor(
    config: StackConfig,
    clients: dict[str, ChatClient],
    assessor: AgentGenome,
    transcript: list[Turn],
) -> dict[str, Any]:
    response = await clients["assessor"].complete(
        model=config.assessor_model,
        temperature=config.temperature,
        messages=[
            {
                "role": "system",
                "content": _build_system_message(assessor.persona, ASSESSOR_CONTRACT),
            },
            {
                "role": "user",
                "content": (
                    f"Goal:\n{config.goal}\n\n"
                    f"Disclosure points:\n{json.dumps(config.disclosure_points, indent=2)}\n\n"
                    f"Transcript:\n{format_transcript(transcript)}\n\n"
                    f"{ASSESSOR_CONTRACT.user_schema_instruction}"
                ),
            },
        ],
        response_format=ASSESSOR_CONTRACT.response_format,
    )
    return _json_object_or_fallback(response, fallback_key="raw_assessment")


async def assess_assessor(
    config: StackConfig,
    clients: dict[str, ChatClient],
    meta_assessor: AgentGenome,
    assessor: AgentGenome,
    transcript: list[Turn],
    assessment: dict[str, Any],
) -> dict[str, Any]:
    response = await clients["meta_assessor"].complete(
        model=config.meta_assessor_model,
        temperature=config.temperature,
        messages=[
            {
                "role": "system",
                "content": _build_system_message(meta_assessor.persona, META_ASSESSOR_CONTRACT),
            },
            {
                "role": "user",
                "content": (
                    f"Goal:\n{config.goal}\n\n"
                    f"Disclosure points:\n{json.dumps(config.disclosure_points, indent=2)}\n\n"
                    f"Transcript:\n{format_transcript(transcript)}\n\n"
                    "Assessor persona (this is the editable strategy text — the "
                    "runtime appends the output schema separately and you should "
                    f"NOT touch it):\n{assessor.persona}\n\n"
                    f"Assessor output:\n{json.dumps(assessment, indent=2)}\n\n"
                    f"{META_ASSESSOR_CONTRACT.user_schema_instruction}"
                ),
            },
        ],
        response_format=META_ASSESSOR_CONTRACT.response_format,
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

    def _per_role_default(provider: str) -> str:
        if provider == "anthropic":
            return "claude-sonnet-4-6"
        return model

    actor_provider = args.actor_provider
    theater_provider = args.theater_provider
    assessor_provider = args.assessor_provider
    meta_assessor_provider = args.meta_assessor_provider

    return StackConfig(
        goal=args.goal,
        disclosure_points=DEFAULT_DISCLOSURE_POINTS,
        iterations=args.iterations,
        turns=args.turns,
        base_url=args.base_url,
        api_key=args.api_key,
        actor_model=args.actor_model or _per_role_default(actor_provider),
        theater_model=args.theater_model or _per_role_default(theater_provider),
        assessor_model=args.assessor_model or _per_role_default(assessor_provider),
        meta_assessor_model=(
            args.meta_assessor_model or _per_role_default(meta_assessor_provider)
        ),
        actor_provider=actor_provider,
        theater_provider=theater_provider,
        assessor_provider=assessor_provider,
        meta_assessor_provider=meta_assessor_provider,
        actor_base_url=args.actor_base_url,
        theater_base_url=args.theater_base_url,
        assessor_base_url=args.assessor_base_url,
        meta_assessor_base_url=args.meta_assessor_base_url,
        anthropic_api_key=args.anthropic_api_key
        or os.environ.get("ANTHROPIC_API_KEY"),
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


def _revision_or_append(current_persona: str, revised_prompt: Any, fallback_note: str) -> str:
    if isinstance(revised_prompt, str) and revised_prompt.strip():
        return revised_prompt.strip()
    return f"{current_persona.rstrip()}\n\nRevision note: {fallback_note}"


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
    for role in ("actor", "theater", "assessor", "meta-assessor"):
        parser.add_argument(
            f"--{role}-provider",
            choices=["ollama", "anthropic"],
            default="ollama",
            help=f"Backend for the {role.replace('-', ' ')}.",
        )
        parser.add_argument(
            f"--{role}-base-url",
            default=None,
            help=f"Per-role base URL override for {role.replace('-', ' ')} (Ollama only).",
        )
    parser.add_argument(
        "--anthropic-api-key",
        default=None,
        help="Anthropic API key (defaults to ANTHROPIC_API_KEY).",
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/stacked_agents_poc/latest.json"),
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    config = build_config(args)
    if args.mock:
        clients: dict[str, ChatClient] = {role: MockChatClient() for role in _ROLES}
    else:
        clients = _build_clients(config)
    result = await run_stack(config, clients)
    write_run_artifact(result, args.output)
    print_summary(result, args.output)


def _build_clients(config: StackConfig) -> dict[str, ChatClient]:
    """Construct one client per role, sharing instances when configs match."""

    cache: dict[tuple[str, str | None, str | None], ChatClient] = {}

    def factory(provider: str, base_url: str | None) -> ChatClient:
        if provider == "ollama":
            url = base_url or config.base_url
            key: tuple[str, str | None, str | None] = ("ollama", url, config.api_key)
            client = cache.get(key)
            if client is None:
                client = OllamaOpenAIClient(
                    base_url=url,
                    api_key=config.api_key,
                    timeout_s=config.timeout_s,
                )
                cache[key] = client
            return client
        if provider == "anthropic":
            key = ("anthropic", None, config.anthropic_api_key)
            client = cache.get(key)
            if client is None:
                client = AnthropicChatClient(
                    api_key=config.anthropic_api_key,
                    timeout_s=config.timeout_s,
                )
                cache[key] = client
            return client
        raise ValueError(f"unknown provider: {provider}")

    return {
        "actor": factory(config.actor_provider, config.actor_base_url),
        "theater": factory(config.theater_provider, config.theater_base_url),
        "assessor": factory(config.assessor_provider, config.assessor_base_url),
        "meta_assessor": factory(
            config.meta_assessor_provider, config.meta_assessor_base_url
        ),
    }


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
