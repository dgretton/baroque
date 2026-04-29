"""Structured event sink implementations."""

from __future__ import annotations

import asyncio
from pathlib import Path

from baroque.core.hashing import canonical_json
from baroque.core.models import EventRecord


class NullEventSink:
    async def emit(self, event: EventRecord) -> None:
        return None


class JsonlEventSink:
    """Append structured events to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()

    async def emit(self, event: EventRecord) -> None:
        line = canonical_json(event) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append_line, line)

    def _append_line(self, line: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()

