"""Deterministic hashing helpers for content-addressed records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _normalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, Mapping):
        return {str(key): _normalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence):
        return [_normalize(item) for item in value]
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialize a value into stable JSON suitable for hashing."""

    return json.dumps(
        _normalize(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def content_hash(value: Any, *, algorithm: str = "sha256") -> str:
    if algorithm != "sha256":
        raise ValueError(f"unsupported hash algorithm: {algorithm}")
    return f"{algorithm}:{sha256_text(canonical_json(value))}"
