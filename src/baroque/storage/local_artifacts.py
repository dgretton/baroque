"""Local filesystem content-addressed artifact store."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

from baroque.core.hashing import sha256_bytes
from baroque.core.models import ArtifactRef


class LocalArtifactStore:
    """Store immutable artifacts under a local content-addressed path."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    async def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str,
        suffix: str = "",
    ) -> ArtifactRef:
        return await asyncio.to_thread(
            self._put_bytes_sync,
            data,
            media_type=media_type,
            suffix=suffix,
        )

    async def get_bytes(self, ref: ArtifactRef) -> bytes:
        return await asyncio.to_thread(Path(ref.uri).read_bytes)

    async def exists(self, ref: ArtifactRef) -> bool:
        return await asyncio.to_thread(Path(ref.uri).exists)

    def _put_bytes_sync(self, data: bytes, *, media_type: str, suffix: str) -> ArtifactRef:
        digest = sha256_bytes(data)
        normalized_suffix = self._normalize_suffix(suffix)
        directory = self._root / "sha256" / digest[:2] / digest[2:4]
        directory.mkdir(parents=True, exist_ok=True)

        final_path = directory / f"{digest}{normalized_suffix}"
        if not final_path.exists():
            temp_path = directory / f".{digest}.{uuid4().hex}.tmp"
            try:
                with temp_path.open("wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, final_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()

        return ArtifactRef(
            uri=str(final_path),
            content_hash=f"sha256:{digest}",
            media_type=media_type,
            size_bytes=len(data),
        )

    @staticmethod
    def _normalize_suffix(suffix: str) -> str:
        if not suffix:
            return ""
        return suffix if suffix.startswith(".") else f".{suffix}"

