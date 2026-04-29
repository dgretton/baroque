import asyncio

from baroque.storage.local_artifacts import LocalArtifactStore


def test_local_artifact_store_round_trips_bytes(tmp_path) -> None:
    async def scenario() -> None:
        store = LocalArtifactStore(tmp_path)
        ref = await store.put_bytes(b"hello", media_type="text/plain", suffix=".txt")

        assert ref.content_hash.startswith("sha256:")
        assert ref.size_bytes == 5
        assert await store.exists(ref)
        assert await store.get_bytes(ref) == b"hello"

    asyncio.run(scenario())

