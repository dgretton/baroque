# Development

Updated: 2026-04-29

## Local Environment

Use Python 3.12 or newer. The current development machine is running Python 3.13, and the scaffold is tested there.

Install the package and development dependencies:

```bash
python3 -m pip install -e ".[dev]"
```

## Checks

```bash
python3 -m pytest
python3 -m ruff check .
python3 -m compileall src tests
```

## Runtime Data

Local runtime data belongs under `data/`, which is gitignored:

```text
data/
  runtime/
  artifacts/
  logs/
  parquet/
```

The first runtime implementation uses DuckDB for local stage/lease metadata and a filesystem content-addressed artifact store.

Retryable stage failures remain in `failed_retryable` until their retry delay expires. This prevents a failing handler from spinning forever inside a local `run_until_idle` call.

The current vertical slice is tested with a fake inference gateway. That verifies config loading, planning, stage dependency gating, handler dispatch, artifact writes, and completion records without requiring Ollama to be running.

## Integration Tests

The first integration target is a real Ollama/OpenAI-compatible endpoint running the prompt-only vertical slice. It is skipped by default.

```bash
BAROQUE_RUN_OLLAMA_INTEGRATION=1 \
BAROQUE_TEST_MODEL=gemma4:e2b \
python3 -m pytest tests/integration/test_ollama_vertical_slice.py
```

`BAROQUE_TEST_MODEL` is optional and defaults to the first model in the small local Gemma pool.
