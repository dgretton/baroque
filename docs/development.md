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
