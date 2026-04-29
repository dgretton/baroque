# Baroque

Baroque is a research framework for evolving multi-agent LLM interrogators.

The codebase starts with a local-first, async Python architecture:

- plural configuration files for runs, models, roles, topologies, and capability profiles
- typed system boundaries for planning, leasing, inference, artifacts, events, and exports
- content-addressed artifacts and deterministic hashes
- local-first storage, with a path toward multi-runner and cloud execution

See `docs/` for the design notes that guide the implementation.

## Local Checks

```bash
python3 -m pytest
python3 -m ruff check .
python3 -m compileall src tests
```
