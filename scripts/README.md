# Scripts

Thin operational and analysis entrypoints live here when they are useful, but
not yet part of the durable runner.

## Three-Level Agent PoC

`stacked_agents_poc.py` is a sidecar presentation demo. It runs:

1. an Actor interrogating a Theater,
2. an Assessor/Adjuster evaluating and revising the Actor,
3. a Meta-assessor evaluating and revising the Assessor.

Fast local mock run:

```bash
python3 scripts/stacked_agents_poc.py --mock
```

Ollama run:

```bash
python3 scripts/stacked_agents_poc.py \
  --base-url http://localhost:11434/v1 \
  --model gemma4:e2b \
  --iterations 2 \
  --turns 3
```

The script writes a JSON artifact to `runs/stacked_agents_poc/latest.json` by
default. The `runs/` directory is ignored by git.
