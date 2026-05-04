# Stacked Agent Proof Of Concept

This is a sidecar demo for the May 2026 progress presentation. It is not the
main Baroque runtime and should stay easy to delete or replace.

## Shape

The script runs a three-level stack:

1. Actor asks the Theater questions to extract disclosure points.
2. Assessor/Adjuster grades extraction and revises the Actor prompt.
3. Meta-assessor grades that assessment and revises the Assessor prompt.

Each iteration saves the transcript, assessment, meta-assessment, and resulting
prompt genomes. The point is to show the recursive theater idea working at toy
scale before the durable lineage/iteration planner is ready.

## Commands

Mock mode is deterministic and needs no model:

```bash
python3 scripts/stacked_agents_poc.py --mock
```

Ollama mode uses the OpenAI-compatible endpoint:

```bash
python3 scripts/stacked_agents_poc.py \
  --base-url http://localhost:11434/v1 \
  --model gemma4:e2b \
  --iterations 2 \
  --turns 3
```

On the Linux machine, set `BAROQUE_LLM_BASE_URL` or pass `--base-url` if Ollama
is listening somewhere other than `localhost:11434`.

The output JSON defaults to:

```text
runs/stacked_agents_poc/latest.json
```
