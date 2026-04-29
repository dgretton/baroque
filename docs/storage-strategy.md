# Storage Strategy

Updated: 2026-04-29

Source details: `research/sources/source-index.md`

## Goal

The storage system should make ordinary chaos boring:

- process killed during an inference call
- laptop sleeps or loses power
- Ollama hangs
- a cloud worker is preempted
- a retry returns a different answer
- config changes halfway through a long run
- two workers try to claim the same stage
- logs are huge and impossible to read linearly
- a paper-writing question appears months later

The answer is not one magical database. Use different storage layers for different kinds of truth:

```text
Config storage
  durable intent: what was supposed to happen

Runtime storage
  operational truth: what is pending, leased, running, complete, failed

Logging storage
  narrative evidence: what happened, when, and with what context

Results storage
  research data: immutable artifacts and analysis-ready tables
```

No layer should be the only place the system knows something essential unless that is its explicit responsibility.

## Config Storage

Config answers: "What was this run allowed and instructed to do?"

Store human-authored config in version-controlled files:

- `configs/models/*.yaml`
- `configs/capability_profiles/*.yaml`
- `configs/topologies/*.yaml`
- `configs/runs/*.yaml`
- `configs/rankers/*.yaml`
- `configs/mutation_operators/*.yaml`
- `configs/storage/*.yaml`

At run start, resolve layered defaults into an immutable run snapshot:

```text
project defaults
  -> run defaults
    -> capability profile defaults
      -> topology defaults
        -> role defaults
          -> agent defaults
            -> genome fields
              -> sample/stage overrides
```

Persist both:

- `config_source_refs`: file paths, git commit, dirty flag, and config hashes.
- `resolved_config_snapshot`: the actual config object used after defaults and environment references are resolved.

Important rules:

- Never rely on live config files to interpret old results.
- Store secret references, not secret values.
- Hash configs after canonical serialization.
- Version config schemas explicitly.
- If config changes during a run, create a new config version; do not mutate the old one.
- Each stage should point to the exact resolved config version it used.

Useful tables:

- `run_configs`
- `capability_profile_versions`
- `topology_versions`
- `model_pool_versions`
- `agent_genome_versions`
- `storage_profile_versions`

This makes "what changed?" answerable without archaeology.

## Runtime Storage

Runtime storage answers: "What work exists, who owns it, and what can safely resume?"

For the local vertical slice, DuckDB can hold runtime state. For multi-worker or cloud execution, use Postgres, Temporal, Prefect, Ray, or another real control plane. Do not use a shared DuckDB file as a distributed work queue.

Minimum runtime concepts:

- `runs`
- `iterations`
- `samples`
- `stages`
- `stage_attempts`
- `leases`
- `stage_dependencies`
- `runner_heartbeats`
- `endpoint_health`
- `retry_policies`

The core invariant:

> A stage is complete only after its output artifacts and success record are both committed.

Suggested lifecycle:

```text
pending
  -> leased
  -> running
  -> succeeded

running
  -> failed_retryable
  -> running after retry delay when reclaimed by a runner

running
  -> failed_terminal

leased/running
  -> abandoned when lease expires
  -> pending
```

Use leases even with one local runner:

- `lease_owner`: runner id
- `leased_until`: timestamp
- `heartbeat_at`: timestamp
- `attempt`: integer

This makes a single-process runner structurally compatible with future distributed workers.

Runtime state should be mostly small metadata. Large prompts, responses, transcripts, screenshots, or logs belong in the artifact store and are referenced by content hash/path.

Failure handling:

- If a process dies before committing a stage, the lease expires and the stage can run again.
- If a retry produces a different output, that is acceptable; the successful attempt gets its own artifact hash.
- If a duplicate worker completes the same content hash, dedupe by content hash and keep duplicate attempt metadata.
- If storage write fails, the stage is not complete.

## Logging Storage

Logging answers: "What happened in human-readable time?"

Logs are not the source of truth for completed work. They are the evidence trail, debugging surface, and dashboard feed.

Use structured JSON logs from day one. Every log event should carry correlation fields:

- `timestamp`
- `level`
- `event`
- `run_id`
- `runner_id`
- `iteration_id`
- `sample_id`
- `stage_id`
- `content_hash`
- `attempt`
- `role`
- `agent_id`
- `model`
- `endpoint`
- `duration_ms`
- `error_type`

Recommended local layout:

```text
data/
  logs/
    run=<run_id>/
      runner=<runner_id>/
        events-000001.jsonl
```

Use rotation by size or time. Keep logs append-only. For week-long runs, logs will become large enough that "just grep it" should remain possible, but not be the only analysis path.

Also write selected log-like events into structured tables:

- `stage_events`
- `runner_events`
- `endpoint_events`
- `retry_events`
- `shutdown_events`

OpenTelemetry can mirror this as traces and metrics:

- one trace per sample or stage chain
- spans for provider calls, parsing, mutation, ranking, storage writes
- metrics for queue depth, active leases, completions/hour, token throughput, retries, timeouts, endpoint health

Logs should include enough context to debug chaos, but not so much duplicated prompt/response text that the logs become the primary data store. Store raw request/response blobs as artifacts and link to them.

## Results Storage

Results storage answers: "What data did the research produce?"

Use immutable, content-addressed artifacts for raw and semi-structured outputs:

```text
data/
  artifacts/
    sha256/
      ab/
        cd/
          abcdef...json
          abcdef...txt
```

Artifact classes:

- raw provider request
- raw provider response
- parsed structured output
- Actor-Theater transcript
- Grader/Director/Producer/Critic evaluation
- mutation objective
- concrete persona edit
- persona/genome snapshot
- ranking comparison
- error payload

Store artifact metadata in tables:

- `artifacts`
- `artifact_links`
- `personas`
- `genomes`
- `conversations`
- `evaluations`
- `rankings`
- `mutation_records`

Keep analysis-friendly derived data in DuckDB and Parquet:

```text
data/
  parquet/
    stages/run_id=<run_id>/stage_type=<stage_type>/part-000001.parquet
    samples/run_id=<run_id>/part-000001.parquet
    evaluations/run_id=<run_id>/role=<role>/part-000001.parquet
    rankings/run_id=<run_id>/part-000001.parquet
```

Do not force paper analysis to parse raw JSON blobs. Raw artifacts preserve evidence; Parquet tables support iteration, plots, filters, joins, and reproducible analysis.

Suggested split:

- Raw artifacts: immutable, complete, compressed if large.
- Normalized tables: one row per meaningful entity.
- Derived tables/views: recomputable summaries for analysis.
- Export manifests: list exactly which files/tables belong to a paper figure or experiment snapshot.

## Local-To-Cloud Path

Start local:

```text
DuckDB
local filesystem artifact store
JSONL logs
Parquet exports
```

Move to multi-host:

```text
Postgres control plane
S3-compatible artifact/object store
Parquet datasets in object storage
DuckDB/Polars reading from S3-compatible paths
JSON logs shipped or copied to object storage
```

The same logical interfaces should survive:

- `ConfigStore`
- `LeaseStore`
- `ArtifactStore`
- `EventLog`
- `ResultsStore`
- `AnalysisExport`

Only their implementations change.

## Backups And Integrity

For long runs, backup the irreplaceable pieces:

- config snapshots
- runtime metadata
- artifact manifests
- raw request/response artifacts
- result Parquet datasets

Practical local rules:

- Use content hashes for artifacts.
- Write artifacts to a temporary path, fsync/close, then atomically rename.
- Verify artifact hash after write.
- Keep a manifest table with size, hash, media type, created_at.
- Periodically export DuckDB tables to Parquet.
- Periodically copy artifacts and Parquet to external disk or object storage.
- Monitor free disk space.

For cloud rules:

- Use bucket versioning if affordable.
- Keep lifecycle policies conservative until the paper is done.
- Separate "hot runtime state" from "cold research archive."
- Prefer append-only writes and immutable paths.

## What Not To Do

- Do not keep important state only in memory.
- Do not make logs the only durable record.
- Do not overwrite artifacts in place.
- Do not let live config files define past runs.
- Do not store large blobs directly in runtime queue rows.
- Do not require the orchestrator to be running for analysis.
- Do not assume a single machine filesystem once cloud workers are introduced.
- Do not treat retries as invisible; attempts are data.

## Minimal First Implementation

Implemented in the current local scaffold:

- file-based YAML config loading with environment-resolved values
- stage metadata snapshots of the relevant resolved config inputs
- DuckDB tables for stages and attempts
- local filesystem content-addressed artifact store
- JSONL structured logs
- deterministic content hashes
- lease fields on stages
- parent-hash dependency gating
- retry delay for retryable failures

Still next:

- explicit run/sample tables
- artifact manifest tables
- structured event tables in DuckDB
- Parquet export command

This remains the smallest storage path that should survive ordinary chaos without drowning the project in infrastructure.
