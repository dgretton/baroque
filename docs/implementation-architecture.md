# Implementation Architecture

Updated: 2026-04-29

## Language Selection

### Core Orchestration: Python 3.12

Python is the right default language for this project:

- native `asyncio` is enough for the first long-running local runner
- the LLM, data, and research ecosystem is Python-first
- Pydantic gives typed config/runtime models without fighting the research workflow
- DuckDB, Polars, PyArrow, httpx, OpenTelemetry, and cloud SDKs are all mature
- experiments and paper analysis can share the same language as the runner

The core package should stay pure Python unless there is a measured bottleneck that Python cannot reasonably handle.

### Configuration: YAML Plus Resolved JSON Snapshots

Human-authored config should be YAML because it is readable and diffable. At runtime, every run should persist a resolved JSON-compatible config snapshot with hashes and source refs. YAML is intent; resolved snapshots are evidence.

### Storage And Analysis: SQL, Parquet, Python

Use DuckDB locally for metadata, stage records, and analysis. Use Parquet for analysis-ready exports. Use Postgres only when the control plane becomes multi-worker or cloud-distributed. Use S3-compatible object storage when artifacts need to leave a single filesystem.

### Inference Backends: OpenAI-Compatible HTTP Boundary

Treat Ollama, exo, vLLM, LiteLLM, cloud endpoints, and remote APIs as inference endpoints behind a common gateway. The runner should not care whether the request is served by one Mac, a Linux workstation, a local cluster, or rented GPUs.

### Dashboard: Python First, Web Later

Start with structured logs, DuckDB queries, and small scripts. If a dashboard becomes necessary, prefer FastAPI plus server-rendered HTML or htmx before reaching for a TypeScript frontend. A larger TypeScript/React dashboard can come later if the interaction complexity earns it.

### Shell, Docker, Kubernetes, Terraform

Use shell scripts only as thin wrappers. Use Docker when cloud workers need reproducible images. Use Kubernetes/Terraform only when infrastructure scale makes them cheaper than manual provisioning.

### Rust/C++/Go

Do not introduce systems languages early. They may become useful for a high-throughput proxy, custom inference hooks, or activation-steering integrations, but the first bottleneck will be model latency rather than Python overhead.

## Repository Structure

```text
configs/
  capability_profiles/   # allowed control surfaces
  models/                # endpoints, models, and model pools
  runs/                  # run-level experiment definitions
  storage/               # local/cloud storage profiles
  topologies/            # role interaction/evaluation graphs

docs/                    # design and architecture documents
research/                # research notes and source index
scripts/                 # thin operational or analysis entrypoints
src/baroque/
  agents/                # role/persona/genome helpers
  builder/               # RoleGenome -> ProviderRequest compilation
  config/                # typed config models and loaders
  core/                  # shared models, hashes, boundary protocols
  evolution/             # mutation, beam, lineage
  gateways/              # Ollama/OpenAI-compatible/local/remote inference
  observability/         # event sinks and telemetry hooks
  orchestration/         # planner, runner, leasing loop
  ranking/               # absolute, depth, pairwise, tournament rankers
  storage/               # DuckDB, artifacts, exports
tests/
```

## System Boundaries

The important interfaces live in `baroque.core.interfaces`:

- `ConfigStore`: loads immutable resolved config snapshots
- `Planner`: resolves missing stages from topology and sample state
- `LeaseStore`: claims, heartbeats, completes, and fails work
- `StageStore`: inserts and looks up planned stages by deterministic content hash
- `ArtifactStore`: persists content-addressed blobs
- `InferenceGateway`: sends provider requests to local or remote endpoints
- `EventSink`: emits structured events/logs/telemetry
- `AnalysisExport`: exports analysis-ready data

These are the boundaries that should survive scaling. Implementations can move from local filesystem/DuckDB to Postgres/S3/Ray/Temporal/cloud workers without changing the research concepts.

## Initial Concrete Pieces

The scaffold includes:

- Pydantic config models with plural maps
- deterministic content hashing
- provider request/response models
- stage and event records
- a simple query builder
- an OpenAI-compatible gateway
- a local content-addressed artifact store
- a DuckDB-backed local runtime/lease store
- JSONL event sink
- prompt-only Ollama config examples

The first runner and prompt-only vertical slice now stand on this foundation. The implementation is still intentionally narrow, but the core nouns and interfaces are live code rather than a paper scaffold.

## Current Implementation Status

Implemented:

- plural Pydantic config models and YAML loading
- deterministic content hashing
- prompt-compatible provider request building
- OpenAI-compatible chat-completions gateway
- local filesystem artifact storage
- DuckDB runtime store with stage insertion, claiming, heartbeats, completion, retryable/terminal failures, and expired-lease reclaim
- async stage runner with handler registry, bounded concurrency, heartbeats, retryable/terminal failure handling, and structured event emission
- static baseline planner that creates Actor-Theater conversation and Grader evaluation stages from config
- prompt-only Actor-Theater and Grader handlers
- child-stage artifact hydration from parent artifacts for the Grader path
- mocked end-to-end vertical slice through planner, runtime store, runner, handlers, fake gateway, and artifacts
- skipped-by-default Ollama integration test for the prompt-only vertical slice
- environment-variable expansion for local endpoint configuration
- JSONL event sink

Not yet implemented:

- successful Ollama-backed smoke run in this development environment
- DuckDB analysis exports
- graceful shutdown orchestration
- container profiles

## Design Bias

Keep code paths narrow and interfaces plural. The first executable run can be tiny, but it should use the same nouns as the eventual larger system: runs, samples, stages, attempts, leases, artifacts, endpoints, model pools, capability profiles, roles, agents, and genomes.
