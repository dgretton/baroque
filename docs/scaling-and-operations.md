# Scaling And Operations

Updated: 2026-04-29

Source details: `research/sources/source-index.md`

## Reflection

This project will spend most of its wall-clock time waiting for LLM inference. Performance matters, but the main system is not a high-performance numerical kernel; it is a long-running orchestration, persistence, and analysis system wrapped around slow and failure-prone model calls.

That changes the engineering center of gravity:

- Native async is important because thousands of in-flight or queued model calls should not require thousands of threads.
- Every meaningful computation stage must be durable, idempotent, and restartable.
- Graceful shutdown and crash recovery matter from the first vertical slice, because week-long runs will eventually be interrupted.
- Logging and telemetry should be structured enough to reconstruct what happened without reading ad hoc console output.
- The inference backend should be swappable: single local Ollama, exo, vLLM, remote APIs, rented GPU nodes, or larger cloud GPU fleets.
- Analysis should remain friendly and local-first: DuckDB, Parquet, Polars/Python notebooks, and reproducible snapshots.

The goal is research-grade robustness, not consumer-product uptime. We do not need staged rollout, global SRE, or five nines. We do need to avoid a design that loses a week of progress because one process died.

## Architecture Shape

Use three planes:

```text
Control plane
  work queue, leases, run state, retries, config versions

Data plane
  content-addressed artifacts, conversations, evaluations, logs, Parquet exports

Analysis plane
  DuckDB/Polars/Jupyter/local scripts over Parquet and content-addressed records
```

This separation is important. DuckDB is excellent for local analytical work and Parquet export, but a multi-machine system should not depend on a single DuckDB file as a distributed transactional queue. For the first local runner, DuckDB can hold almost everything. As the system grows, move the work queue/control plane to Postgres, Temporal, Prefect, Ray, or a purpose-built queue while keeping artifacts and analysis data in Parquet/object storage.

## Stage Model

Every LLM call, mutation, ranking step, and parser retry should be represented as a stage.

Minimum stage fields:

- `stage_id`: UUID for this attempted execution.
- `content_hash`: deterministic hash of stage type, inputs, config snapshot, and parent hashes.
- `run_id`, `iteration_id`, `sample_id`.
- `stage_type`: actor turn, theater response, grader eval, mutation, ranking comparison, etc.
- `parents`: upstream content hashes.
- `requested_controls` and `effective_controls`.
- `provider`, `endpoint`, `model`, `role`, `agent_id`, `genome_id`.
- `status`: pending, leased, running, succeeded, failed_retryable, failed_terminal, cancelled.
- `attempt`, `lease_owner`, `leased_until`.
- `started_at`, `heartbeat_at`, `completed_at`, `duration_ms`.
- `artifact_refs`: pointers to conversation text, raw response, parsed response, logs, metrics.
- `error`: structured error class/message when applicable.

The rule: if a completed stage's content hash already exists, reuse it. If a process dies mid-stage, the stage was not completed and can be reclaimed after its lease expires.

## Async Runner

The first runner can be one Python process using `asyncio`, `httpx`, bounded semaphores, and a durable store.

Core loop:

1. Resolve missing stages for active samples.
2. Claim eligible stages with leases.
3. Execute claimed stages asynchronously, bounded by endpoint/model concurrency limits.
4. Persist successful results atomically.
5. Mark retryable failures with retry policy and backoff.
6. Make downstream stages eligible when parents complete.
7. Periodically heartbeat active leases.
8. On startup, reclaim expired leases and resume.

Important details:

- Use endpoint-level and model-level concurrency limits, even if the initial limit is `1`.
- Treat timeouts as normal events, not exceptional surprises.
- Preserve raw provider responses before parsing when possible.
- Store request snapshots, not just responses.
- Do not let in-memory queues be the only source of truth.
- Prefer append-only event records plus derived views.
- Either enqueue downstream stages at completion time or seed dependent stages up front and let parent-hash claim gating control eligibility. The current local implementation seeds the first Actor-Theater and Grader stages up front and uses parent-hash gating in DuckDB.

## Graceful Shutdown

Shutdown should be boring and explicit.

On `SIGINT` or `SIGTERM`:

1. Stop claiming new work.
2. Let short in-flight stages finish until a configured grace timeout.
3. Cancel or mark remaining in-flight stages as abandoned/retryable.
4. Flush artifact writes, logs, and telemetry.
5. Release or let expire leases.
6. Persist a `runner_stopped` event.

Do not attempt to checkpoint partial model generations unless a backend makes that cheap and reliable. The simpler rule is: a stage is complete only after its output and metadata are committed.

## Storage Strategy

See `storage-strategy.md` for the fuller storage design across config, runtime, logging, and research-result layers.

### Local First

For the first vertical slice:

- DuckDB for run metadata, stage records, beam state, and analysis tables.
- Filesystem content-addressed artifact store for raw text/JSON blobs.
- Periodic Parquet export or Parquet-first tables for analysis.
- JSONL structured logs as a plain fallback.

This keeps the system inspectable on a laptop and avoids deploying infrastructure too early.

### Multi-Host / Cloud Ready

When multiple runners or cloud workers enter:

- Postgres or Temporal for control-plane state and leases.
- S3-compatible object storage for artifacts and Parquet datasets.
- DuckDB for local or remote analysis by reading Parquet/S3.
- Optional Redis only for ephemeral coordination, not as the sole durable record.

S3-compatible storage is a good long-term target because it works with AWS S3, MinIO, Google Cloud, Cloudflare R2, and other providers, while DuckDB can read/write Parquet through S3-compatible APIs.

## Containerization Strategy

Use partial containerization with execution profiles, not mandatory full containerization.

Recommended modes:

```text
native-local
  host Python or venv
  host Ollama/exo
  local DuckDB and filesystem artifacts

compose-local
  containerized Baroque runner
  optional Postgres, MinIO, telemetry collector, dashboard
  host Ollama/exo/vLLM as an endpoint

cloud-worker
  containerized Baroque runner
  remote control plane and object storage
  cloud/local inference endpoint configured externally
```

The inference layer should not be containerized by default in early local development. On macOS and Apple Silicon especially, Ollama, exo, MLX, and Metal-backed inference are simpler and usually faster as host services. Treat them as external endpoints.

Good early container targets:

- Baroque runner
- tests and scripts
- local dashboard when one exists
- Postgres/MinIO/OpenTelemetry integration testing
- cloud worker image

Poor early mandatory container targets:

- local Ollama on Mac
- exo cluster
- Apple Metal/MLX inference
- anything whose GPU/driver path is still changing

Pros:

- reproducible Python/runtime dependencies
- easier CI and cloud-worker packaging
- cleaner local integration tests with Postgres/MinIO/telemetry
- easier migration to Modal, RunPod, Lambda, Kubernetes, or other GPU infrastructure
- fewer "works on this machine" dependency mismatches

Cons:

- local model serving may be slower or more awkward inside containers
- GPU containers add CUDA/driver/device-plugin complexity
- Docker on macOS adds virtualization overhead and host-network friction
- volume/data ownership can make long-run artifacts harder to reason about
- containerization does not solve the main bottleneck, which is inference latency

Design rule:

> Containerization is an execution profile, not a project assumption.

The code should only see configured endpoints, storage profiles, runner IDs, and capability profiles. Whether the runner is native, in Docker Compose, in Kubernetes, or inside a cloud worker should not change the research model.

## Logging And Telemetry

Use structured logs from day one.

Each log event should include:

- `run_id`
- `runner_id`
- `stage_id`
- `content_hash`
- `sample_id`
- `role`
- `agent_id`
- `model`
- `endpoint`
- `attempt`
- `event`
- `duration_ms` where relevant

Recommended layers:

- JSON logs through the standard library or `structlog`.
- OpenTelemetry spans for stage execution, provider calls, retries, parsing, ranking, and storage writes.
- Metrics for queue depth, active leases, completions/hour, tokens/sec if available, error rates, retry rates, timeout rates, and endpoint health.
- Lightweight dashboard later, not necessarily first.

OpenTelemetry traces are especially well matched to the DAG/stage model: parent spans can mirror parent content hashes.

## Platform Path

### 1. Single-Host Local

Use Ollama's OpenAI-compatible API as the first target. Keep `base_url`, model name, API key, and capability profile configurable.

This is the prompt-only baseline and the easiest place to validate:

- async runner
- content-addressed stage cache
- DuckDB/artifact storage
- restart recovery
- model registry
- basic observability

### 2. Local Multi-Resource

Use exo or similar local clustering when the goal is to pool machines into one local inference endpoint. exo is attractive because it exposes OpenAI-compatible, Claude-compatible, Responses-compatible, and Ollama-compatible APIs.

Architectural implication: treat exo as another endpoint. Do not make the orchestration layer know how exo shards a model.

### 3. Linux GPU Box

On the 100GB+ shared-memory Linux machine, continue supporting Ollama, but also consider vLLM for stronger throughput and OpenAI-compatible serving. vLLM is a good fit when the bottleneck becomes serving performance, batching, or production-ish OpenAI-compatible APIs.

This should still be a provider entry:

```yaml
runtime_endpoints:
  linux_vllm:
    provider: openai_compatible
    base_url: http://linux-box:8000/v1
```

### 4. Cloud Burst

For rented GPU capacity, prefer platforms where workers can run containers or simple Python functions and talk to the same control/data planes:

- Modal: strong developer experience for Python, autoscaling GPU functions, batch jobs, and code-defined infrastructure.
- RunPod: flexible GPU pods and serverless endpoints with bring-your-own-container workflows.
- Lambda Cloud: good for larger dedicated GPU instances or clusters, including Kubernetes/Slurm-style modes.
- Major clouds: useful when grants, institutional procurement, object storage, or managed Kubernetes matter more than raw convenience.

Use these for burst experiments or higher-level evaluator optimization, not as a required baseline.

### 5. Larger GPU Fleets

If the project reaches a genuinely large serving phase:

- Kubernetes with GPU device plugins is the standard substrate.
- Ray is attractive for Python-native distributed tasks and actors with explicit CPU/GPU resource requirements.
- vLLM can serve models behind OpenAI-compatible APIs.
- NVIDIA Dynamo is worth watching for large distributed inference, especially with vLLM/SGLang/TensorRT-LLM backends and KV-cache-aware routing.
- Temporal or Prefect can take over orchestration if the homegrown runner becomes the research bottleneck.

Do not start here. Keep the API boundaries clean enough that this remains a migration path rather than a rewrite.

## Orchestration Options

### Custom Async Runner

Best first choice.

Pros:

- Small conceptual footprint.
- Directly matches the research DAG.
- Easy to keep content-addressed stage semantics.
- No external service required for local runs.

Cons:

- You must implement leases, retries, observability, and recovery carefully.
- Distributed execution will eventually need a real control plane.

### Prefect

Good if you want Python-native workflow tracking, retries, task states, and a UI without adopting a deep distributed-systems model. Prefect's hybrid model can keep execution in your infrastructure while tracking state/logs.

Risk: the project DAG is itself part of the research artifact, so Prefect may either help with visibility or become a layer to work around.

### Temporal

Good if durable execution becomes central. Temporal is designed for workflows that survive failures and resume from durable state.

Risk: more operational and conceptual weight than needed for the first slice. It is a serious tool, and adopting it early could slow the research loop.

### Ray

Good when you have many Python workers or GPU resources and want explicit resource scheduling. Ray tasks and actors can specify CPU/GPU requirements.

Risk: Ray solves distributed execution, not the experiment schema. It should be an execution backend, not the core representation of samples/stages.

## Recommended Near-Term Decision

Build a custom async runner now, but design the storage and endpoint interfaces so it can later hand off execution to Ray, Prefect, Temporal, Kubernetes jobs, Modal, or RunPod workers.

The runner should have these interfaces:

```text
Planner
  resolves missing stages from topology + sample configs

LeaseStore
  claims, heartbeats, releases, and reclaims work

ArtifactStore
  writes and reads content-addressed blobs

InferenceGateway
  sends provider requests to local/remote endpoints

EventSink
  logs structured events and telemetry

AnalysisExport
  writes Parquet snapshots/views
```

If those boundaries are clean, the first implementation can stay small while the project remains structurally ready for weeks-long runs and larger compute budgets.

## Non-Negotiables For The First Code

- All model calls are async.
- All stages have deterministic content hashes.
- All completed stages are committed before downstream work depends on them.
- Work claims use leases, even in the single-process runner.
- Shutdown is signal-aware and leaves restartable state.
- Logs are structured.
- Provider requests and raw responses are persisted.
- Config versions are persisted with stage records.
- Endpoint/model concurrency limits are configuration, not constants.
- Analysis data can be exported or queried without running the orchestrator.

This is the smallest amount of operational seriousness that keeps the research system from becoming fragile.
