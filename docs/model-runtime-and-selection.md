# Model Runtime And Selection

Updated: 2026-04-29

Source details: `research/sources/source-index.md`

## Current Recommendation

Use Gemma 4 as the default local model family and make model selection fully configurable. Gemma 4 was announced on 2026-04-02 and is now the latest Gemma generation. Ollama exposes the main local sizes:

| ID | Ollama model | Approx package size | Context | Notes |
| --- | --- | ---: | ---: | --- |
| `gemma4_e2b` | `gemma4:e2b` | 7.2 GB | 128K | Edge model; text, image, audio in the model card. Start here on small machines. |
| `gemma4_e4b` | `gemma4:e4b` | 9.6 GB | 128K | Default `gemma4:latest` as of this check; good local baseline. |
| `gemma4_26b` | `gemma4:26b` | 18 GB | 256K | MoE model, 25.2B total / 3.8B active parameters. Likely attractive on the Linux box. |
| `gemma4_31b` | `gemma4:31b` | 20 GB | 256K | Dense workstation model; slower but strongest local candidate. |

Do not hardcode any of these into the orchestration logic. Treat them as registry entries that roles can sample from.

## Local Status

On this Mac, the Ollama client exists at `/opt/homebrew/bin/ollama` and reports client version `0.12.1`, but the Ollama server was not reachable from the sandbox during this check. That means the repo should include commands/configuration, but we should not assume which Gemma 4 sizes this Mac can run until the server is started and each model is probed.

Also note: the latest Ollama OpenAI compatibility docs mention Responses API support added in Ollama `0.13.3`. Upgrade Ollama before relying on the newest Responses API or reasoning-control behavior.

## Pull And Probe Order

Pull models in increasing size. Stop wherever latency or memory becomes impractical.

```bash
ollama pull gemma4:e2b
ollama pull gemma4:e4b
ollama pull gemma4:26b
ollama pull gemma4:31b
```

Quick smoke tests:

```bash
ollama run gemma4:e2b
ollama run gemma4:e4b
ollama run gemma4:26b
ollama run gemma4:31b
```

For API use, prefer the OpenAI-compatible endpoint:

```text
http://localhost:11434/v1
```

Use `api_key = "ollama"` for OpenAI-compatible clients; Ollama ignores the value but many clients require one.

## Configuration Shape

The orchestrator should load a model registry from a file or environment variable, then assign role pools from that registry. This keeps the MacBook, Framework/Linux box, and future clusters using the same code.

Example:

```yaml
runtime:
  gateway: ollama_openai
  base_url: ${BAROQUE_LLM_BASE_URL:-http://localhost:11434/v1}
  api_key: ${BAROQUE_LLM_API_KEY:-ollama}
  default_timeout_s: 600

defaults:
  temperature: 1.0
  top_p: 0.95
  provider_options:
    top_k: 64

models:
  gemma4_e2b:
    provider: ollama_openai
    model: gemma4:e2b
    context_window: 131072
    capabilities: [chat, text, image, audio, tools, thinking]
    local_size_gb: 7.2

  gemma4_e4b:
    provider: ollama_openai
    model: gemma4:e4b
    context_window: 131072
    capabilities: [chat, text, image, audio, tools, thinking]
    local_size_gb: 9.6

  gemma4_26b:
    provider: ollama_openai
    model: gemma4:26b
    context_window: 262144
    capabilities: [chat, text, image, tools, thinking]
    architecture: moe
    active_parameters_b: 3.8
    local_size_gb: 18

  gemma4_31b:
    provider: ollama_openai
    model: gemma4:31b
    context_window: 262144
    capabilities: [chat, text, image, tools, thinking]
    architecture: dense
    local_size_gb: 20

profiles:
  macbook_probe:
    models: [gemma4_e2b, gemma4_e4b]
    max_parallel_requests: 1

  linux_100gb:
    models: [gemma4_e2b, gemma4_e4b, gemma4_26b, gemma4_31b]
    max_parallel_requests: 1

roles:
  theater:
    pool: [gemma4_e2b, gemma4_e4b, gemma4_26b, gemma4_31b]
    selection: weighted_random

  actor:
    pool: [gemma4_e2b, gemma4_e4b]
    selection: sampled

  grader:
    pool: [gemma4_e4b, gemma4_26b, gemma4_31b]
    selection: sampled

  director:
    pool: [gemma4_e4b, gemma4_26b, gemma4_31b]
    selection: sampled
```

`max_parallel_requests` is deliberately conservative. The runtime should use async dispatch even when the active profile allows only one inference at a time, because the same orchestration code can later benefit from exo, LiteLLM routing, Ollama cloud, or multiple local endpoints.

## Multiplexing Options

Start simple:

1. Use Ollama directly on `localhost:11434`.
2. Let the runner's queue choose the model per role call.
3. Keep every model choice in the sample record.

When multiple providers or endpoints appear, put LiteLLM in front:

- LiteLLM provides one OpenAI-format interface across many providers.
- It supports Ollama calls, retry/fallback logic, routing/load balancing, and observability callbacks.
- The code should see a single `base_url` and `model` string, regardless of whether the backend is one Ollama server, multiple Ollama hosts, exo, a LiteLLM proxy, or a remote API.

Recommended abstraction:

```text
RoleGenome -> QueryBuilder -> ProviderRequest -> InferenceGateway
```

The `RoleGenome` may contain fields the current provider does not support. The `QueryBuilder` should compile the nearest valid request while preserving the full intended genome in storage.

## Model Management Notes

- `ollama pull <model>` downloads or updates a model.
- `ollama list` shows local models.
- `ollama show <model>` inspects a model.
- `ollama rm <model>` removes a model.
- `ollama cp <source> <alias>` creates aliases for tools that expect specific model names.
- A Modelfile can set `num_ctx`, `SYSTEM`, `TEMPLATE`, and `PARAMETER` values.
- A Modelfile can also reference adapters with `ADAPTER`, but confirm current Gemma 4 adapter support before relying on Ollama for adapter experiments.

## Design Constraint

No role should assume a single "best" model. The research system should be able to ask:

- Which Theater model was interrogated?
- Which model graded the result?
- Which model mutated the Actor?
- Which controls were requested?
- Which controls were actually supported by the provider?
- Did the winning Actor generalize to held-out Theater models?

That means model choice is part of the experiment, not a deployment detail.
