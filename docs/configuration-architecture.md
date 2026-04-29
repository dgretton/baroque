# Configuration Architecture

Updated: 2026-04-29

## Core Principle

Prefer plural data structures at the durable boundaries of the system. Singular convenience should be a projection layered on top, not the stored shape.

In practice:

- Store `agents`, not `agent`.
- Store `roles`, not a single role enum hardcoded into control flow.
- Store `model_pools`, not one model string.
- Store `runtime_endpoints`, not one base URL.
- Store `capability_profiles`, not one global execution mode.
- Store `control_requests` and `effective_controls`, not just the final prompt.
- Store `topologies`, `scenarios`, `rankers`, `judges`, and `mutation_operators` as named config objects.

Defaults are still useful, but they should fill in missing plural settings rather than erase the plural shape.

## Why This Matters

The research question is not only "what is the best Actor?" It is "what kinds of control surfaces produce better agents under what constraints?"

That means a sample should be able to answer:

- Which Actor instance ran?
- Which role definition did it use?
- Which model pool was it allowed to draw from?
- Which model did it actually draw?
- Which capability profile governed the request?
- Which controls were requested?
- Which controls were compiled into the provider request?
- Which controls were dropped because the runtime did not support them?
- Which topology and ranking method evaluated it?

If the schema starts singular, these questions become retrofits. If the schema starts plural, a "one Actor, one Theater, one Ollama model" run is just the smallest case.

## Prioritized Build

The first implementation should be narrow in behavior but broad enough in shape.

### Phase 0: Prompt-Only Ollama Baseline

Implement only the controls that an ordinary local-model user could use with a default Ollama download:

- model selection from installed Ollama model names
- persona text
- system/user message construction
- few-shot examples as plain messages
- transcript/context inclusion policy
- plain text output-format instructions
- basic sampling knobs only if exposed by the normal API path

Exclude for this lane:

- tools
- retrieval
- custom Modelfiles
- adapters
- fine-tuning
- activation steering
- hidden external validators
- provider-specific templates beyond normal chat messages

This profile is the clean "what can prompting alone do?" baseline.

### Phase 1: Structured Ollama Controls

Add controls still available through ordinary Ollama/OpenAI-compatible calls:

- JSON mode or response-format controls
- reasoning/thinking controls where supported
- seeds and logprobs where supported
- image inputs for multimodal scenarios
- richer per-role sampling policy

### Phase 2: Runtime/Model Multiplexing

Add support for multiple local or remote endpoints without changing the experiment model:

- multiple Ollama hosts
- LiteLLM proxy
- exo/OpenAI-compatible cluster endpoint
- role-specific model pools
- endpoint health checks and routing metadata

### Phase 3: Advanced Shaping Tracks

Add experimental control surfaces as optional profiles:

- custom Modelfiles and prompt templates
- tool declarations and tool policies
- adapters/LoRA
- continuous prompts/prefix tuning
- activation or representation steering

These should never be required for prompt-only experiments.

## Layering Rules

Use layered defaults, with the more specific layer overriding the broader one:

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

Every completed stage should persist the resolved snapshot. Long runs may change defaults later, but old samples must remain interpretable.

## Capability Profiles

A capability profile defines what the runtime is allowed to honor. It is both an implementation switch and an experimental condition.

Example profiles:

```yaml
capability_profiles:
  prompt_only_ollama:
    allowed_controls:
      - model_choice
      - persona_text
      - message_examples
      - transcript_policy
      - plain_output_instructions
    denied_controls:
      - tools
      - retrieval
      - custom_modelfile
      - adapter
      - activation_steering
    provider_requirements:
      provider: ollama_openai

  ollama_prompt_params:
    extends: prompt_only_ollama
    allowed_controls:
      - sampling
      - seed

  ollama_structured:
    extends: ollama_prompt_params
    allowed_controls:
      - response_format
      - json_mode
      - thinking_mode

  adapter_runtime:
    extends: ollama_structured
    allowed_controls:
      - adapter

  hooked_runtime:
    extends: ollama_structured
    allowed_controls:
      - activation_steering
      - representation_features
```

The compiler should keep two records:

- `requested_controls`: what the genome wanted.
- `effective_controls`: what the capability profile and provider actually used.

This makes prompt-only and advanced runs comparable without hiding the difference.

## Plural Configuration Shape

Sketch:

```yaml
runs:
  baseline_prompt_only:
    capability_profile: prompt_only_ollama
    topology: actor_theater_grader
    active_agent_sets: [baseline_agents]
    active_scenario_sets: [starter_scenarios]
    active_rankers: [absolute_rating_debug]

runtime_endpoints:
  local_ollama:
    provider: ollama_openai
    base_url: http://localhost:11434/v1
    api_key: ollama

  linux_ollama:
    provider: ollama_openai
    base_url: http://linux-box.local:11434/v1
    api_key: ollama

models:
  gemma4_e2b:
    endpoint_pool: [local_ollama, linux_ollama]
    model: gemma4:e2b
    capability_tags: [chat, text, image, audio, thinking]

  gemma4_e4b:
    endpoint_pool: [local_ollama, linux_ollama]
    model: gemma4:e4b
    capability_tags: [chat, text, image, audio, thinking]

model_pools:
  small_local_gemma:
    models: [gemma4_e2b, gemma4_e4b]

  workstation_gemma:
    models: [gemma4_e2b, gemma4_e4b, gemma4_26b, gemma4_31b]

roles:
  actor:
    default_model_pool: small_local_gemma
    default_controls:
      transcript_policy: actor_running_window

  theater:
    default_model_pool: workstation_gemma

  grader:
    default_model_pool: small_local_gemma
    default_controls:
      plain_output_instructions: grader_json_text_contract

agents:
  actor_a:
    role: actor
    genomes: [actor_a_seed]

  grader_a:
    role: grader
    genomes: [grader_a_seed]

agent_sets:
  baseline_agents:
    agents: [actor_a, grader_a]

genomes:
  actor_a_seed:
    control_requests:
      persona_text:
        value_ref: personas/actor_a_seed.md
      message_examples:
        examples: []
      sampling:
        temperature: 0.8
```

This is intentionally more plural than the first implementation needs. The first runner can validate only the subset it understands, while the schema remains ready for multiple agents, model pools, endpoints, and genomes.

## Dynamic Changes During A Run

Do not mutate past config records. If something changes mid-run, record it as a new version:

- `run_config_version`
- `capability_profile_version`
- `agent_genome_version`
- `model_pool_version`
- `topology_version`

Samples and stages point to the versions they used. This supports later experiments where model pools, rankers, or mutation operators change during a long run.

## Implementation Bias

Start with:

1. `prompt_only_ollama`
2. one topology
3. a small model registry
4. one Actor population
5. one Theater pool
6. one Grader
7. one debug ranker

But keep the config shape plural from day one. That keeps the first vertical slice humane without painting the project into a single-agent, single-model, single-runtime corner.

