# Can The Theater Model Work?

Research started: 2026-04-29

Source details: `research/sources/source-index.md`

## Bottom Line

The theater architecture is worth preserving, but "persona" should not mean only "a prompt prefix." The safer framing is:

> An Actor has a role-control genome. Text persona is the first gene, not the whole organism.

That genome can include persona text, system-template choices, structured output schemas, sampling parameters, thinking/tool controls, role-specific model choice, retrieval/tool access, LoRA/adapters, and eventually activation/representation steering. The existing design already points in this direction through topology-as-data, sampling-as-random-variables, and the builder pattern. The main change is to make the builder's control surface explicit enough that prompt engineering remains the universal baseline without becoming a conceptual trap.

My recommendation: start prompt-first, but not prompt-only. Evolve strings in the minimal vertical slice because they are cheap, inspectable, portable across models, and compatible with Ollama. At the same time, store the design as if every model call is produced from a structured genome. That lets later experiments swap in adapters or activation steering without rewriting the evolutionary loop.

## What The Current Design Gets Right

The design documents have several strong choices:

- The theater metaphor maps cleanly to experimental roles: Actor, Theater, Screenwriter, Director, Grader, Producer, Critic.
- Topology-as-data keeps the role hierarchy empirical rather than sacred.
- Treating model choice, temperature, topology, parsing, retry policy, and persona text as sampled variables is exactly the right move.
- The builder pattern is the right abstraction boundary. It can translate a single role-control genome into Ollama, LiteLLM, OpenAI-compatible endpoints, or future local inference engines.
- Pairwise/tournament ranking is a better long-term candidate than raw 1-10 scoring, because LLM judges often miscalibrate absolute scores.

The largest missing idea is vocabulary: call the optimized unit a control genome, not just a persona. Then the theater metaphor can stay without forcing every intervention through English prose.

## Evidence Map

### 1. String prompt optimization is not obsolete

Prompt/pipeline optimization is an active modern research direction, not just folklore. DSPy frames LM programs as optimizable text transformation graphs with parameterized modules. TextGrad treats text feedback as an optimization signal for variables in compound AI systems. OPRO shows LLMs can themselves act as black-box optimizers over natural-language task descriptions.

Implication for this project: the planned semantic-then-structural persona mutation is aligned with current work. The theater metaphor is not weird in a bad way; it is a domain-specific, inspectable language for compound-system optimization.

Risk: prompt strings are a lossy interface. They are easy to mutate and easy to inspect, but they cannot express every modern control mechanism efficiently.

### 2. PEFT/adapters are real alternatives, but not the first inner loop

PEFT methods, including LoRA, prefix tuning, prompt tuning, P-tuning, and IA3, adapt pretrained models by training a small number of extra parameters rather than all weights. Prefix tuning specifically learns continuous virtual-token prefixes while keeping base weights frozen.

Implication: for local models, adapters are the most practical "beyond prompt" method once enough training/evaluation data exists. They are likely better as a slower outer-loop artifact than as a per-sample mutation:

- Generate and evaluate many prompt-level Actor variants.
- Mine high-quality conversations, failure modes, and judge feedback.
- Periodically train role-specific adapters from those traces.
- Add adapter identity/version/hash to the genome and compare adapters against prompt-only controls.

Ollama can import Safetensors/GGUF adapters through Modelfiles, but its current adapter docs explicitly list Gemma 1 and Gemma 2. For Gemma 4 adapters, assume Transformers/PEFT, MLX, Unsloth, or llama.cpp may be needed until Ollama's adapter support is confirmed for Gemma 4.

### 3. Activation and representation steering are promising but backend-specific

Activation Addition and Representation Engineering show that inference-time activation manipulation can steer model behavior without fine-tuning. Anthropic's Golden Gate Claude demonstration made the point vividly: they changed a feature's activation directly, not by adding a system prompt or doing traditional fine-tuning.

Gemma is unusually relevant here because Google released Gemma Scope for Gemma 2 and Gemma Scope 2 for Gemma 3, including sparse autoencoders and interactive feature exploration/steering. That makes Gemma a good research family for asking whether local models can be shaped below the prompt layer.

Implication: activation steering should be a separate experimental track, not a dependency of the first system. It requires model-internal access and a hookable inference stack. Ollama is excellent for serving/multiplexing models, but it is not the right abstraction for per-layer interventions. Use Transformers/SAELens/pyvene/Neuronpedia-style tooling for this track and store steering vectors/features as genome fields when mature.

### 4. API-level controls are already broader than text

Gemma 4 and Ollama have moved beyond a minimal text-only interface. Gemma 4 advertises native system instructions, function calling, structured JSON output, thinking controls, multimodal input, and long context. Ollama's OpenAI-compatible API supports JSON mode, tools, reasoning/thinking controls, logprobs, logit bias, image content, seeds, temperature, top_p, and more.

Implication: even if the Actor's "persona" remains textual, the evolved object should also include:

- `system_style`: where and how role instructions are injected.
- `thinking_mode`: disabled, enabled, or role-dependent.
- `tool_policy`: no tools, declared tools only, retrieval tools, evaluator-only tools.
- `output_contract`: JSON schema or response format.
- `sampling`: temperature, top_p, top_k where supported, seed policy, max tokens.
- `context_policy`: summarization, transcript inclusion, scratchpad inclusion, memory window.
- `model_pool`: which models each role may use.

This makes the design more modern without abandoning prompts.

### 5. LLM-as-judge needs pairwise structure and held-out validation

G-Eval and related work support LLM-as-judge as useful, but also flag evaluator bias. Chatbot Arena uses pairwise human preference at scale, and PairS shows pairwise-preference search can align LLM evaluators better than direct scoring.

Implication: the existing rating-collapse concern is real. Keep absolute scoring for early debugging, but plan to rank serious candidates by pairwise/tournament methods with repeated judgments, randomized order, model-diverse judges, and held-out human-authored scenarios.

## Proposed Research Program

### Phase 0: Prompt-first vertical slice

Use Gemma 4 through Ollama. Keep roles minimal: Screenwriter scenario, Actor, Theater, Grader. Evolve:

- Actor persona text.
- Sampling parameters.
- Thinking mode.
- Role/model assignment.
- Structured-output contract for the Grader.

Do not attempt adapters or activation steering yet. The purpose is to prove the loop produces improving, inspectable behavior.

### Phase 1: Broaden the genome

Add non-prompt controls to the same sampling framework:

- Tool declarations and tool permissions.
- Output schema variants.
- Context-window policies.
- Theater model sampling.
- Grader/Director model sampling.
- Prompt template/system-message variants.

This will test whether the optimizer discovers useful combinations that prompt-only mutation would miss.

### Phase 2: Adapter track

Train role-specific adapters from accumulated traces. Compare:

- Prompt-only Actor.
- Prompt plus adapter Actor.
- Adapter-only or minimal-prompt Actor.
- Role-specific adapters for Grader/Director.

Evaluate on held-out scenarios and across multiple Theater models. Store adapter paths and hashes as immutable artifacts.

### Phase 3: Activation steering track

Run controlled experiments on hookable Gemma-family models:

- Use Gemma Scope/Gemma Scope 2 features where available.
- Compare steering-vector interventions against equivalent text personas.
- Measure effect size, generalization, stability, and failure modes.

Only fold this into the main system if it produces robust gains per unit engineering complexity.

## Crucial Experiments

1. Prompt-only vs structured genome:
   - Same scenarios, same models, same compute budget.
   - Arm A evolves only persona text.
   - Arm B evolves persona text plus sampling, thinking mode, output contracts, and role model choice.
   - Success means held-out Actor performance improves, not just judge score.

2. Single Theater vs multi-Theater:
   - Evolve against one Theater model.
   - Evolve against a sampled pool.
   - Test both against held-out Theater models.
   - This answers whether generality pressure helps or just weakens the solution.

3. Absolute rating vs pairwise ranking:
   - Compare stability of 1-10 scores, depth rating, and pairwise tournament ranking.
   - Track judge disagreement and correlation with human spot checks.

4. Prompt vs adapter:
   - Train adapters from top-performing traces.
   - Compare quality, compute cost, portability, and overfitting.

5. Prompt vs activation steering:
   - Start with narrow, measurable behavioral axes.
   - Avoid treating a striking demo as proof of broad controllability.

## Design Guidance

- Keep the theater metaphor.
- Rename the optimized unit from "persona" to "role-control genome" internally.
- Keep textual personas as the first-class, human-readable layer.
- Make every non-text control a typed field, even if most fields are empty in the first implementation.
- Record unsupported fields rather than silently losing them. The builder can drop unsupported fields at request time, but the sample record should preserve what was intended.
- Prefer experimental comparisons over architectural conviction. This project is strongest if it can say which parts of the metaphor worked and which control surfaces actually mattered.

## Practical Verdict

String-based prompting is ubiquitous enough to make it the right initial substrate. It may even approximate many structural changes inefficiently, and inefficiency is acceptable for this project. But optimizing only strings would likely miss important modern controls: tool use, thinking mode, schemas, role-specific model routing, adapters, and possibly activation steering.

The best path is not to abandon prompt evolution. It is to make prompt evolution one gene in a broader, experimentally testable control genome.
