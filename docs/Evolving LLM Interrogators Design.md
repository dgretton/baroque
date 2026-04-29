# Evolving Multi-Agent Interrogators for Large Language Models — Design Document

## 1\. Project Summary

This project builds a framework for evolving populations of cooperating and supervising LLM agents that extract high-quality, relevant information from target language models. Agents are defined by personas — prompt-prefixed behavioral specifications — and are iteratively refined through structured multi-agent evaluation and evolutionary optimization.

The goal is not a single optimal interrogator. It is a system in which interrogators are bred, evaluated, compared, and improved through recursive feedback, and which remains adaptable as LLM capabilities change underneath it.

The system is designed for long-duration autonomous operation: weeks-long runs against local models (e.g., via Ollama on a Framework Desktop AI Max), with the ability to scale to a small cluster via tools like exo, and with enough observability and fault tolerance that it can run unattended.

## 2\. Agent Hierarchy and Roles

The system organizes several distinct agent types into a supervisory hierarchy. Each agent is defined primarily by its persona — a block of text that prefixes its interactions and shapes its behavior.

### 2.1 Core Roles

**Actor.** The primary target of optimization. The Actor plays the role of an interrogator: it conducts a multi-turn conversation with a target model (the Theater), attempting to extract information relevant to a scenario defined by the Screenwriter. The Actor's persona encodes its interrogation strategy, style, follow-up behavior, and domain framing.

**Theater.** The model being interrogated. The Theater is not subject to optimization. It receives the Actor's queries and responds. In the simplest configuration, the Theater is a single fixed model. In more advanced configurations, the Theater model may be sampled from a distribution per-sample or even per-turn, to force evolved personas toward generality rather than overfitting to one model's quirks.

**Grader.** Evaluates the *informational quality* of what the Actor extracts: relevance to the scenario, completeness, factual correctness, task progress. The Grader sees the scenario, the full Actor-Theater conversation, and produces a structured evaluation.

**Director.** Evaluates how well the Actor *performs the role itself*: interrogation strategy, adherence to persona, quality of follow-up questions, ability to probe deeper when answers are vague. Where the Grader judges the output, the Director judges the process.

**Screenwriter.** Defines the problem: objectives, constraints, domain, and the information-extraction scenario the Actor will pursue. Initially this is a human role (the PI provides scenarios). The system should keep this interface modular so that once good example scenarios exist, an LLM agent can take over scenario generation — and potentially become subject to optimization itself.

### 2.2 Why the Screenwriter Role Matters

The Screenwriter is easy to underestimate because it sits outside the optimization loop, but it has enormous leverage over the quality of evolved interrogators. A poorly specified scenario can make even an optimal Actor persona look mediocre — if the task is too easy, all Actors score high and there is no selection pressure; if the task is incoherent, scores become noisy and optimization drifts. When the Screenwriter eventually becomes an LLM agent, there is an additional risk: the system could evolve scenarios that are easy to ace rather than genuinely informative, collapsing the evaluation signal. Mitigation strategies include fixing the Screenwriter outside the optimization loop (current approach), using a held-out set of human-authored scenarios for validation, or adding an explicit adversarial constraint to the Screenwriter's objective.

### 2.3 Meta-Roles

**Producer.** Supervises the Grader, Director, Screenwriter (if an agent), and Actor. The Producer evaluates how well the evaluators themselves are performing — whether the Grader is catching real quality issues, whether the Director's strategic advice is coherent, whether the Screenwriter's scenarios are well-formed. The Producer also performs the semantic analysis that drives persona mutation (Section 8).

**Critic.** Evaluates the Producer, forming the highest layer of recursion in the system. The Critic asks whether the Producer's meta-evaluations are themselves sound.

### 2.4 Topology as Data

The hierarchy described above is a starting point, not a fixed architecture. Different supervisory structures may yield different optimization strength, stability, or interpretability. The topology should therefore be represented as data — a directed acyclic graph (DAG) where nodes are agent roles and edges represent "evaluates" or "supervises" relationships.

This means:

- The orchestration engine reads a topology definition (e.g., a dict or config file mapping roles to their evaluators and inputs) rather than hardcoding control flow.  
- Different topologies can be compared experimentally by simply swapping the definition.  
- Topology variations can be folded into the sampling framework (Section 3\) as another sampled random variable, letting the optimizer explore topologies alongside personas, temperatures, and model choices.  
- The topology must be a validated DAG — cycles would create infinite recursion. Validation at startup is straightforward: topological sort, reject if impossible.

Role-recursion depth (Critic → Producer → Director/Grader → Actor) is likely a sensitive parameter and should be varied experimentally.

## 3\. Sampling Framework

### 3.1 What Is a Sample

A sample is the complete record of one evaluation run: the Screenwriter's scenario, all Actor-Theater turns, all evaluations by the Grader and Director, the Producer's assessment, the Critic's assessment, all personas, all configurations, and all metadata (model choices, temperatures, timestamps, etc.).

### 3.2 Monte Carlo Exploration

The sampling approach treats the system's many degrees of freedom as random variables, explored via Monte Carlo–style draws rather than enumeration or grid search. Any property of the system can be made into a sampled variable:

- Persona text (drawn from a population or mutated from a parent)  
- Temperature settings  
- Model choices for evaluator roles (which need not be the same as the Theater)  
- Theater model (to force generality)  
- Topology variations  
- Parsing and retry parameters  
- Structured output format variations

This is powerful because it lets the optimizer discover unexpected beneficial combinations of parameters that would never be found by optimizing each dimension independently.

### 3.3 Hierarchical Probability

If a sample's configuration arises from a hierarchy of random choices c\_1, c\_2, ..., c\_k, each conditioned on the preceding ones, the probability of drawing a specific configuration is:

P(sample) \= prod\_{i=1}^{k} P(c\_i | c\_1, ..., c\_{i-1})

Parameters of variable-length processes (e.g., stochastic retry loops) fit naturally into this formulation because their downstream contribution to scoring remains well-defined.

### 3.4 Functional Sampling

To enforce sample independence, limit statefulness, and avoid evaluating anything that will not be scored, sampling can be framed functionally. In the default topology, sampling means evaluating the Critic's assessment of the Producer — since this is the terminal node, evaluation blocks while all lower-level activity is recursively generated as attempts to fetch inputs. In practice, a stack-based procedural machine that simulates this functional unfolding is more practical for debugging long-running processes, and the DAG-based orchestration (Section 12\) provides exactly this.

### 3.5 Sample Space Dimensionality

An open design question: which properties should be randomized within samples, and which should remain fixed? Too few randomized dimensions may cause the optimizer to miss rare high-performing parameter combinations. Too many may make optimization noisy — the signal from a good persona gets drowned out by variance in other dimensions. The right balance will depend on the specific roles and properties being optimized and must be determined empirically. A reasonable starting strategy is to fix the Theater model and topology, randomize persona text (via mutation from a parent) and temperature, and add dimensions incrementally as the system matures.

## 4\. Optimization Strategy

### 4.1 Modified Beam Search

The primary optimization strategy maintains a population ("beam") of samples at a constant size (e.g., 1000). Each iteration:

1. Each existing sample proposes at least one variant (mutated persona, different configuration).  
2. Additional variants may be generated stochastically, with higher-scoring samples producing more offspring.  
3. All candidates — new and existing — form a pool.  
4. The pool is ranked (Section 5).  
5. The lowest-ranking samples are dropped until the population returns to the beam size.

This procedure maintains diversity and avoids early convergence while allowing strong performers to propagate. It is not constrained by a fixed search depth or a strongly tree-structured process; it leverages the rich variability of the sample format. More sophisticated evolutionary strategies may later be substituted. This is a practical and interpretable starting point.

### 4.2 Compute Budget and Throughput

Each sample involves multiple LLM calls: Actor-Theater conversation (possibly multi-turn), Grader evaluation, Director evaluation, Producer assessment, Critic assessment, plus the mutation pipeline. At beam size 1000, each iteration involves thousands of LLM calls.

On the Framework Desktop AI Max running local models via Ollama, the throughput constraint is GPU memory bandwidth. Even two evaluations in parallel may not be practical. The system should dispatch all queries asynchronously as a general principle — if they all queue behind a single inference resource, that is fine. The key benefit of async is architectural: it cleanly separates "what needs to be computed" from "how many resources are available," and it means the system can transparently benefit from additional compute if it becomes available (see Section 11 on exo).

Expected iteration time will be long. Beam populations will evolve over days and weeks, not minutes. This is acceptable — the system is designed for long-duration autonomous operation, and the iteration timescale provides time for observation and course correction.

### 4.3 Per-Call Model Choice Sampling

Rather than implementing a separate "tiered evaluation" system where cheap models filter before expensive models evaluate, model choice per call should be treated as another sampled variable within the sampling framework. Some samples will use a cheap small model for their Grader; others will use a more capable one. If using a capable Grader consistently produces better optimization signal, that combination will be selected for. This avoids adding a second optimization mechanism outside the sampling framework and keeps the design unified.

## 5\. Ranking and Rating

### 5.1 Design Principle: Modularity

The system needs to produce a ranking of samples in a population so that the worst can be culled and the best can propagate. How that ranking is produced should be modular — the interface is "given a population of samples, produce a ranking," and the implementation can be swapped.

This modularity matters because rating collapse (Section 5.2) is a real problem with LLM-as-judge setups, and the best solution may not be obvious in advance. Multiple approaches should be implementable behind the same interface.

### 5.2 The Rating Collapse Problem

When LLMs are asked to rate quality on a numeric scale, they tend to saturate at the high end. Models frequently choose 9 or 10 when judging quality. When many evaluations cluster at the maximum value, ranking resolution collapses — the landscape becomes stepped and low-fidelity, and the optimizer cannot distinguish between good and excellent.

### 5.3 Approach: Iterative Depth Rating

One approach to recovering resolution is to ask for conditional ratings. If too many evaluators give a rating of 10, those providing a 10 are asked to rate the same item on a new scale: "Among items you would rate 10, rate this from 1-10 and explain." If it again gives 10, the question repeats. This recursion continues until the model produces a non-10 or a maximum depth is reached.

Ratings map to the unit interval via r\_i \= (R\_i \- 1\) / 10, and the composite rating is the power series:

r\_sum \= sum\_{i=0}^{inf} r\_i \* 10^{-i}

For example, ratings \[10, 10, 10, 4\] produce r\_sum \= 0.9 \+ 0.09 \+ 0.009 \+ 0.0003 \= 0.9993.

This is mathematically clean but needs empirical validation: can models actually discriminate meaningfully at depth 3+? The information content of each additional depth level should be checked early — do deeper digits actually correlate with downstream optimization signal, or do they become noise after depth 2?

### 5.4 Approach: Pairwise Comparison and Tournament Ranking

An alternative that sidesteps the saturation problem entirely: instead of asking for absolute scores, ask "which of these two is better?" Pairwise judgments avoid the problem of scale calibration because the model never needs to assign a number — it only needs to make a relative comparison, which LLMs do more reliably.

Pairwise outcomes can be accumulated into a ranking via:

- **ELO-style ratings**: Each comparison updates both candidates' ratings based on the outcome and the expected outcome given current ratings.  
- **Bradley-Terry model**: Fit a maximum-likelihood model to the pairwise outcome data, producing a strength parameter for each candidate.  
- **Confidence intervals**: Both methods can produce uncertainty estimates, which are useful for deciding which pairs to compare next.

The naive approach requires O(n^2) comparisons for n candidates (full round-robin), which is impractical for large populations. A **Swiss tournament** structure provides O(n log n) comparisons while still producing a reliable ranking: in each round, candidates are paired with others of similar current standing, and after O(log n) rounds every candidate has been compared enough times to place it accurately. This maps well to the sampling framework — a "comparison sample" is a sample where two candidates are presented to a judge agent.

### 5.5 Hybrid and Future Approaches

A hybrid approach is possible: use absolute ratings for coarse filtering (drop anything below a threshold), then use pairwise comparisons for fine-grained ranking among the survivors. Or use pairwise comparisons as the primary ranking mechanism and only fall back to absolute ratings when the population is too large for even Swiss-tournament efficiency.

The key decision: the ranking module's interface should accept a population and return an ordered ranking (or a partial order with confidence). Everything else — the mechanism, the number of judge calls, the rating format — is an implementation detail behind that interface.

## 6\. Structured Outputs and Retry Logic

Many stages of the evaluation pipeline require machine-parseable outputs. Evaluators must produce structured data (e.g., JSON with a numeric rating field and a justification) so that downstream logic can parse, aggregate, and compare.

Structured response formats make it possible to:

- Detect malformed outputs automatically.  
- Request a retry until a parseable form is produced.  
- Enforce consistency across evaluators.

Example structured output request:

Please respond with a JSON object:

{

  "rating": \<integer 1-10\>,

  "justification": "\<one-sentence explanation\>"

}

If the format is incorrect, I will ask you to retry.

Stochastic retry chains (where the number of retries varies per sample) fit naturally into the hierarchical probability formulation — each retry is a conditioned random variable, and the downstream contribution to scoring remains well-defined. Retry chains can themselves be rated by meta-level agents without breaking the scoring system.

## 7\. Model Flexibility and the Builder Pattern

### 7.1 The Problem

LLM APIs differ widely. Some expose extensive configuration: temperature, top-p, reasoning effort, tools, multiple instruction channels, multimodal inputs. Others — particularly local models served through Ollama — provide only a minimal interface: a model name, a list of messages, and perhaps temperature.

### 7.2 Builder Pattern

A builder pattern provides a consistent interface that:

- Starts from a standard baseline configuration for a given provider.  
- Applies persona-dependent modifications.  
- Applies evaluator- or Director-recommended changes.  
- Compiles a request appropriate for the target model.  
- Silently strips or ignores unsupported fields.  
- Encodes supported fields correctly for the target API.  
- Injects persona prefixes into whichever input slot the model uses (system prompt, message prefix, etc.).

Example (fluent style):

query \= (QueryBuilder("ollama/llama3")

    .with\_persona(actor.persona\_str)

    .with\_scenario(scenario\_prompt)

    .temperature(0.7)

).build()

response \= query.send(message)

When the target is a minimal API (Ollama), fields like `tools` or `reasoning_effort` are silently dropped. When the target is a richer API (Anthropic, OpenAI), they are included. The builder abstracts this so the rest of the system does not need to know which fields a given model supports.

### 7.3 litellm as the Abstraction Layer

The builder can use litellm as its backend for API normalization. litellm already handles routing across OpenAI, Anthropic, Ollama, and many other providers with a unified interface. This avoids re-implementing provider-specific request formatting. The builder pattern sits on top of litellm, adding persona injection, configuration staging, and field filtering specific to this project.

### 7.4 Future Capabilities

Prompt-prefix persona engineering is the primary way agents are shaped, since it is universally supported. However, modern models increasingly support capabilities beyond prompt prefixes: internal conditioning states, tool use declarations, domain-biasing fields. Tool use in particular has transformed LLM capabilities rapidly.

The builder pattern naturally accommodates these extensions: modifications to local models are staged before querying, advanced remote models receive structurally richer requests, and simpler models fall back to text-only personas. The system should be built so it can support tool use, even if it is not used in the initial implementation, to avoid the outputs becoming irrelevant to what modern models actually produce.

For local models specifically, techniques like activation pinning and weight subsetting may eventually outperform filling context windows with persona text. A few local inference engines now expose richer configuration than the minimal defaults. The builder pattern accommodates these as additional staging steps before querying.

## 8\. Persona Mutation: Semantic-Then-Structural Edits

### 8.1 The Problem

The mutation operator — how new persona variants are generated from existing ones — is arguably the most important part of the evolutionary loop. If mutations are too coarse (replacing whole paragraphs), fine-grained optimization is impossible. If mutations are too fine (single-word edits), coherent behavioral changes are unlikely. The mutation needs to be guided by the evaluator's understanding of what went wrong.

### 8.2 Three-Phase Mutation Within a Single Context

A key design decision: the agent that judges the Actor's performance should also be the one that proposes and implements the edits, within the same context window. This preserves the full evaluation context when making edits, which produces dramatically better edits compared to passing off edit instructions to a separate process without context.

The mutation pipeline has three phases, executed as three sequential calls within the same conversation (preserving context via message history):

**Phase 1: Judgment.** The supervising agent (Director or Producer) evaluates the Actor's performance, with the full Actor-Theater conversation, the scenario, and the current persona all in context. It produces a structured assessment of what worked and what did not.

**Phase 2: Edit Objective.** In the same context, the agent articulates what needs to change — a goal-oriented, semantic description of the desired modification. For example: "Make the actor press harder on vague answers instead of accepting them and moving on" or "Add a step where the actor summarizes what it has learned so far before asking follow-up questions."

**Phase 3: Concrete Edits.** Still in the same context, the agent produces the actual text edits to the persona definition that implement the objective from Phase 2\. These are block-level search-and-replace operations on the persona text.

### 8.3 Why Three Calls Instead of One

Using three separate calls (rather than one call with three output sections) provides:

- **Restartability.** If the process dies after Phase 2, the judgment and objective are persisted. Phase 3 can resume by replaying them into a new context.  
- **Documentation.** Each phase's output is a separately loggable, inspectable artifact. Over the course of an optimization run, the collected edit objectives are themselves valuable data — they reveal what kinds of changes the system proposes over time, which can be analyzed for a paper.  
- **Debuggability.** If an edit makes things worse, you can look at the Phase 2 objective to see whether the diagnosis was wrong (Phase 1 was bad) or the implementation was wrong (Phase 3 didn't achieve the Phase 2 objective).

### 8.4 Baseline Edit Mechanism

The concrete edits from Phase 3 are block-level search-and-replace operations on the persona text. This is simple, transparent, and compatible with all models. The method by which changes are implemented is flexible — search-and-replace on text blocks is the baseline, but any technique that modifies structured text would work, including structured diffs or targeted insertions/deletions. This flexibility extends naturally to more sophisticated transformation methods as the system matures.

## 9\. Inspecting Multi-Agent Interactions

To make the complex interactions between agents inspectable, samples are structured as email-like threads for each agent. Whenever agents communicate or respond to each other, the resulting "conversation" appears as a message in that agent's inbox. Multi-agent interactions that might otherwise be a confusing tangle become familiar, readable, and debuggable — long and branching threads can be read the same way one reads an email chain.

This is especially important for a system designed to run for weeks. When checking in on a long-running optimization, the ability to pull up a specific sample and read its interactions as a set of threaded conversations is critical for understanding what is happening and why.

## 10\. Execution Environment and Hardware

### 10.1 Primary Compute

The primary compute resource is a Framework Desktop AI Max running Ubuntu 24 LTS. This machine will run local model inference via Ollama for weeks-long optimization campaigns. The AI Max has substantial memory but is memory-bandwidth limited, meaning that even two inferences in parallel may not provide meaningful speedup.

The system should dispatch all work asynchronously regardless. If all queries queue behind a single inference resource, that is acceptable — the async architecture means the system transparently benefits from additional compute when it becomes available.

### 10.2 Development Machines

Development will occur across multiple machines: the Framework Desktop, a small MacBook (Ubuntu 24 LTS), and a 2020 MacBook M1 (macOS). All are Unix environments. The codebase should have no platform-specific dependencies.

### 10.3 Scaling with exo

exo pools heterogeneous devices into a single inference endpoint with automatic model partitioning. If scale-up is needed (e.g., the project receives hardware funding, potentially \~$100k), exo can combine multiple machines into a cluster behind a single OpenAI-compatible API endpoint.

The architectural implication: the system should talk to a single inference endpoint URL and not care what is behind it. Since Ollama already exposes an OpenAI-compatible API, and exo does too, switching from single-machine Ollama to a cluster is just changing a base URL. No special design work is needed now — just avoid hardcoding anything Ollama-specific.

### 10.4 Storage

The Framework Desktop has 4TB of storage. The system should make liberal use of this to store everything that might be useful for later analysis and paper documentation. All sample data, evaluation results, persona snapshots, and mutation logs should be retained indefinitely. Storage is append-only — computed results are never mutated or deleted.

## 11\. Data Storage and Content-Addressable Memoization

### 11.1 Design Goals

The storage layer must support:

- Persisting every completed evaluation step immediately (for fault tolerance).  
- Memoizing completed computations so they can be reused when inputs are unchanged.  
- Making the causal chain of each sample's computation inspectable: where it starts, how it progresses, where it ends.  
- Efficient analytical queries months later (for paper-writing).  
- Avoiding the cognitive load of sprawling JSON files, ad-hoc SQLite databases, and mystery CSVs.

### 11.2 DuckDB as the Primary Store

DuckDB is an embeddable analytical database — like SQLite but designed for analytical queries over structured data. It requires no server, stores data in a single file, has native Python bindings, and is fast for the kinds of aggregate/filter/join queries needed for analyzing optimization runs.

DuckDB reads and writes Parquet natively. Parquet files are compressed, columnar, and self-describing — any tool in the Python/R ecosystem can read them directly. When it comes time to analyze runs for a paper, you write SQL or use pandas/polars against the Parquet files. No JSON parsing, no schema guessing.

### 11.3 Content-Addressable Hashing for Memoization

Each computation step's identity is a hash of its inputs: the configuration, the outputs of its input stages, and any other parameters. If that hash already exists in the store, the computation has already been done and the result can be reused.

This gives a unified structure for both memoization and progress tracking:

- **Memoization**: Before running a computation, hash its inputs. If the hash exists, skip it and use the cached result.  
- **Progress tracking**: A sample's progress is the set of content hashes that have been computed for it. An incomplete sample is one where some expected hashes are missing.

The UUID scheme should be richer than per-sample. Each computation step gets its own content-addressed ID, and a sample is defined by the set of step IDs that compose it. This facilitates reuse: if a Theater interaction is unchanged but the Grader persona has been mutated, the Theater interaction step is already cached and only the Grader step needs to be recomputed.

### 11.4 Schema Design

The core data structure is a table of computation events. Each row represents a completed stage:

| Column | Description |
| :---- | :---- |
| `content_hash` | Primary key. Hash of (stage\_type, input\_hashes, configuration). |
| `sample_id` | UUID of the sample this step belongs to. |
| `stage_type` | e.g., "actor\_theater", "grader\_eval", "director\_eval", "producer\_eval", "critic\_eval", "mutation" |
| `parent_hashes` | List of content hashes of input stages (the causal parents). |
| `config_snapshot` | The configuration used for this stage (model, temperature, persona hash, etc.). |
| `persona_hash` | Reference to the persona text used (stored separately to avoid duplication). |
| `output_artifact` | The output of this computation (the conversation, the rating, the edit, etc.). May reference a separate blob store for large artifacts. |
| `score` | Numeric score, if this is an evaluation stage. |
| `started_at` | Timestamp when computation began. |
| `completed_at` | Timestamp when computation completed. |
| `duration_ms` | Wall-clock duration. |
| `iteration` | Which beam iteration this belongs to. |

Querying "show me the full causal chain for this sample" is a recursive query on `parent_hashes`. Querying "show me all samples where the Actor persona contained X and the Grader score was above Y" is a join and filter.

Separate tables or Parquet partitions for:

- **Personas**: keyed by hash of the persona text, storing the full text. Avoids duplication when many samples share a persona.  
- **Beam state**: per-iteration snapshots of the population (which sample IDs are alive, their scores, their lineage).  
- **Run metadata**: top-level configuration, topology definition, start time, current iteration.

### 11.5 Phasing in Records

Each computation event record should make its causal position clear:

- `parent_hashes` links to the upstream stages this stage depended on.  
- `stage_type` identifies what kind of computation this is.  
- `sample_id` groups stages into their sample.  
- `iteration` places the sample in the optimization timeline.

This means inspecting a record always tells you where in the causal chain it sits, what produced it, and what consumes it. No need to reconstruct this from filenames or directory structure.

## 12\. Orchestration: Custom Lightweight DAG Runner

### 12.1 Why Custom

Full workflow orchestration frameworks (Dagster, Prefect, Airflow) provide a lot — DAG management, retry logic, scheduling, UIs — but they are also frameworks you build *inside* rather than *on top of*. For a research project where the pipeline itself is the research artifact, a thinner custom layer provides more control, less conceptual overhead, and tighter integration with the sampling and optimization logic that is unique to this project.

### 12.2 Core Responsibilities

The DAG runner must:

1. **Accept a topology definition** (Section 2.4) and a population of sample configurations.  
2. **Resolve the DAG** for each sample: determine which stages need to be computed, check the content-addressable store for cached results, and enqueue only the stages that are missing.  
3. **Dispatch stages** to the inference endpoint asynchronously, respecting dependency order (a stage cannot run until its parents are complete).  
4. **Persist results** atomically as each stage completes (Section 13).  
5. **Handle failures** by re-enqueueing failed stages (Section 13).  
6. **Report progress** to the observability layer (Section 14).

### 12.3 Implementation Sketch

The runner is an async Python process built on asyncio. Key components:

- **Work queue**: An async queue of (sample\_id, stage\_type, input\_hashes) tuples. The runner pulls from this queue and dispatches inference calls.  
- **Completion handler**: When a stage completes, its result is persisted to DuckDB, its content hash is registered, and any downstream stages that were waiting on it are enqueued.  
- **Task claiming**: If multiple workers are ever needed (e.g., with exo providing multiple inference slots), stages in the queue need a simple claim mechanism to prevent double-execution. A "claimed\_at" timestamp on the queue entry, with a timeout for reclaiming abandoned work, is sufficient.

### 12.4 Beam Iteration Loop

The outermost loop:

1. Load current beam population from DuckDB.  
2. For each sample in the beam, generate variant configurations (mutated personas, sampled parameters).  
3. For each candidate (existing \+ new), resolve the DAG and enqueue missing stages.  
4. Run the DAG runner until all candidates are fully evaluated.  
5. Rank the candidates (Section 5).  
6. Cull the population to beam size.  
7. Persist the new beam state.  
8. Repeat.

## 13\. Fault Tolerance

### 13.1 The Problem

Over weeks of continuous operation, the system will encounter crashes, power interruptions, OOM kills, Ollama hangs, network errors (if using remote APIs), and other failures. The system must tolerate all of these without losing completed work or corrupting state.

### 13.2 Atomic Persistence

Every completed evaluation step is persisted to DuckDB immediately upon completion. Writes should be atomic — a step's result is either fully written or not written at all. DuckDB supports transactions, so this is straightforward: begin transaction, insert the computation event, commit. If the process dies mid-write, the incomplete transaction is rolled back and the step is treated as not-yet-computed on restart.

### 13.3 Resumption from Incomplete State

On startup (or after a crash), the system:

1. Loads the current beam state from DuckDB.  
2. For each sample in the beam (and any in-flight candidates), resolves the DAG against the content-addressable store to find which stages are complete and which are missing.  
3. Enqueues the missing stages.  
4. Resumes execution.

No special checkpoint files or restart logic needed — the content-addressable store *is* the checkpoint. If a stage was completed and persisted before the crash, its hash exists and it will be skipped. If it was in progress, its hash does not exist and it will be re-run. The result of the re-run may differ from what the first attempt would have produced, but this does not matter — each stage's output is valid regardless, and for Monte Carlo sampling, the randomness is a feature.

### 13.4 Work Queue Recovery

If the runner crashes while stages are claimed but not completed, the claim timeout mechanism (Section 12.3) ensures they are eventually reclaimed and retried. No manual intervention is needed.

### 13.5 Ollama-Specific Resilience

Ollama can hang or become unresponsive under load. The inference dispatch layer should:

- Set reasonable timeouts on all API calls.  
- Detect hung connections and retry after a backoff.  
- Optionally monitor the Ollama process and restart it if it becomes unresponsive (a simple health-check loop).

## 14\. Observability and Dashboard

### 14.1 Why It Matters

When a system runs for weeks unattended, you need to be able to check in and understand what is happening without interrupting it. The dashboard is not a nice-to-have — it is critical infrastructure.

### 14.2 Key Metrics

At minimum, the dashboard should show:

- **Current beam statistics**: population size, score distribution (min, max, mean, median, quartiles), score distribution over time (per iteration).  
- **Throughput**: samples completed per hour, stages completed per hour, average stage duration.  
- **Error rates**: failed stages, retries, timeouts.  
- **Persona evolution**: which personas are surviving vs. being culled, lineage visualization (which parents produced which offspring).  
- **Current state**: what is currently being computed, what is queued, estimated time to next iteration completion.

### 14.3 Implementation

A simple approach: the DAG runner emits structured log events (JSON lines or similar) that a separate lightweight process reads and aggregates into a dashboard. The dashboard can be a simple web page served by a local HTTP server (e.g., FastAPI \+ htmx, or a Jupyter notebook with auto-refreshing widgets).

More sophisticated: use OpenTelemetry (Section 14.4) to emit spans and metrics, and view them in a tracing UI.

### 14.4 OpenTelemetry and Distributed Tracing

OpenTelemetry is a standard for emitting structured telemetry (traces, metrics, logs) from applications. Each computation stage can be modeled as a "span" with a parent span, start/end times, and metadata. Tracing UIs (Jaeger, Grafana Tempo) display these as waterfall diagrams that show the causal chain and timing of each stage — exactly the phasing and causal inspection the storage layer is designed to support.

This may be overkill for the initial implementation, but designing the system so that telemetry emission is a pluggable concern (a hook that fires when stages start and complete) means it can be added later without restructuring. Even if a full tracing UI is not used initially, emitting spans to a local collector is cheap and provides a rich data source for debugging and analysis.

## 15\. Python Stack and Dependencies

### 15.1 Core Framework

- **Python 3.12+** (or latest stable).  
- **asyncio** for concurrency. The system is I/O-bound (waiting on LLM inference), so async is the right concurrency model. It also means the system transparently benefits from additional inference resources without code changes.  
- **pydantic** for data modeling, configuration validation, and structured output parsing. Pydantic models define the schema for computation events, personas, topology definitions, and all structured data flowing through the system.

### 15.2 LLM Interface

- **litellm** as the API abstraction layer under the builder pattern. litellm normalizes across OpenAI, Anthropic, Ollama, and many other providers with a unified calling convention. The builder pattern (Section 7\) sits on top of litellm, adding persona injection, configuration staging, and field filtering.  
- **httpx** (async) as the HTTP client for direct API calls if needed, and as the underlying transport for litellm's async mode.

### 15.3 Data

- **duckdb** (Python bindings) for the primary data store.  
- **pyarrow** or **polars** for Parquet I/O and analytical queries. Polars is faster and more ergonomic than pandas for the kinds of filter/aggregate/join operations needed here.

### 15.4 Observability

- **opentelemetry-api** and **opentelemetry-sdk** (optional, added when needed) for span emission.  
- A simple web server (FastAPI or similar) for the dashboard.

### 15.5 Hashing

- **hashlib** (stdlib) for content-addressable hashing. SHA-256 is fine; the hashes are for identity, not security.

## 16\. Project Structure (Initial)

A suggested starting layout:

llm-interrogator-evo/

    DESIGN.md                   \# this document

    pyproject.toml              \# project metadata, dependencies

    src/

        interrogator/

            \_\_init\_\_.py

            agents/             \# agent role definitions, persona management

                actor.py

                grader.py

                director.py

                producer.py

                critic.py

                base.py         \# base agent class, persona loading

            builder/            \# query builder pattern, litellm integration

                query\_builder.py

                providers.py    \# provider-specific field mapping

            dag/                \# DAG runner, topology, orchestration

                topology.py     \# topology-as-data, DAG validation

                runner.py       \# async DAG execution engine

                queue.py        \# work queue, task claiming

            evolution/          \# beam search, mutation, ranking

                beam.py         \# beam population management

                mutation.py     \# three-phase mutation pipeline

                ranking.py      \# ranking module interface

                ranking\_pairwise.py

                ranking\_depth.py

            storage/            \# DuckDB, content-addressable store

                store.py        \# core storage operations

                hashing.py      \# content-addressable hashing

                schema.py       \# table definitions, migrations

            observability/      \# dashboard, telemetry

                dashboard.py

                telemetry.py    \# OpenTelemetry span emission (pluggable)

            config.py           \# top-level configuration, defaults

            sample.py           \# sample data model

    tests/

    scripts/                    \# analysis scripts for paper-writing, run inspection

    data/                       \# default data directory (configurable)

## 17\. Minimal Vertical Slice

Before building the full system, validate the core loop with a minimal configuration:

- **One scenario** (human-authored by the PI).  
- **One Theater model** (a local model via Ollama).  
- **Two roles only**: Actor and Grader. No Director, Producer, or Critic.  
- **Small beam**: \~20 samples.  
- **Simple ranking**: absolute ratings with no depth recursion.  
- **Simple mutations**: Phase 2 \+ Phase 3 only (skip Phase 1 judgment; the Grader's output serves as the judgment).  
- **DuckDB storage** from the start — it is no harder to set up than JSON files, and avoids a migration later.

This slice validates:

1. The Actor-Theater interaction produces meaningful conversations.  
2. The Grader can distinguish good from bad Actor performance.  
3. The mutation pipeline produces coherent persona variants.  
4. The beam search loop converges — scores improve over iterations.  
5. The storage and resumption logic works.  
6. The async dispatch and Ollama integration are reliable over multi-hour runs.

Once this loop is running and producing improving scores, add roles incrementally: Director, then Producer, then Critic. Add ranking module alternatives. Add topology variation. Scale the beam.

## 18\. Open Questions

- **Optimal topology**: What supervisory structure produces the best optimization signal? This is an empirical question, enabled by the topology-as-data design.  
- **Ranking method**: Depth rating vs. pairwise comparison vs. hybrid — which produces the most useful selection signal? Validate early with the minimal slice.  
- **Sample space dimensionality**: How many parameters should be randomized per sample? Start narrow, expand as the system matures.  
- **Generalization across Theater models**: Sampling the Theater model forces generality but may constrain expressivity to the lowest common denominator. Whether to prioritize generality, expressivity, or a hybrid remains open.  
- **Screenwriter automation**: When and how to transition from human-authored scenarios to LLM-generated ones, and how to prevent scenario gaming.  
- **Activation engineering and beyond-prompt conditioning**: Whether local models can be shaped via activation pinning, weight subsetting, or other techniques that outperform prompt-prefix personas. The builder pattern accommodates this, but the empirical value is unknown.  
- **Role-recursion depth sensitivity**: How many meta-evaluation layers actually help, and when do additional layers add noise rather than signal?

