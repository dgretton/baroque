# Source Index

Research started: 2026-04-29

This file records the external sources used for the initial design check. Preference is given to official documentation, model cards, and peer-reviewed or preprint research pages.

## Current Gemma and Local Runtime

- Google, "Gemma 4: Byte for byte, the most capable open models" (2026-04-02): https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
  - Primary source for Gemma 4 release timing, sizes, Apache 2.0 licensing, agentic capabilities, function calling, structured JSON, context lengths, and supported deployment tools.
- Google Hugging Face model card, `google/gemma-4-26B-A4B-it`: https://huggingface.co/google/gemma-4-26B-A4B-it
  - Primary source for Gemma 4 model family details: E2B, E4B, 26B A4B MoE, 31B Dense, context windows, modalities, active parameter counts, thinking mode, and sampling defaults.
- Ollama Gemma 4 library page: https://ollama.com/library/gemma4
  - Primary source for Ollama tags, local package sizes, context windows, and run commands for `gemma4:e2b`, `gemma4:e4b`, `gemma4:26b`, and `gemma4:31b`.
- Ollama OpenAI compatibility docs: https://docs.ollama.com/api/openai-compatibility
  - Source for supported OpenAI-compatible chat fields, JSON mode, tools, reasoning controls, logprobs, model listing, local pull workflow, and context-size Modelfile workaround.
- Ollama Modelfile reference: https://docs.ollama.com/modelfile
  - Source for `FROM`, `PARAMETER`, `TEMPLATE`, `SYSTEM`, `ADAPTER`, and context-window configuration.
- Ollama model import docs: https://docs.ollama.com/import
  - Source for importing Safetensors or GGUF models/adapters and using LoRA adapters with `ADAPTER`.
- LiteLLM docs: https://docs.litellm.ai/docs/
  - Source for using one OpenAI-format interface across providers, Ollama support, retry/fallback logic, LiteLLM proxy, routing, load balancing, and observability hooks.

## Beyond String Prompting

- Hugging Face PEFT docs: https://huggingface.co/docs/peft/v0.6.1/en/index
  - Source for PEFT framing: adapt pretrained models by training a small number of extra parameters; supported methods include LoRA, prefix tuning, prompt tuning, P-tuning, and IA3.
- Hugging Face PEFT prefix tuning docs: https://huggingface.co/docs/peft/v0.17.0/package_reference/prefix_tuning
  - Source for continuous prefix vectors/virtual tokens learned while keeping model weights frozen.
- Hugging Face Transformers PEFT adapter docs: https://huggingface.co/docs/transformers/main/en/main_classes/peft
  - Source for runtime adapter injection and the distinction between injectable adapters and prompt-learning methods.
- Turner et al., "Activation Addition: Steering Language Models Without Optimization" (arXiv:2308.10248): https://huggingface.co/papers/2308.10248
  - Source for inference-time activation steering using activation differences/steering vectors.
- Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency" (arXiv:2310.01405): https://arxiv.gg/abs/2310.01405
  - Source for representation engineering as population-level monitoring and manipulation of high-level representations.
- Anthropic, "Golden Gate Claude" (2024-05-23): https://www.anthropic.com/news/golden-gate-claude
  - Source for the public demonstration of direct feature activation steering as distinct from system prompting and fine-tuning.
- Google DeepMind, "Gemma Scope": https://deepmind.google/models/gemma/gemma-scope/
  - Source for Gemma Scope and Gemma Scope 2, including Gemma 3 feature exploration and steering via Neuronpedia.
- Google DeepMind, "Gemma Scope: helping the safety community shed light on the inner workings of language models" (2024-07-31): https://deepmind.google/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/
  - Source for the original Gemma Scope release: hundreds of open sparse autoencoders for Gemma 2 2B and 9B.

## Optimizing LLM Pipelines and Evaluators

- Khattab et al., "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines" (arXiv:2310.03714): https://huggingface.co/papers/2310.03714
  - Source for optimizing LM pipelines as text transformation graphs with parameterized modules.
- Yuksekgonul et al., "TextGrad: Automatic Differentiation via Text" (arXiv:2406.07496): https://huggingface.co/papers/2406.07496
  - Source for using text-based feedback to optimize variables in compound AI systems.
- Yang et al., "Large Language Models as Optimizers" / OPRO (arXiv:2309.03409): https://huggingface.co/papers/2309.03409
  - Source for LLMs proposing improved prompts/solutions as an optimization method.
- Liu et al., "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment" (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.153/
  - Source for LLM-as-judge with chain-of-thought/form filling and known evaluator bias concerns.
- Chiang et al., "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference" (ICML 2024): https://proceedings.mlr.press/v235/chiang24b.html
  - Source for pairwise comparison as a scalable preference-evaluation method.
- Liu et al., "Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators" (COLM 2024): https://research.monash.edu/en/publications/aligning-with-human-judgement-the-role-of-pairwise-preference-in-/
  - Source for pairwise-preference search outperforming direct scoring in LLM evaluation alignment.

## Scaling, Operations, and Infrastructure

- exo GitHub repository: https://github.com/exo-explore/exo
  - Source for local multi-device inference clustering and OpenAI/Claude/Responses/Ollama-compatible APIs.
- vLLM OpenAI-compatible server docs: https://docs.vllm.ai/serving/openai_compatible_server.html
  - Source for serving local/cloud models behind OpenAI-compatible Chat, Completions, and Responses APIs.
- Ray resource scheduling docs: https://docs.ray.io/en/latest/ray-core/scheduling/resources.html
  - Source for Python distributed task/actor CPU, GPU, and custom-resource scheduling.
- Ray actor docs: https://docs.ray.io/en/latest/ray-core/actors.html
  - Source for Ray actors as stateful workers, resource requirements, and restart/retry options.
- Prefect workflow orchestration overview: https://www.prefect.io/how-it-works
  - Source for Python-native flows/tasks, local execution with cloud/server tracking, retries, state, and observability.
- Temporal durable execution overview: https://temporal.io/
  - Source for durable workflows, automatic state capture, retries, task queues, signals, timers, and long-running workflow recovery.
- DuckDB S3 API support: https://duckdb.org/docs/stable/core_extensions/httpfs/s3api
  - Source for reading/writing/globbing files on S3-compatible object storage, including Parquet workflows.
- OpenTelemetry Python docs: https://opentelemetry.io/docs/languages/python/
  - Source for Python traces, metrics, logs status, and supported Python versions.
- OpenTelemetry Python instrumentation docs: https://opentelemetry.io/docs/languages/python/instrumentation/
  - Source for manual traces, spans, metrics, and log instrumentation.
- structlog stable docs: https://www.structlog.org/en/stable/
  - Source for structured JSON/logfmt/console logging, asyncio/contextvars support, and Python logging integration.
- Python asyncio queues docs: https://docs.python.org/3/library/asyncio-queue.html
  - Source for async queues, worker patterns, and join/task_done coordination.
- Python asyncio tasks docs: https://docs.python.org/3.12/library/asyncio-task.html
  - Source for task scheduling, cancellation behavior, and cleanup recommendations.
- Kubernetes GPU scheduling docs: https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
  - Source for GPU scheduling through Kubernetes device plugins.
- Modal GPU docs: https://modal.com/docs/guide/gpu
  - Source for cloud GPU function execution, GPU type/count selection, and GPU fallbacks.
- Modal introduction docs: https://modal.com/docs/guide
  - Source for serverless Python infrastructure, autoscaling, batch jobs, and low-latency inference.
- RunPod docs: https://docs.runpod.io/
  - Source for GPU pods, serverless endpoints, and scalable GPU/CPU resources.
- Lambda Cloud docs: https://docs.lambda.ai/
  - Source for on-demand GPU instances, larger H100/B200 clusters, Kubernetes, and Slurm options.
- NVIDIA Dynamo introduction: https://docs.dynamo.nvidia.com/dynamo/getting-started/introduction
  - Source for distributed inference serving, engine-agnostic vLLM/SGLang/TensorRT-LLM support, KV-cache routing/offloading, autoscaling, observability, and fault tolerance.
