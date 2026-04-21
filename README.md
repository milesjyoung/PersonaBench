# PersonaBench

A benchmark for evaluating personalized AI behavior and user understanding.

## Goal

PersonaBench measures whether an LLM can:

1. **Infer** a persona's identity, preferences, and life patterns from raw app logs alone.
2. **Act** as a useful, safe personal agent for that persona given those inferred traits.

Rather than testing generic task completion, PersonaBench probes deep user context — the kind of behavioral, relational, and safety-relevant signals that a personal assistant would need to serve a user well. A SOTA baseline is evaluated against the benchmark to expose current model limitations.

## Pipeline

Each persona is constructed through a six-step pipeline. Every step has its own generator, verifier, and README.

| Step | What it does |
|---|---|
| [step1_seed](step1_seed) | Pull baseline demographics from the NVIDIA Nemotron-Personas-USA dataset. |
| [step2_interview](step2_interview) | Generate a 109-question life-history interview and a structured profile with a `hidden_facts` registry (60-150 atomic facts per persona). |
| [step3_social_circle](step3_social_circle) | Synthesize the five people with whom the persona most frequently exchanges personal text messages. |
| [step4_app_log_synthesizer](step4_app_log_synthesizer) | Generate mobile app activity logs (messenger + calendar) that implicitly encode each hidden fact through behavioral signals distributed across multiple sources. |
| [step5_testcases_synthesis](step5_testcases_synthesis) | Construct typed test cases (five types, including agent-behavior scenarios) grounded in the hidden_facts registry. |
| [step6_benchmark_runner](step6_benchmark_runner) | Execute the benchmark against a frontier LLM using a two-pass design that prevents ground-truth leakage. |

## Scope

The current pipeline produces **5 personas end-to-end** for initial validation. The target at benchmark release is **100 personas**. The 5-persona run validates the pipeline before scaling.

## Getting started

Run `generator.py` inside each step folder in order, from step1 through step6. Each step consumes the previous step's output from `data_samples/output/` and writes its own output to `data_samples/output/`. `Top_level_generator.py` at the repo root chains all six steps for a single persona.

## References

- Park et al. 2024 — *Generative Agent Simulations of 1,000 People.* [[paper]](https://arxiv.org/abs/2411.10109) — 109 interview questions and verification approach.
- Ge et al. 2024 — *PersonaHub (Tencent AI Lab).* [[paper]](https://arxiv.org/abs/2406.20094) — Persona-to-Persona method for generating social circles.
- Jiang et al. 2025 — *PersonaMem-v2.* [[paper]](https://arxiv.org/abs/2502.15910) — Design principles: implicit preferences, broad domains, compact summaries.
- Jimenez et al. 2024 — *SWE-bench Verified.* [[paper]](https://openai.com/index/introducing-swe-bench-verified/) — Human-in-the-loop test case verification methodology.
- NVIDIA Nemotron-Personas-USA. [[dataset]](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA) — Starter people (seed data).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All commits must include a DCO sign-off.

## License

[Apache 2.0](LICENSE).
