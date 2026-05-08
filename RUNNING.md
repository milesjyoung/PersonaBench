# Running PersonaBench

## Prerequisites

- Python 3.10+
- `pip install anthropic` or `pip install openai`
- Repo cloned, working directory at repo root
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` set in your environment

## Run a single step

```bash
bash run_step.sh <step> <persona> [provider] [model]
```

Steps: 2 (interview), 3 (social circle), 4 (app logs), 5 (test cases), 6 (benchmark).

Examples:

```bash
bash run_step.sh 6 julio_simmons openai gpt-5.5
bash run_step.sh 4 maria_buendia anthropic claude-opus-4-7
bash run_step.sh 6 julio_simmons                  # defaults: anthropic, claude-opus-4-7
```

## Run the full pipeline (steps 2 through 6)

```bash
bash run_pipeline.sh <persona> [provider] [model]
```

Examples:

```bash
bash run_pipeline.sh julio_simmons openai gpt-5.5
bash run_pipeline.sh julio_simmons                 # defaults: anthropic, claude-opus-4-7
```

## Personas

`julio_simmons`, `mary_alberti`, `alicia_gonzalez`, `deeva_cintron`, `maria_buendia`

## Outputs

Each step writes to its own `data_samples/output/` directory. Benchmark results land in `step6_benchmark_runner/data_samples/output/`.

## Running from a coding agent

Open the repo in Codex, Claude Code, or Cursor and run the commands above.

## Pass 1 / Pass 2 independence

Use different models for Pass 1 (answering) and Pass 2 (scoring).
