# Running PersonaBench

## Quick start (check a model's score)

```bash
git clone https://github.com/TruePersona/PersonaBench.git
cd PersonaBench
pip install anthropic openai   # whichever SDK you need

export ANTHROPIC_API_KEY="sk-ant-..."   # if using Anthropic
export OPENAI_API_KEY="sk-..."          # if using OpenAI

for P in julio_simmons mary_alberti alicia_gonzalez deeva_cintron maria_buendia; do
  bash run_step.sh 6 "$P" openai gpt-5.5
done
```

Wall time: 30 to 60 minutes. Outputs land in `step6_benchmark_runner/data_samples/output/`. Compare against the figures in `step6_benchmark_runner/RESULTS.md`.

## Prerequisites

- Python 3.10 or newer
- `pip install anthropic` or `pip install openai`
- Repo cloned, working directory at the repo root

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

## Three ways to run (advanced)

### A. Direct API

Call the step generators directly:

```bash
python step6_benchmark_runner/generator.py \
  --app-logs step4_app_log_synthesizer/data_samples/output/julio_simmons_app_logs.json \
  --test-cases step5_testcases_synthesis/data_samples/output/julio_simmons_test_cases.json \
  --provider openai \
  --model-pass1 gpt-5.5 \
  --model-pass2 gpt-5.5
```

### B. Pipeline orchestrator

```bash
python openclaw_pipeline.py --persona julio_simmons --start 2 --stop 6 --provider openai --model gpt-5.5 --verifier-model gpt-5.5
```

### C. From inside a coding agent

Open the repo in Codex, Claude Code, Cursor, or any coding agent with shell access. Run the shell commands above.

## Personas

`julio_simmons`, `mary_alberti`, `alicia_gonzalez`, `deeva_cintron`, `maria_buendia`

## Models

Use exact version strings (`gpt-5.5`, `claude-opus-4-7`, `gemini-2.5-pro`), not aliases. The shell scripts accept provider and model as arguments. Defaults are `anthropic` and `claude-opus-4-7`.

## Reproducing the published numbers

The dataset on `main` is the canonical artifact. To check a model's score, run step 6 only:

```bash
bash run_step.sh 6 <persona> openai gpt-5.5
```

## Outputs

Each step writes to its own `data_samples/output/` directory. Benchmark results land in `step6_benchmark_runner/data_samples/output/`.

## Pass 1 / Pass 2 independence

Use different models for Pass 1 (answering) and Pass 2 (scoring).
