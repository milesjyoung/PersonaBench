# Running PersonaBench

## Quick start

```bash
git clone https://github.com/TruePersona/PersonaBench.git
cd PersonaBench
pip install anthropic openai   # whichever SDK you need

export ANTHROPIC_API_KEY="sk-ant-..."   # if using Anthropic
export OPENAI_API_KEY="sk-..."          # if using OpenAI

python step6_benchmark_runner/generator.py \
  --app-logs step4_app_log_synthesizer/data_samples/output/julio_simmons_app_logs.json \
  --test-cases step5_testcases_synthesis/data_samples/output/julio_simmons_test_cases.json \
  --provider openai \
  --model-pass1 gpt-5.5 \
  --model-pass2 gpt-5.5
```

Repeat for each persona: `mary_alberti`, `alicia_gonzalez`, `deeva_cintron`, `maria_buendia`.

Outputs land in `step6_benchmark_runner/data_samples/output/`.

## Prerequisites

- Python 3.10+
- `pip install anthropic` or `pip install openai`
- Repo cloned, working directory at repo root

## Running a single step

Use `--start` and `--stop` with the same step number:

```bash
python openclaw_pipeline.py --persona julio_simmons --start 4 --stop 4 --provider openai --model gpt-5.5 --verifier-model gpt-5.5
```

Steps: 2 (interview), 3 (social circle), 4 (app logs), 5 (test cases), 6 (benchmark).

## Running the full pipeline

```bash
python openclaw_pipeline.py --persona julio_simmons --start 2 --stop 6 --provider openai --model gpt-5.5 --verifier-model gpt-5.5
```

## Running from a coding agent

Open the repo in Codex, Claude Code, or Cursor and run the commands above.

## Pass 1 / Pass 2 independence

Use different models for Pass 1 (answering) and Pass 2 (scoring).
