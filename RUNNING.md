# Running PersonaBench

## Quick start (check a model's score)

```bash
git clone https://github.com/TruePersona/PersonaBench.git
cd PersonaBench
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

for P in julio_simmons mary_alberti alicia_gonzalez deeva_cintron maria_buendia; do
  bash run_step.sh 6 "$P"
done
```

Wall time: 30 to 60 minutes. Outputs land in `step6_benchmark_runner/data_samples/output/`. Compare against the figures in `step6_benchmark_runner/RESULTS.md`.

## Prerequisites

- Python 3.10 or newer
- `pip install anthropic` (only if running with the API)
- Repo cloned, working directory at the repo root

## Three ways to run

### A. Anthropic API

Set `ANTHROPIC_API_KEY` and run:

```bash
python Top_level_generator.py \
  --seed step1_seed/data_samples/output/julio_simmons_seed.json \
  --start 2 --stop 6 \
  --model claude-opus-4-7 \
  --verifier-model claude-sonnet-4-6
```

For all five personas, loop over the `*_seed.json` files in `step1_seed/data_samples/output/`.

### B. Claude CLI on subscription

Install the `claude` CLI and sign in with `claude login`. Then either:

```bash
python openclaw_pipeline.py --persona julio_simmons --start 2 --stop 6
```

or use the bash entry points:

```bash
bash run_pipeline.sh julio_simmons         # all six steps
bash run_step.sh 4 julio_simmons           # one step
```

To run all five personas, loop the persona names in the shell.

### C. From inside a coding tool

Open the repo in Claude Code, Cursor, Antigravity, Codex, Continue, Cline, or Devin. Ask the tool to run:

```
bash run_pipeline.sh julio_simmons
```

The tool's shell tool invokes the same scripts as B. To run a single step, ask it to run `bash run_step.sh <step> <persona>`.

## Models

Edit the `CONFIG` block at the top of `run_pipeline.sh` to change models. Use exact version strings (`claude-opus-4-7`), not aliases (`opus`). Aliases are rejected at parse time.

## Reproducing the published numbers

The dataset on `main` is the canonical artifact. To check a model's score, run step 6 only:

```bash
bash run_step.sh 6 <persona>
```

Run three times for a variance estimate. To regenerate the full dataset (steps 2 through 6), use any of A, B, or C above. Outputs are equivalent across runs but not byte-identical because LLM responses are stochastic. See [ARCHITECTURE.md](ARCHITECTURE.md) for details.
