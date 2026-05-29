# Running PersonaBench

PersonaBench measures whether an LLM can infer a persona's identity from raw app logs and act safely as their personal assistant.

## Prerequisites

```
Python 3.10+
pip install anthropic openai datasets
```

## Authentication

**Subscription CLI** (no API key needed): install and log in to the Claude Code CLI (`claude -p`) or the Codex CLI (`codex exec`). Select with `--backend claude` or `--backend codex`.

**API key** (metered): set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment. Select with `--backend anthropic-api` or `--backend openai-api`.

Verify your backend works before starting a long run: `python openclaw_pipeline.py --help`.

## Score a model against the existing dataset

All 5 personas have complete data through Step 5. To benchmark a model, run Step 6 only:

```bash
# Benchmark GPT 5 with GPT 5.4 as judge (Codex subscription)
python openclaw_pipeline.py --persona julio_simmons --start 6 --stop 6 \
  --backend codex \
  --model gpt-5.4-mini \
  --judge-model gpt-5.4

# Benchmark Opus 4.7 with Sonnet 4.6 as judge (Claude subscription)
python openclaw_pipeline.py --persona julio_simmons --start 6 --stop 6 \
  --backend claude \
  --model claude-opus-4-7 \
  --judge-model claude-sonnet-4-6

# Score all 5 personas at once
python openclaw_pipeline.py --all --start 6 --stop 6 \
  --backend codex \
  --model gpt-5.4-mini \
  --judge-model gpt-5.4
```

## Generate a new persona end-to-end

**Step 1: fetch a seed.** Pick a UUID from the NVIDIA Nemotron-Personas-USA dataset and fetch it:

```bash
python step1_seed/generator.py \
  --uuid 50f90a6f17de473f9ca15f00afdedf7a \
  --output step1_seed/data_samples/output/
```

**Steps 2-5: generate the dataset.** This produces the interview, social circle, app logs, and test cases:

```bash
python openclaw_pipeline.py \
  --seed step1_seed/data_samples/output/alicia_gonzalez_seed.json \
  --start 2 --stop 5 \
  --backend codex \
  --model gpt-5.5 \
  --verifier-model gpt-5.4
```

**Step 6: benchmark.** Run separately to maintain evaluator independence:

```bash
python openclaw_pipeline.py \
  --persona alicia_gonzalez --start 6 --stop 6 \
  --backend codex \
  --model gpt-5.4-mini \
  --judge-model gpt-5.4
```

Time estimate: 3-5 hours per persona (dominated by Step 4).

## Run a single step

```bash
python openclaw_pipeline.py --persona alicia_gonzalez --start 4 --stop 4 --backend claude
```

Each step's generator also accepts direct invocation with `--help` for its full CLI. See each step's README for details.

## Model independence

The pipeline uses four model roles. Three independence boundaries prevent systematic bias:

| Boundary | Requirement | Why |
|---|---|---|
| Generator differs from Verifier | The model that creates app log fragments must not verify its own output. | A model pattern-matching its own generation is not an independent recoverability test. |
| Evaluator differs from Judge | The model being benchmarked must not grade its own answers. | Same-model self-grading inflates scores by ~27% (empirically measured). |
| Generator differs from Judge | The model that created the test cases must not grade answers to them. | Shared systematic biases in question construction and answer evaluation. |

**Default model assignments:**

| Role | Claude | GPT | Flag |
|---|---|---|---|
| Generator | `claude-opus-4-7` | `gpt-5.5` | `--model` (Steps 2-5) |
| Verifier | `claude-sonnet-4-6` | `gpt-5.4` | `--verifier-model` |
| Evaluator | (run Step 6 separately) | `gpt-5.4-mini` | `--model` (Step 6 only) |
| Judge | `claude-sonnet-4-6` | `gpt-5.4` | `--judge-model` |

**OpenAI backend (openai-api, codex) reasoning defaults:**

- Generation steps (Steps 2-5, Step 6 Pass 2) use `high` reasoning effort.
- Evaluated responses (Step 6 Pass 1) use `low` reasoning effort.

Always run Step 6 in a separate invocation (`--start 6 --stop 6`) so the evaluator model can differ from the generator.

## Troubleshooting

**`ValueError: Streaming is required`** -- Update to the latest `llm.py`. The pipeline uses streaming by default.

**`UnicodeEncodeError: 'charmap'`** -- Set `$env:PYTHONIOENCODING = "utf-8"` or use Windows Terminal.

**`JSONDecodeError: Invalid \escape`** -- The generator sanitizes these automatically.

**Step 4 crashes mid-run** -- Trace and fragment files are saved after every fact. Re-running regenerates from scratch.

**AMBIGUOUS after 3 attempts** -- Fragments are merged anyway; the trace records `"passed": false`.
