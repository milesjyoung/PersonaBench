# Running PersonaBench

PersonaBench supports three execution backends. All three produce the same outputs given the same seed, model, and prompt — choose the one that matches your tooling.

| Backend | How you run it | What it uses | When to pick it |
|---|---|---|---|
| [A. Direct API](#backend-a-direct-api) | `python Top_level_generator.py` | Anthropic SDK + API key | Headless batch runs, CI, or reproduction by anyone with an API key |
| [B. Vendor CLI](#backend-b-vendor-cli) | `python openclaw_pipeline.py` | `claude` CLI on a subscription | Running on subscription pricing instead of per-token API |
| [C. AI coding agent with subagents](#backend-c-ai-coding-agent-with-subagents) | Open the repo inside the agent, follow the runbook | Claude Code, CodenX, Antigravity, Cursor Agents, Devin, etc. | Interactive orchestration, true per-subagent model independence, long-context subagents |

---

## Backend A: Direct API

Canonical clone-and-run path. Each step has its own `generator.py` using the Anthropic SDK.

### Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Per-step invocation

```bash
python step2_interview/generator.py \
  --seed step1_seed/data_samples/output/{persona}_seed.json

python step3_social_circle/generator.py \
  --profile step2_interview/data_samples/output/{persona}_verification.json \
  --transcript step2_interview/data_samples/output/{persona}_interview.json

python step4_app_log_synthesizer/generator.py \
  --profile step2_interview/data_samples/output/{persona}_verification.json \
  --social-circle step3_social_circle/data_samples/output/{persona}_social_circle_verification.json \
  --model claude-opus-4-6 \
  --verifier-model claude-sonnet-4-6

python step5_testcases_synthesis/generator.py \
  --profile step2_interview/data_samples/output/{persona}_verification.json \
  --app-logs step4_app_log_synthesizer/data_samples/output/{persona}_app_logs.json \
  --social-circle step3_social_circle/data_samples/output/{persona}_social_circle_verification.json

python step6_benchmark_runner/generator.py \
  --app-logs step4_app_log_synthesizer/data_samples/output/{persona}_app_logs.json \
  --test-cases step5_testcases_synthesis/data_samples/output/{persona}_test_cases.json
```

### End-to-end

```bash
python Top_level_generator.py \
  --seed step1_seed/data_samples/output/julio_simmons_seed.json \
  --start 2 --stop 6
```

### Notes

- Step 4's `--verifier-model` defaults to a different model family than `--model` so the reverse-inferability gate is not pattern-matching its own generator.
- Step 6 Pass 1 and Pass 2 can use different models via `--model-pass1` and `--model-pass2` (recommended for independence).
- Costs scale with the `hidden_facts` count per persona. For ~150 facts, expect several hundred API calls in Step 4 alone.

---

## Backend B: Vendor CLI

Runs the pipeline through a vendor CLI on a subscription. Reference implementation is `openclaw_pipeline.py`, written against the `claude` CLI.

### Setup

Install the vendor CLI and put it on PATH. For the reference implementation:

```bash
# Install the claude CLI per vendor docs
export CLAUDE_CMD=claude  # or claude.cmd on Windows
```

### End-to-end

```bash
python openclaw_pipeline.py \
  --seed step1_seed/data_samples/output/julio_simmons_seed.json \
  --start 4 --stop 6

# All five personas
python openclaw_pipeline.py --all --start 2 --stop 6
```

### Step 6 isolation

Step 6's benchmark runner passes `--openclaw` to its own `generator.py`, which deletes the CLI session directory between each test case. Without this, prior answers leak into context and invalidate accuracy numbers.

### Adapting to other vendor CLIs

`openclaw_pipeline.py` assumes the vendor CLI:
- accepts a prompt on stdin with a flag like `-p`
- returns the model's response on stdout
- exits non-zero on failure

For a different vendor's CLI, replace `call_claude()` in `openclaw_pipeline.py` with a wrapper that matches the vendor's invocation shape. The rest of the pipeline is vendor-agnostic.

### Model independence caveat

On a single subscription, fragment generation (Step 4) and its reverse-inferability verifier may run on the same model. This weakens the "independent verifier" argument compared to Backend A, which can route the verifier to a different model family. Prompt-level isolation still holds (the verifier sees fragments only, not the hidden fact), but users who need true model independence should use Backend A or Backend C.

---

## Backend C: AI coding agent with subagents

Runs the pipeline inside any AI coding agent that exposes a subagent / Task / Agent orchestration tool — Claude Code, CodenX, Antigravity, Cursor Agents, Devin, or equivalent.

### Why this backend exists

The orchestrating agent spawns a fresh subagent per inference call. This gives two properties that the API and CLI backends have to approximate:

- **Inherent context isolation.** Every subagent starts with a blank context. No session state leaks between calls. Step 6's `--openclaw` session-delete hack is unnecessary here.
- **Per-subagent model selection.** Fragment generation and reverse-inferability verification can be routed to different models (e.g., opus for generation, sonnet/haiku for verification), giving true model independence for Step 4's gate without needing two separate API keys.

### Prerequisites

- An agent with a subagent tool. Claude Code has `Agent`; CodenX, Antigravity, and Cursor Agents have equivalent mechanisms.
- The agent needs file read/write access to the repo.
- A JSON extraction helper on the orchestrator side (regex + `json.loads`).

### Runbook

The orchestrator reads inputs from disk, spawns subagents with each step's `prompt.txt` filled in, and writes outputs back to disk. The pattern below is vendor-neutral; translate the pseudocode into your agent's subagent API.

#### Step 4 — App Log Synthesis

Inputs:
- `step2_interview/data_samples/output/{persona}_verification.json` → `corrected_extracted_profile.hidden_facts`
- `step3_social_circle/data_samples/output/{persona}_social_circle_verification.json` → `corrected_social_circle`

Procedure:

```
contact_usage = {}
verified_fragments = []
trace = []

for each hidden_fact HF in hidden_facts:
    for attempt in 1..3:
        # Spawn GENERATOR subagent (preferred: opus-class model)
        generator_prompt = step4_app_log_synthesizer/prompt.txt
          with {{INSERT_HIDDEN_FACT_JSON_HERE}} = HF
          with {{INSERT_CORRECTED_SOCIAL_CIRCLE_JSON_HERE}} = corrected_social_circle
          with {{LOG_START_DATE}}, {{LOG_END_DATE}} = log window
          with {{INSERT_CONTACT_USAGE_COUNTS_JSON_HERE}} = contact_usage
        fragments = extract_json( spawn_subagent(generator_prompt) )

        # Spawn VERIFIER subagent (preferred: different model family — sonnet/haiku)
        verifier_prompt = step4_app_log_synthesizer/verification_prompt.txt
          with {{INSERT_FRAGMENTS_JSON_HERE}} = fragments  # NO hidden fact
          with {{PERSONA_NAME}}, {{PERSONA_AGE}}, etc. = persona identity
        verification = extract_json( spawn_subagent(verifier_prompt) )

        if verification.verdict == "RECOVERED" AND
           label_tokens(verification.candidate_label) overlaps
           label_tokens(HF.ground_truth_label) >= 40%:
            verified_fragments += fragments tagged with HF.fact_id
            update contact_usage with fragments.contacts_used
            record in trace: passed=true
            break
    else:
        record in trace: passed=false (unrecoverable)

# MERGE subagent (preferred: opus-class model)
merge_prompt = step4_app_log_synthesizer/merge_prompt.txt
  with verified_fragments, hidden_facts, corrected_social_circle,
       persona identity, log window, news events (optional, can be [])
app_logs = extract_json( spawn_subagent(merge_prompt) )

write app_logs to step4_app_log_synthesizer/data_samples/output/{persona}_app_logs.json
write trace to step4_app_log_synthesizer/data_samples/output/{persona}_app_logs_trace.json
```

#### Step 5 — Test Case Synthesis

```
# GENERATOR subagent
generator_prompt = step5_testcases_synthesis/prompt.txt
  with corrected_extracted_profile, app_logs, corrected_social_circle
test_cases = extract_json( spawn_subagent(generator_prompt) )

# VERIFIER subagent
verifier_prompt = step5_testcases_synthesis/verification_prompt.txt
  with test_cases, corrected_extracted_profile, app_logs
verification = extract_json( spawn_subagent(verifier_prompt) )

if verification.coverage_deficit is not empty or rejected count > 0:
    re-run generator with feedback, up to 3 iterations

write test_cases, verification to step5_testcases_synthesis/data_samples/output/
```

#### Step 6 — Benchmark Runner (two-pass)

```
# PASS 1: strip ground_truth fields from test_cases, flatten app_logs to raw text
pass1_prompt = step6_benchmark_runner/prompt.txt (PASS 1 section)
  with raw app log text (no theme_plan, no hidden_facts, no metadata)
  with test_cases WITHOUT ground_truth / expected_evidence / source_hidden_fact_ids
pass1_answers = extract_json( spawn_subagent(pass1_prompt) )

# PASS 2: different subagent, different context — scores answers
pass2_prompt = step6_benchmark_runner/prompt.txt (PASS 2 section)
  with full test_cases (including ground_truth)
  with pass1_answers
final_results = extract_json( spawn_subagent(pass2_prompt) )

write pass1_answers, final_results to step6_benchmark_runner/data_samples/output/
```

### Parallelism

Because each subagent is isolated, runs for different personas can be parallelized:

- Batch all 5 personas' Step 4 runs simultaneously (one orchestrator per persona, each spawning its own chain of subagents).
- Within one persona's Step 4, batch 5-10 hidden facts per generator subagent if you want to reduce orchestration overhead (the per-fact prompt is written for one fact at a time by default, so batching requires a small prompt tweak).

### Reproducibility

Given the same inputs and the same model selection, the orchestrated subagent runs produce the same outputs as Backends A and B — the prompts are the program, the subagents are the execution substrate. The repo is deterministic at the prompt level; execution backend is swappable.

---

## Which backend is used for each file on disk

The generated files in `data_samples/output/` directories do not record which backend produced them — outputs are expected to be equivalent. If you want provenance tracking, add a `backend` field to the output metadata in your local run; it is not required by any downstream step.
