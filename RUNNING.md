# Running PersonaBench

Three execution backends. Pick the one that matches how you bill, what tools you have installed, and whether you want to walk away or iterate interactively. All three produce equivalent outputs given the same seed and model selection.

| | Backend A | Backend B | Backend C |
|---|---|---|---|
| Billing | API per-token | Vendor subscription | Vendor subscription |
| Auth | API key in env var | Vendor CLI login | Coding tool login |
| Orchestration | Python script calls a provider SDK | Python script shells out to a vendor CLI | Coding tool's orchestrator agent spawns subagents |
| Interactive? | No | No | Yes |
| Scheduled / CI? | Yes | Yes | No |
| Best for | CI, headless reproduction by a stranger with an API key, scaling to 100 personas overnight | Headless scripted runs against a subscription | Iterative prompt work, debugging a step, all-files-in-one-directory workflow |
| Reference impl in this repo | `Top_level_generator.py` | `openclaw_pipeline.py` | Runbook in this doc |

A reader who only wants to reproduce the published numbers should use Backend A or B. A reader who wants to iterate on prompts or debug Step 4's reverse-inferability gate should use Backend C.

---

## Prerequisites (all backends)

1. Python 3.10 or newer.
2. Repo cloned locally. All commands below assume the repo root is the working directory:
   ```bash
   git clone https://github.com/TruePersona/PersonaBench.git
   cd PersonaBench
   ```
3. Provider SDK installed only if you want Backend A:
   ```bash
   pip install anthropic
   ```
   (Replace with `openai`, `google-genai`, etc. if you wire Backend A against a non-Anthropic provider; see Adapting to a different provider in the Backend A section.)
4. About 2 GB of free disk for the 5-persona output set.

The 5 sample seeds ship in `step1_seed/data_samples/output/`. You do not need to run Step 1 unless you are adding a new persona, since Step 1 just pulls a record from the NVIDIA Nemotron-Personas-USA HuggingFace dataset.

---

## Backend A: Direct API

You hold an API key for an LLM provider. Each LLM call inside the pipeline is a direct SDK invocation. You pay per input + output token. Output: JSON files on disk.

### Setup

1. Install the provider SDK:
   ```bash
   pip install anthropic
   ```
2. Set the API key:
   ```bash
   # macOS / Linux
   export ANTHROPIC_API_KEY="sk-ant-..."

   # Windows PowerShell
   $env:ANTHROPIC_API_KEY = "sk-ant-..."
   ```
3. Smoke test the key:
   ```bash
   python -c "import anthropic; r=anthropic.Anthropic().messages.create(model='claude-sonnet-4-6', max_tokens=10, messages=[{'role':'user','content':'reply with the word ok'}]); print(r.content[0].text)"
   ```
   Expected output: a short reply containing `ok`. An auth error means the key is wrong.

### One persona, end to end (Steps 2 through 6)

Run time: roughly 3 to 5 hours, dominated by Step 4. Outputs land in each step's `data_samples/output/` directory.

```bash
python Top_level_generator.py \
  --seed step1_seed/data_samples/output/julio_simmons_seed.json \
  --start 2 --stop 6 \
  --model claude-opus-4-6 \
  --verifier-model claude-sonnet-4-6
```

Resulting files:

| Step | Output |
|------|--------|
| 2 | `step2_interview/data_samples/output/julio_simmons_interview.json` and `_verification.json` |
| 3 | `step3_social_circle/data_samples/output/julio_simmons_social_circle.json` and `_verification.json` |
| 4 | `step4_app_log_synthesizer/data_samples/output/julio_simmons_app_logs.json` and `_app_logs_trace.json` |
| 5 | `step5_testcases_synthesis/data_samples/output/julio_simmons_test_cases.json` and `_verification.json` |
| 6 | `step6_benchmark_runner/data_samples/output/julio_simmons_pass1_answers.json` and `_benchmark_results.json` |

### All five sample personas

Bash:
```bash
for SEED in step1_seed/data_samples/output/*_seed.json; do
  python Top_level_generator.py --seed "$SEED" --start 2 --stop 6
done
```

PowerShell:
```powershell
Get-ChildItem step1_seed/data_samples/output/*_seed.json | ForEach-Object {
  python Top_level_generator.py --seed $_.FullName --start 2 --stop 6
}
```

### Re-running a single step

Useful when you have iterated on one step's prompt and only want to regenerate that step's output. Step 6 is the most common case:

```bash
python step6_benchmark_runner/generator.py \
  --app-logs   step4_app_log_synthesizer/data_samples/output/julio_simmons_app_logs.json \
  --test-cases step5_testcases_synthesis/data_samples/output/julio_simmons_test_cases.json \
  --output     step6_benchmark_runner/data_samples/output/ \
  --model-pass1 claude-opus-4-6 \
  --model-pass2 claude-sonnet-4-6
```

`--model-pass1` and `--model-pass2` should be different models so the scorer is independent of the answerer.

### Adapting to a different provider

The reference implementation uses the Anthropic SDK. To run against another provider:

1. Replace the SDK calls in each step's `generator.py`. The pattern is roughly `client.messages.create(model=..., max_tokens=..., messages=[{"role":"user","content":prompt}])`. Map this to your provider's chat-completion API.
2. Update model names. Step prompts are vendor-neutral and never reference a model by name; the model name is only set in the runner script's argparse defaults.
3. The `extract_json()` helper expects the model output to contain a JSON object somewhere in the response (fenced or not). Most providers comply. If yours always wraps responses in a known envelope, adjust `extract_json()` accordingly.

### Cost notes

Step 4 dominates the API spend. Roughly 100 to 175 hidden facts per persona, up to 3 attempts per fact, 2 LLM calls per attempt (generator plus reverse-inferability verifier). Total: several hundred API calls per persona for Step 4 alone, plus a handful each for Steps 2, 3, 5, and 6. Step 6 Pass 1 sends the full app log (up to ~220K tokens for the largest persona) once per test case. A frontier model with a large context window keeps the log unsummarized.

---

## Backend B: Vendor CLI on subscription

You have a vendor subscription and the vendor's command-line tool installed. Each LLM call shells out to the CLI, billing your subscription instead of an API key. Same outputs as Backend A on disk. Same scriptable single-command end-to-end run, no human in the loop.

The reference implementation is `openclaw_pipeline.py`, written against the `claude` CLI. The naming reflects an internal session-isolation pattern (delete the CLI session directory between Pass 1 calls in Step 6) and not a specific vendor.

### Setup

1. Install the vendor CLI per the vendor's installation instructions. After install, confirm the binary is on PATH:
   ```bash
   claude --version
   ```
   Expected output: a version string. `command not found` means the binary is not on PATH.

2. Sign in. This typically pairs the CLI with your subscription via a browser flow:
   ```bash
   claude login
   ```

3. Smoke test a non-interactive call:
   ```bash
   echo "reply with the word ok" | claude -p
   ```
   Expected output: a short reply containing `ok`.

4. Windows only: the `claude` CLI is shipped as `claude.cmd`. Either rename to `claude` on PATH, or set the env var so `openclaw_pipeline.py` finds it:
   ```powershell
   $env:CLAUDE_CMD = "claude.cmd"
   ```

5. Optional: pin the CLI's default model so model selection is deterministic across CLI default drifts:
   ```bash
   claude config set model claude-opus-4-6
   ```
   (You can also pass `--model` per call from the pipeline; see the next section.)

### One persona, end to end

```bash
python openclaw_pipeline.py \
  --seed step1_seed/data_samples/output/julio_simmons_seed.json \
  --start 2 --stop 6 \
  --model claude-opus-4-6
```

`--model` is plumbed through to every `claude -p` invocation so the run is reproducible regardless of any CLI-side default change.

### All five personas

```bash
python openclaw_pipeline.py --all --start 2 --stop 6 --model claude-opus-4-6
```

The `--all` flag iterates every `*_seed.json` in `step1_seed/data_samples/output/` and walks each through Steps 2 to 6. Kickoff once, come back hours later.

### Step 6 session isolation (the OpenClaw pattern)

Step 6 is two-pass (answer, then score). When run via a CLI that persists session state between calls, the runner must delete that session state between every Pass 1 test case so prior answers cannot leak into the current one. The pipeline passes `--openclaw` to `step6_benchmark_runner/generator.py`, which deletes the CLI session directory between every test case. This is automatic for Backend B; nothing to configure.

### Adapting to a different vendor's CLI

`openclaw_pipeline.py` assumes:

- the CLI accepts a prompt on stdin with a `-p` flag,
- the CLI returns the model's response on stdout,
- the CLI exits non-zero on failure.

If your vendor's CLI looks different (different flag, different stream, different exit semantics), replace `call_claude()` in `openclaw_pipeline.py` with a wrapper that matches your CLI's invocation shape. The rest of the pipeline is vendor-agnostic.

### Per-step invocation

```bash
python step6_benchmark_runner/generator.py \
  --app-logs   step4_app_log_synthesizer/data_samples/output/julio_simmons_app_logs.json \
  --test-cases step5_testcases_synthesis/data_samples/output/julio_simmons_test_cases.json \
  --output     step6_benchmark_runner/data_samples/output/ \
  --openclaw \
  --claude-cmd claude
```

`--claude-cmd` lets you point the runner at any vendor CLI binary that satisfies the three assumptions above.

---

## Backend C: Coding agent on subscription

You have a coding agent open on the repo (Claude Code, Cursor, Antigravity, Codex, Continue, Cline, Devin, or any equivalent that exposes a subagent / Task tool). You drive the pipeline by talking to an orchestrator agent that reads inputs from disk, spawns one subagent per LLM call with that step's prompt filled in, and writes outputs back to disk. Billing routes through whatever subscription powers the coding tool.

This is the right backend when you want to iterate: tweak a prompt, ask the orchestrator to rerun just Step 4 on one persona, inspect the trace, tweak again. It is also the only backend where every LLM call can be routed to a different model in the same run with a single sign-in.

### Why this backend exists

Two structural properties that Backends A and B have to approximate:

- **Inherent context isolation.** Every subagent starts with a blank context. There is no session state to delete between Pass 1 calls. Step 6's session-delete hack is unnecessary here.
- **Per-call model selection without juggling auth.** The orchestrator can spawn the Step 4 fragment generator on a frontier model and the reverse-inferability verifier on a smaller model in the same run, all under one coding-tool subscription, no API key juggling.

### Prerequisites

- A coding agent that exposes a subagent tool (the ability to spawn an isolated agent with a custom prompt).
- The agent has read and write access to the repo at the same paths used in Backends A and B.
- A small JSON-extraction helper on the orchestrator side (regex plus `json.loads` is enough).

### Runbook (one persona, Steps 4 through 6)

The orchestrator pseudocode below is vendor-neutral. Translate it into your agent's subagent API. Every `spawn_subagent(prompt)` is one isolated LLM call with the prompt as the only context.

#### Step 4: App log synthesis

Inputs (read from disk):
- `step2_interview/data_samples/output/{persona}_verification.json`, field `corrected_extracted_profile.hidden_facts`
- `step3_social_circle/data_samples/output/{persona}_social_circle_verification.json`, field `corrected_social_circle`

```
contact_usage = {}
verified_fragments = []
trace = []

for HF in hidden_facts:
    for attempt in 1..3:
        # GENERATOR subagent (preferred: frontier-class model)
        gen_prompt = step4_app_log_synthesizer/prompt.txt
            with {{INSERT_HIDDEN_FACT_JSON_HERE}} = HF
            with {{INSERT_CORRECTED_SOCIAL_CIRCLE_JSON_HERE}} = corrected_social_circle
            with {{LOG_START_DATE}}, {{LOG_END_DATE}} = log window
            with {{INSERT_CONTACT_USAGE_COUNTS_JSON_HERE}} = contact_usage
        fragments = extract_json( spawn_subagent(gen_prompt) )

        # VERIFIER subagent (preferred: a smaller or different model from the generator)
        ver_prompt = step4_app_log_synthesizer/verification_prompt.txt
            with {{INSERT_FRAGMENTS_JSON_HERE}} = fragments        # NO hidden fact
            with {{PERSONA_NAME}}, {{PERSONA_AGE}}, {{PERSONA_OCCUPATION}}, {{PERSONA_LOCATION}}
        verification = extract_json( spawn_subagent(ver_prompt) )

        if verification.verdict == "RECOVERED" and label_overlap >= 40%:
            verified_fragments += tag(fragments, HF.fact_id)
            update contact_usage from fragments
            break

# MERGE subagent (preferred: frontier-class model)
merge_prompt = step4_app_log_synthesizer/merge_prompt.txt
    with verified_fragments, hidden_facts, corrected_social_circle, persona, window
app_logs = extract_json( spawn_subagent(merge_prompt) )

write app_logs   -> step4_app_log_synthesizer/data_samples/output/{persona}_app_logs.json
write trace      -> step4_app_log_synthesizer/data_samples/output/{persona}_app_logs_trace.json
```

#### Step 5: Test case synthesis

```
gen_prompt = step5_testcases_synthesis/prompt.txt
    with corrected_extracted_profile, app_logs, corrected_social_circle
test_cases = extract_json( spawn_subagent(gen_prompt) )

ver_prompt = step5_testcases_synthesis/verification_prompt.txt
    with test_cases, corrected_extracted_profile, app_logs
verification = extract_json( spawn_subagent(ver_prompt) )

if verification.coverage_deficit OR rejected_count > 0:
    re-run gen_prompt with feedback (up to 3 iterations)

write test_cases, verification -> step5_testcases_synthesis/data_samples/output/
```

#### Step 6: Benchmark runner (two-pass)

```
# PASS 1: ONLY raw text logs + questions. STRIP ground_truth from test cases first.
raw_text = strip_to_raw_logs(app_logs)        # contact / date / messages + calendar fields ONLY
                                              # source_fact_ids and theme_plan are NEVER in this text
stripped = strip_ground_truth(test_cases)

pass1_prompt = step6_benchmark_runner/prompt.txt   (PASS 1 section)
    with raw_text, stripped
pass1_answers = extract_json( spawn_subagent(pass1_prompt) )

# PASS 2: different subagent, different context. Sees ground truth + Pass 1 answers.
pass2_prompt = step6_benchmark_runner/prompt.txt   (PASS 2 section)
    with full test_cases (including ground_truth), pass1_answers
final_results = extract_json( spawn_subagent(pass2_prompt) )

write pass1_answers, final_results -> step6_benchmark_runner/data_samples/output/
```

**Critical for Backend C:** never hand the raw `app_logs.json` to a Pass 1 subagent. Always run it through `strip_to_raw_logs()` (in `step6_benchmark_runner/generator.py`) first. The JSON contains traceability fields that the evaluator must not see. Backends A and B handle this automatically; Backend C orchestrators have to do it explicitly.

### Parallelism on Backend C

Subagents are isolated, so personas run in parallel:

- One orchestrator instance per persona, each spawning its own chain. Five personas equals five orchestrators.
- Within one persona's Step 4, batching 5 to 10 hidden facts per generator subagent reduces orchestration overhead. The default prompt is single-fact; batching needs a small prompt edit.

For scaling to 100 personas, prefer Backend B over Backend C unless you are willing to keep an interactive session open the entire run. Backend C's interactive nature is its strength for iteration and its weakness for headless overnight runs.

---

## Choosing between B and C

Both are subscription-based and produce equivalent output files. They differ on operational mode:

- **Pick B when** you want to fire one command and walk away (overnight runs, CI, scheduled cron, scaling to 100 personas).
- **Pick C when** you want to iterate on prompts, debug a single step's failure, surface a Step 4 trace and propose a prompt edit, or run with all files visible in one directory and an agent driving the steps.

Same outputs on disk. Same data quality. Different ergonomics.

---

## Which backend produced which file

The shipped `data_samples/output/` files do not record their backend. Outputs are expected to be equivalent given the same model selection and seed. If you need provenance, add a `backend` field to your local run's metadata. Nothing downstream reads it.

## Reproducing the published numbers

To reproduce the canonical 5-persona benchmark numbers in `step6_benchmark_runner/RESULTS.md`:

- Backend A: API key set, then run the all-five loop above. Wall time depends on rate limits.
- Backend B: vendor CLI logged in, then `python openclaw_pipeline.py --all --start 2 --stop 6 --model claude-opus-4-6`. Wall time roughly 5 to 10 hours unattended.
- Backend C: open the repo in your coding tool, ask the orchestrator to walk Steps 2 through 6 for each seed in `step1_seed/data_samples/output/`. Wall time depends on how interactively you want to drive it.

Output diff against the shipped files should be cosmetic only (timestamps, message text wording within the prompt's design constraints). Numerical accuracy figures should match within 1 to 2 percentage points on a frontier model and per-judge.
