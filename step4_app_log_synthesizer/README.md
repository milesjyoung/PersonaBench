# Step 4 — App Log Synthesizer

## Purpose

Generate the messenger conversations and calendar events that the benchmark's evaluator LLM will see. The raw app log is the ONLY context the evaluator receives in Step 6 — so the encoding of every hidden fact into implicit behavioral signals is the structural heart of PersonaBench.

## Architecture — per-fact verify-before-merge

Fragment generation and verification are decoupled from log merging:

```
for each hidden_fact:
    generate 2-3 implicit fragments
    run reverse-inferability gate on those fragments alone
    if gate recovers the ground_truth_label → keep
    if not → regenerate the fragments (up to N attempts)

once all facts' fragments pass:
    merge fragments + filler (3:1 filler:meaningful ratio) + surprises
    produce the final app_logs.json
```

The gate is run against a **cold, independent verifier model** (different from the generation model) that does not see the hidden fact — only the fragments and a persona sketch (name, age, occupation, location). If the verifier cannot recover the fact, the evaluator in Step 6 also cannot, so the fragments are rejected.

This is the structural reason the benchmark is defensible: every fact in the answer key has already been proven implicitly recoverable at the fragment level.

## Behaviors, never labels

Fragments encode hidden facts through **behaviors and context**, never through clinical or categorical words. If a hidden fact is "Generalized Anxiety Disorder + sertraline", the fragments never contain the words "anxiety", "SSRI", "antidepressant", "depression", etc. — they contain pharmacy pickup texts, biweekly Thursday 3pm video calls with "Dr. P", refill alerts, offhand supportive lines from family. The inference is recoverable but not stated.

## Inputs

- `data_samples/input/{persona}_verification.json` — corrected extracted profile with `hidden_facts` registry (from `../step2_interview/data_samples/output/`)
- `data_samples/input/{persona}_social_circle_verification.json` — corrected social circle (from `../step3_social_circle/data_samples/output/`)
- Optional: `data_samples/input/news_events.json` — real news events within the log window for surprise weaving

## Outputs

- `data_samples/output/{persona}_app_logs.json` — the final merged log with messenger sessions, calendar events, hidden_facts registry (mirrors Step 2 schema + adds verification + embedding_strategy blocks), cross_app_index, token_stats
- `data_samples/output/{persona}_app_logs_trace.json` — per-fact verification trace showing attempts made and whether each fact passed the gate

## Rules baked into `prompt.txt`

The prompt encodes concrete rules addressing common failure modes in open-ended log synthesis:

- **Concrete details preserved** — dollar amounts, medication names, store names, dates, named people outside the circle (no "picked up my prescription" without a named pharmacy).
- **Contact balancing** — fragment generation reads the running contact usage count and prefers under-used contacts. Final merge enforces no contact below 10% or above 35% of meaningful sessions.
- **Realistic timestamps** — 60/25/10/5 distribution across 1-5min / 5-15min / 15-45min / 1-24hr gaps. No uniform cadence.
- **Filler quality** — mundane logistics (3-15 words), not philosophical essays. Each contact has its own opener vocabulary.
- **Temporal distribution** — fragments and filler spread across the full log window, not bunched in the last 3 days.
- **Voice differentiation** — each contact's voice matches their `personality_mini_profile` and `communication_style` from the social circle.

## Running

```bash
python generator.py \
  --profile       data_samples/input/{persona}_verification.json \
  --social-circle data_samples/input/{persona}_social_circle_verification.json \
  --output        data_samples/output/

# Optional: add real news for surprise weaving
python generator.py \
  --profile       ... \
  --social-circle ... \
  --news-events   data_samples/input/news_events.json
```

Advanced: use different models for generation vs verification (recommended — keeps the gate independent):

```bash
python generator.py \
  --profile ... --social-circle ... \
  --model          claude-opus-4-6 \
  --verifier-model claude-sonnet-4-6
```

## Per-fact fallback

If a hidden fact fails the gate after `--per-fact-max-attempts` tries (default 3), the fragments are still merged into the log but the trace file records `passed: false` for that fact. Step 5's test case generator can choose to skip unrecoverable facts, and Step 6's final accuracy should be interpreted in that light.

## Log window and filler ratio

| Parameter | Default | Where set |
|---|---|---|
| Log window | March 1-31, 2026 (30 days) | `--log-start` / `--log-end` flags |
| Filler ratio | 2.5 filler messages per 1 meaningful message | `merge_prompt.txt` |

Filler is mundane logistics noise that the evaluator must sift through to find implicit behavioral signals. Without filler, the benchmark degenerates into a reading comprehension test. Changing either parameter requires regenerating Step 4 and downstream steps (5-6).

## Verifying a successful run

Check the trace file. Every hidden fact should show `"passed": true`:

```bash
python -c "import json; t=json.load(open('step4_app_log_synthesizer/data_samples/output/julio_simmons_app_logs_trace.json')); print(f'{sum(1 for x in t if x[\"passed\"])}/{len(t)} passed')"
```

## Notes on reproducibility

This step is clone-and-run. Given the same `hidden_facts` and `corrected_social_circle` inputs (both themselves reproducible from Step 2 and Step 3), the pipeline produces equivalent app logs. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the repo's reproducibility philosophy.
