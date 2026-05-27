# Step 4 — App Log Synthesizer

## Purpose

Generate the messenger conversations and calendar events that the benchmark's evaluator LLM will see. The raw app log is the ONLY context the evaluator receives in Step 6 — so the encoding of every hidden fact into implicit behavioral signals is the structural heart of PersonaBench.

## Architecture -- cluster verify-before-merge

Fragment generation and verification are decoupled from log merging:

```
cluster selected hidden_facts into related groups:
    generate 4-5 shared implicit fragments per cluster
    run reverse-inferability gate on those fragments alone
    if the gate recovers every fact in the cluster -> keep
    if not -> regenerate the cluster fragments (up to N attempts)

once all selected clusters pass:
    assemble verified fragments deterministically
    add mundane filler at about 2.5:1 filler:meaningful tokens
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
- `data_samples/output/{persona}_app_logs_trace.json` — per-cluster verification trace showing attempts made and whether every selected fact passed the gate
- `data_samples/output/{persona}_verified_fragments.json` — verified fragment cache used by `--resume` and `--retry-failed`
- `data_samples/output/{persona}_verified_decoys.json` — optional pre-verified decoy pool; the merge selects hard messenger decoys first when `--verified-decoys` and `--decoy-count` are provided

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
  --profile        data_samples/input/{persona}_verification.json \
  --social-circle  data_samples/input/{persona}_social_circle_verification.json \
  --output         data_samples/output/ \
  --model          claude-opus-4-7 \
  --verifier-model claude-sonnet-4-6

# Optional: add real news for surprise weaving
python generator.py \
  --profile        ... \
  --social-circle  ... \
  --news-events    data_samples/input/news_events.json \
  --model          claude-opus-4-7 \
  --verifier-model claude-sonnet-4-6
```

Advanced: use different models for generation vs verification (recommended — keeps the gate independent):

```bash
python generator.py \
  --profile ... --social-circle ... \
  --model          claude-opus-4-6 \
  --verifier-model claude-sonnet-4-6
```

## Failed-cluster retry

If a cluster fails the gate after `--per-cluster-max-attempts` tries (default
3), the trace file records `passed: false` for that cluster. A professor-ready
run requires every selected fact to pass. Use `--resume --retry-failed` to
reuse passed clusters and regenerate only failed clusters.

## Log window and filler ratio

| Parameter | Default | Where set |
|---|---|---|
| Log window | March 1-31, 2026 (30 days) | `--log-start` / `--log-end` flags |
| Filler ratio | 2.5 filler tokens per 1 meaningful token | `merge_prompt.txt` / `--filler-ratio` |

Filler is mundane logistics noise that the evaluator must sift through to find implicit behavioral signals. Without filler, the benchmark degenerates into a reading comprehension test. Changing either parameter requires regenerating Step 4 and downstream steps (5-6).

## Verifying a successful run

Check the trace file. Every hidden fact should show `"passed": true`:

```bash
python -c "import json; t=json.load(open('step4_app_log_synthesizer/data_samples/output/julio_simmons_app_logs_trace.json')); print(f'{sum(1 for x in t if x[\"passed\"])}/{len(t)} passed')"
```

## Notes on reproducibility

This step is clone-and-run. Given the same `hidden_facts` and `corrected_social_circle` inputs (both themselves reproducible from Step 2 and Step 3), the pipeline produces equivalent app logs. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the repo's reproducibility philosophy.
