# Running a Qwen Pass 1 via Ollama

This is the recipe for using a local quantized Qwen model as the Pass 1 answering model in PersonaBench. Pass 2 (scoring) still runs through the Claude judge so the small model never grades its own answers.

The default target is `qwen3.5:9b` at 256K context. That fits a 16GB consumer GPU and lets the answering model see the full meaningful log without any summarization.

## Hardware reality

| GPU VRAM | Recommended Qwen tag | Context |
|---|---|---|
| 16 GB | `qwen3.5:9b` (6.6 GB weights) | 256K |
| 24 GB | `qwen3.5:9b` or `qwen3.6:27b` (17 GB) | 256K |
| 48 GB+ | `qwen3.5:35b` (24 GB) or `qwen3.6:35b` | 256K |

Anything that requires offloading layers to CPU is a non-starter for this workload. Step 6 runs ~100 inferences per persona at near-context-limit prompts and the per-call latency on CPU offload makes a single persona take days.

## Setup (one-time)

1. **Install Ollama.** Download from <https://ollama.com/download> and run the installer for your OS. After install, confirm:
   ```bash
   ollama --version
   ```
   Expected: a version string.

2. **Start the Ollama server** (it usually auto-starts on install):
   ```bash
   ollama serve
   ```
   Leave it running. The server listens on `http://127.0.0.1:11434`.

3. **Pull the model.** This downloads ~6.6 GB:
   ```bash
   ollama pull qwen3.5:9b
   ```

4. **Confirm GPU placement.** A second terminal:
   ```bash
   ollama ps
   ```
   The output should show `qwen3.5:9b` with `100% GPU` after a single test prompt. If it shows any percentage on CPU, the model is offloading and the run will be too slow to be useful. Drop to a smaller tag (`qwen3.5:4b`) or extend VRAM.

5. **Smoke-test the model:**
   ```bash
   ollama run qwen3.5:9b "reply with the word ok"
   ```
   Expected: `ok` (or similar). Type `/bye` to exit.

## Running PersonaBench Pass 1 against Qwen

The `_scripts/run_small_model.py` harness handles the chunking, JSON parsing, and Pass 2 hand-off. Default Pass 2 judge is the Claude CLI with Sonnet so install Claude CLI per [RUNNING.md](../RUNNING.md) Backend B before running.

One persona, full meaningful context (262K window, no summarization):

```bash
python -m step6_benchmark_runner._scripts.run_small_model \
  --persona julio_simmons \
  --provider ollama \
  --model qwen3.5:9b \
  --truncation full \
  --batch-size 5 \
  --judge-model sonnet
```

`--truncation full` renders the entire app log (messenger meaningful + filler + calendar) at 262K. Use this when the model's context window comfortably fits the log; the 5 sample personas top out at ~220K tokens of raw log so 262K leaves room for the prompt template and test cases.

If a future persona's log exceeds 262K, fall back to `--truncation meaningful_preserved` which keeps all meaningful sessions and calendar events and only samples filler:

```bash
python -m step6_benchmark_runner._scripts.run_small_model \
  --persona alicia_gonzalez \
  --provider ollama --model qwen3.5:9b \
  --truncation meaningful_preserved \
  --log-budget 250000 \
  --batch-size 5
```

The output JSON's `metadata.preservation` block records the meaningful-session count before vs after, proving no meaningful evidence was dropped:

```json
"preservation": {
  "meaningful_sessions_in": 134,
  "meaningful_sessions_kept": 134,
  "calendar_events_in": 47,
  "calendar_events_kept": 47,
  "filler_sessions_total": 2445,
  "filler_sessions_sampled": 1820,
  "filler_sessions_dropped": 625,
  "total_tokens_estimated": 248000,
  "target_tokens": 250000,
  "filler_sample_seed": 42
}
```

If `meaningful_sessions_kept != meaningful_sessions_in`, the runner raises an `AssertionError` rather than silently shipping a degraded run.

## Tuning the Ollama call

The runner sets these options when calling Ollama. You should not need to override them, but the knobs are documented here for reproducibility:

| Option | Value | Why |
|---|---|---|
| `num_ctx` | 262144 | Match Qwen3.5's native context. Lower values cause silent log truncation. |
| `num_gpu` | 99 | Force every layer to GPU. Avoids the CPU-offload trap. |
| `num_predict` | 32000 | Headroom for batched answers + thinking-mode scratchpad. |
| `think` | false | Suppresses Qwen's `<think>` block so the JSON answer comes out cleanly. |
| `temperature` | 0 | Deterministic answering. |
| `stream` | false | One JSON object per response, no streaming parsing. |

## Troubleshooting

**Symptom:** `ollama ps` shows the model on CPU.
**Cause:** Either no CUDA driver, or the chosen `num_ctx` made the KV cache exceed VRAM.
**Fix:** Try `num_ctx=131072` first to confirm GPU placement works at all. If yes, the 262K KV cache is the issue and the GPU is too small for this model at full context. Drop to `qwen3.5:4b` or use a 24GB+ box.

**Symptom:** Pass 1 outputs are empty strings or non-JSON.
**Cause:** The thinking-mode scratchpad is consuming the output budget.
**Fix:** Confirm the runner is sending `"think": false`. Bump `num_predict` if a single batch's answers (5 cases × ~1.5K each) plus formatting need more room.

**Symptom:** Connection refused on `http://127.0.0.1:11434`.
**Cause:** Ollama server is not running.
**Fix:** Run `ollama serve` in a separate terminal and re-run.

## Output paths

```
step6_benchmark_runner/data_samples/output_small_models/
  julio_simmons_pass1_ollama.json            # Qwen Pass 1 answers + preservation metadata
  julio_simmons_benchmark_results_ollama.json # Sonnet-judged Pass 2 verdict
```

Compare against the canonical Anthropic Pass 1 in `data_samples/output/` to score the open-weights tier against the frontier baseline on identical test cases.
