# Review Policy

This document defines how PersonaBench changes are reviewed. It applies to every pull request that touches a step's prompt, generator, sample data, or downstream benchmark output.

## Bundle review

Prompt, data, and benchmark results are reviewed and approved as a single bundle. A prompt is not reviewed in isolation. A data file is not reviewed in isolation. A benchmark result is not reviewed in isolation. The three are reviewed together because each depends on the others.

A bundle is the set of files that move together for one change:

| Step | Prompt | Data produced | Downstream signal |
|---|---|---|---|
| Step 2 interview | `step2_interview/prompt.txt`, `verification_prompt.txt` | `step2_interview/data_samples/output/{persona}_interview.json`, `_verification.json` | Hidden facts registry sanity, expert summary coherence |
| Step 3 social circle | `step3_social_circle/prompt.txt`, `verification_prompt.txt` | `step3_social_circle/data_samples/output/{persona}_social_circle*.json` | 2-direction verification verdicts |
| Step 4 app log synthesis | `step4_app_log_synthesizer/prompt.txt`, `verification_prompt.txt`, `merge_prompt.txt` | `step4_app_log_synthesizer/data_samples/output/{persona}_app_logs.json`, `_app_logs_trace.json` | Reverse-inferability gate pass rate, contact balance, filler ratio |
| Step 5 test cases | `step5_testcases_synthesis/prompt.txt`, `verification_prompt.txt` | `step5_testcases_synthesis/data_samples/output/{persona}_test_cases.json` | Coverage report, type distribution, no overlapping evidence sets |
| Step 6 benchmark runner | `step6_benchmark_runner/prompt.txt` | `step6_benchmark_runner/data_samples/output/{persona}_benchmark_results.json`, `_pass1_answers.json` | Per-type accuracy, Type 5 dimension averages, capstone verdict |

A reviewer cannot approve the prompt without seeing the data it produced. A reviewer cannot approve the data without seeing the benchmark numbers downstream. If the prompt looks fine but the data shows a regression, the bundle is rejected. If the data looks fine but the benchmark numbers spike or collapse implausibly, the bundle is rejected.

Cascading rejection is the rule, not the exception. When any one of {prompt, data, benchmark results} fails review, the whole bundle is rejected and reworked together. There is no partial approval.

## Test cases are decoupled from app log synthesis

Test cases are anchored to the `hidden_facts` registry produced by Step 2, not to the app logs produced by Step 4. If Step 4 changes (for example, dynamic events are added or filler density is tuned), the test cases do not automatically regenerate.

This decoupling exists for two reasons:

1. The test case bank is reusable across different log generations. Iterating on Step 4 prompts does not invalidate Step 5 outputs.
2. A test case asserts ground truth from `hidden_facts` directly. The app logs are how the evaluator recovers that ground truth, not the source of it.

When Step 4 ships a meaningful change (new affordances, dynamic state, denser anchoring), the Step 5 prompt may need a deliberate update to match. Such an update is a separate bundle reviewed under the same rule. Test cases never regenerate as a side effect.

## What a reviewable PR looks like

A PR that changes any prompt, generator, or sample data must include:

1. The prompt or code change.
2. The regenerated sample data for at least one persona (preferably all five), so the reviewer can see what the change produces.
3. The downstream benchmark result for that data, or a clear statement that the result is forthcoming and the bundle approval is pending.
4. A PR description that names the motivation, the bundle's before/after deltas, and any flagged anomalies.
5. DCO sign-off on every commit (`git commit -s`). No sign-off, no review.

PRs that ship code without data, or data without the benchmark signal, are returned for completion before review begins.

## Rejection cascade in practice

If Step 4 is changed and Step 6 results swing by more than expected, the reviewer flags the bundle. Cause analysis happens before any partial merge:

- Was the new data more explicit (anchors easier to retrieve)?
- Did the Pass 1 invocation change in a way that could leak ground truth?
- Was the Pass 2 prompt or judge model changed?
- Were the test cases regenerated against the new logs (rule violation)?

Until those questions are answered, the bundle stays out of `main`.

## See also

- [CONTRIBUTING.md](CONTRIBUTING.md) for the DCO sign-off requirement and AI-assisted contribution rules.
- [step4_app_log_synthesizer/README.md](step4_app_log_synthesizer/README.md) for the per-fact verify-before-merge architecture that backs the bundle's claim of recoverability.
- [step6_benchmark_runner/README.md](step6_benchmark_runner/README.md) for the two-pass design that prevents ground truth leakage during scoring.
