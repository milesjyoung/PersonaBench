# Step 5 — Test Case Synthesis

## Purpose

Generate evaluation test cases that probe two questions about a frontier LLM:

1. Can it infer the persona's identity and life patterns from raw app logs alone?
2. Does it behave as a useful, safe personal agent for the persona given those inferred traits?

The test cases are the questions and user requests used to drive Step 6 (benchmark runner).

## Process

Two phases in one step folder:

1. **Generation** — `prompt.txt` takes the corrected extracted profile, app logs, and corrected social circle; produces a typed, coverage-checked set of test cases.
2. **Verification** — `verification_prompt.txt` takes the generated test cases plus the profile and app logs; checks ground-truth sourcing, non-duplication, and Type 5 risk validity. Outputs a `corrected_test_cases` structure with REJECTED cases removed and FLAGGED cases annotated with revision notes.

The `generator.py` script orchestrates both phases and regenerates via an iterative refinement loop if verification reports rejections or coverage gaps.

## Inputs

- `data_samples/input/{persona}_verification.json` — corrected extracted profile with `hidden_facts` registry (from `../step2_interview/data_samples/output/`)
- `data_samples/input/{persona}_app_logs.json` — merged messenger + calendar logs with its own `hidden_facts` array (from `../step4_app_log_synthesizer/data_samples/output/`)
- `data_samples/input/{persona}_social_circle_verification.json` — corrected social circle (from `../step3_social_circle/data_samples/output/`)

## Outputs

- `data_samples/output/{persona}_test_cases.json` — typed test cases with ground truth labels, evidence anchors, and coverage metadata
- `data_samples/output/{persona}_test_cases_verification.json` — verification report and corrected test cases

## Test case types

| Type | Intent | Framing | Proportion |
|---|---|---|---|
| 1 — Simple fact-check | Retrieve one fact from one source | third-party | 7% |
| 2 — Cross-log fact-check | Connect info across 2+ sources | third-party | 11% |
| 3 — Dynamic tracking | Identify a temporal shift | third-party | 7% |
| 4 — Reasoning | Synthesize 3+ sources | third-party | 60% |
| 5 — Agent-behavior | Serve the persona safely as an agent | first-person user prompt | 15% |

Type 5 has four subtypes:
- **5a Safety** — request intersects a risk-bearing hidden fact (the LLM must surface the risk).
- **5b Conflict** — two hidden facts point opposite directions (the LLM must resolve).
- **5c Accommodation** — benign request requiring persona-specific tailoring.
- **5d Clarification** — ambiguous request requiring a persona-informed clarifying question.

The final test case for each persona is a **Type 5a capstone** built on that persona's highest `risk_surface_score` hidden fact.

## Ground truth sourcing

Every `ground_truth` field is a literal copy of one (or more, joined with ` | `) `ground_truth_label` strings from the corrected profile's `hidden_facts` registry. No paraphrasing. Preferences and summaries are not consulted — they exist in the profile for human reference only.

## Coverage rules

1. Every preference subcategory present in the profile must be covered by at least one Type 4 case.
2. Every `category == "temporal"` hidden fact must be covered by at least one Type 3 or Type 2 case.
3. All four Type 5 subtypes (5a, 5b, 5c, 5d) must be present. The capstone counts toward 5a.
4. No two test cases may cite overlapping evidence sets of more than 50%.
5. Every test case must cite at least one real `fact_id` from the profile.

## Running

```bash
python generator.py \
  --profile     data_samples/input/{persona}_verification.json \
  --app-logs    data_samples/input/{persona}_app_logs.json \
  --social-circle data_samples/input/{persona}_social_circle_verification.json \
  --output      data_samples/output/
```

## Notes

- Answerability is NOT verified here. It is owned by the Step 4 app-log synthesis stage, which runs a per-fragment reverse-inferability gate on every hidden fact before merging into the full log. This prompt trusts that gate.
- The verification step never invents new test cases. If removals cause a coverage rule to fail, the verification output records a `coverage_deficit` and the generator.py refinement loop re-runs generation.
