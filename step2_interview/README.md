# Step 2 — Interview and Extracted Profile

## Purpose

Produce a full qualitative life-history interview and a structured extracted profile for a synthetic persona. The interview uses the 109-question American Voices Project protocol (Park et al., 2024). The extracted profile is a structured summary of everything important from the interview, organized so downstream steps can consume it.

## Process

Step 2 is a two-phase pipeline run inside a single step folder:

1. **Generation** — `prompt.txt` takes the seed data and produces the interview transcript plus the draft extracted profile (preferences across 18 subcategories plus four free-text expert summaries).

2. **Verification** — `verification_prompt.txt` takes the seed, transcript, and draft profile and runs a four-direction consistency check (seed vs. interview, interview vs. profile, profile internal consistency, completeness). It corrects gaps and inconsistencies, preserves the expert summaries verbatim, and generates the deduplicated `hidden_facts` registry.

The `generator.py` script orchestrates both phases and regenerates via an iterative refinement loop if verification fails.

## Inputs

- Seed JSON (from `../step1_seed/data_samples/output/`)
- Optional: additional inputs in `data_samples/input/`

## Outputs

- `data_samples/output/{name}_interview.json` — interview transcript and draft extracted profile
- `data_samples/output/{name}_verification.json` — verification report, corrected extracted profile, and hidden_facts registry

## Extracted profile structure

```
{
  "preferences": { 18 subcategories with summary, details, and Q## evidence },
  "summaries":   { demographer, behavioral_economist, political_scientist, psychologist — free-text }
}
```

After verification, the corrected profile is accompanied by a `hidden_facts` registry of 60-100 atomic facts enumerated and deduplicated from the corrected preferences.

## The 18 preference subcategories

health_and_wellness, food_and_dining, travel, social_and_relationships, home_and_lifestyle, entertainment, hobbies_and_recreation, work_and_career, arts_and_culture, sports_and_fitness, nature_and_outdoors, technology, community_and_civic, music, fashion_and_appearance, cultural_and_linguistic, life_events_and_transitions, media_and_information_diet.

## The hidden_facts registry

`hidden_facts` is the canonical answer key used by all downstream steps. It is
generated in the verification step by a three-pass process:

1. **Direct enumeration** of atomic facts from each of the 18 preference
   subcategories (`source: direct`).
2. **Reprojection** of claims asserted in the four expert summaries into the
   subcategory taxonomy. Claims already covered by a Pass 1 fact are skipped;
   novel claims become new facts (`source: reprojected`).
3. **Deduplication** — facts expressing the same underlying claim are merged;
   `duplicate_of` and `also_supported_by` record the trace.

Each entry contains:

- `fact_id`, `source` (`direct` | `reprojected`), `source_subcategory`
- `reprojected_from` (which expert summary the claim came from, if reprojected)
- `claim`, `ground_truth_label`
- `behavioral_evidence_from_interview` (Q## quotes)
- `duplicate_of`, `also_supported_by`

This design ensures downstream steps (app log synthesis, test case generation,
benchmarking) can operate on `hidden_facts` alone without consulting the
preferences or summaries.

## Running

```bash
python generator.py \
  --seed data_samples/input/{name}_seed.json \
  --output data_samples/output/
```

## Notes

- The interview questionnaire (109 questions) is embedded directly in `prompt.txt`. No separate questionnaire file is required.
- The expert summaries are generated in Step 2 and are NOT regenerated during verification. Verification only corrects the preferences and produces the hidden_facts registry.
