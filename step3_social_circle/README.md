# Step 3 — Social Circle

## Purpose

Generate the five people with whom the subject persona most frequently exchanges personal text messages. These five people become the conversational partners in the downstream app-log synthesis step.

## Method

Persona-to-Persona generation (Ge et al., 2024, *PersonaHub*). New personas are derived through interpersonal relationships anchored in the subject's documented life (interview transcript and corrected extracted profile).

## Process

Step 3 is a two-phase pipeline run inside a single step folder:

1. **Generation** — `prompt.txt` takes the corrected extracted profile and the interview transcript and produces the draft social circle (five members with demographics, personality, shared activities, recent text topics, and Q## evidence citations).

2. **Verification** — `verification_prompt.txt` runs a two-direction consistency check (social circle vs. interview, social circle vs. extracted profile), corrects inconsistencies, fills gaps, and produces the `corrected_social_circle`.

The `generator.py` script orchestrates both phases and regenerates via an iterative refinement loop if verification fails.

## Inputs

- Corrected extracted profile (from `../step2_interview/data_samples/output/`)
- Interview transcript (from `../step2_interview/data_samples/output/`)

## Outputs

- `data_samples/output/{name}_social_circle.json` — draft social circle
- `data_samples/output/{name}_social_circle_verification.json` — verification report plus corrected social circle

## Social circle scope

The five members are the subject's closest PERSONAL messaging contacts. Work colleagues who would realistically communicate only on a company-owned channel are excluded. No rigid quota on relationship type — the five are whichever contacts the interview evidence most strongly supports.

## Running

```bash
python generator.py \
  --profile ../step2_interview/data_samples/output/{name}_verification.json \
  --transcript ../step2_interview/data_samples/output/{name}_interview.json \
  --output data_samples/output/
```
