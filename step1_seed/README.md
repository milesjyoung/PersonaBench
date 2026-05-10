# Step 1 — Seed Persona

## Purpose

Fetch a base demographic record for a synthetic persona from NVIDIA's Nemotron-Personas-USA dataset. The seed is the starting point for the interview step — a flat demographic sketch that subsequent steps enrich into a full life history.

## Input

A participant ID from the Nemotron-Personas-USA dataset (HuggingFace: `nvidia/Nemotron-Personas-USA`).

## Output

`data_samples/output/{name}_seed.json` — a JSON file containing:

- Participant ID, name, age, sex, marital status, education, occupation, location
- Narrative paragraphs across professional, sports, arts, travel, culinary, and cultural domains
- Skills, hobbies, and career goals

## Running

```bash
python generator.py --uuid <id> --output data_samples/output/
```

The generator pulls the record from the HuggingFace dataset and writes the seed JSON.

## Notes

- Seed records are the ONLY point in the pipeline where NVIDIA data is used.
- Downstream steps consume only the outputs of the preceding step, not the seed directly (except Step 2, which uses the seed to anchor the interview).
