# Contributing to PersonaBench

Thank you for your interest in contributing. This document covers repository layout, prompt conventions, the Developer Certificate of Origin, and expectations for AI-assisted contributions.

## Getting Started

1. Clone the repository.
2. Review the top-level `README.md` for the pipeline overview.
3. Each step folder (`step*`) contains its own `README.md` with step-specific documentation, prompts, and a generator script.

## Repository Layout

Every step folder follows the same structure:

- `prompt.txt` — generation prompt for the step
- `verification_prompt.txt` — verification prompt for the step (where applicable)
- `generator.py` — orchestration script that runs generation, verification, and iterative refinement
- `README.md` — step documentation
- `data_samples/input/` — sample input data (when the step requires input beyond the previous step's output)
- `data_samples/output/` — sample output data

## Prompt Conventions

- Prompts are model-agnostic. Do not reference specific model names or providers inside prompt bodies.
- Prompts do not reference step numbers or pipeline positions. When another step's output is needed, refer to it by path (e.g., `step2_interview/data_samples/output/`).
- Every generation prompt has a matching verification prompt that runs at the end of the step and produces a corrected canonical output.
- Schema fields use `verdict` (consistent | inconsistent) and `gap` (true | false) as independent concepts. Resolution fields are populated only when an action is taken.

## Adding a New Step

1. Create a new folder under `` following the naming pattern `stepN_<name>`.
2. Add `prompt.txt`, `verification_prompt.txt` (if verification applies), `generator.py`, and `README.md`.
3. Add `data_samples/input/` and `data_samples/output/`.
4. Update the top-level `README.md` pipeline table to reference the new step.
5. Update `Top_level_generator.py` to orchestrate the new step into the pipeline.

## Regenerating Personas

Run the pipeline end-to-end for a single persona:

```bash
python Top_level_generator.py --participant_id <id>
```

Run a single step for an existing persona:

```bash
python step2_interview/generator.py \
  --seed step1_seed/data_samples/output/{name}_seed.json \
  --output step2_interview/data_samples/output/
```

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting Python code.

```bash
pip install ruff
ruff check .
ruff check --fix .
ruff format .
```

The Ruff configuration lives in `pyproject.toml`. Ensure your code passes all checks before submitting a pull request.

## Developer Certificate of Origin (DCO)

When contributing changes to this project, you must agree to the [Developer Certificate of Origin](DCO.txt). Commits must include a `Signed-off-by:` header which certifies agreement with the terms of the DCO.

### How to Sign Off

Add a `Signed-off-by` line to every commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

The easiest way is the `-s` flag:

```bash
git commit -s -m "Your commit message"
```

### Configuring VS Code for Automatic Sign-Off

1. Open VS Code Settings (`Ctrl+,` or `Cmd+,`).
2. Search for `git.alwaysSignOff`.
3. Check the box to enable it.

Or add this to `settings.json`:

```json
{
  "git.alwaysSignOff": true
}
```

## AI Assisted Contributions

Before making an AI-assisted contribution, you must:

- **Review thoroughly.** Do not submit "pure agent" commits. The human submitter is responsible for reviewing all changed lines, validating behavior end-to-end, and running relevant tests.
- **Ensure significance.** Avoid one-off "busywork" commits (single typo, isolated style cleanup, one mutable default fix, etc.). Bundle mechanical cleanups into a clear, systematic scope.
- **Verify rights.** Ensure, to the best of your knowledge, that your contribution does not knowingly violate any third-party rights or licenses.

## What the DCO Means

By signing off on a commit, you certify that:

- The contribution was created in whole or in part by you, and you have the right to submit it under the project's open-source license; **or**
- The contribution is based upon previous work that, to the best of your knowledge, is covered under an appropriate open-source license and you have the right to submit that work with modifications; **or**
- The contribution was provided directly to you by some other person who certified (a) or (b), and you have not modified it.
- You ensure, to the best of your knowledge, that your contribution does not knowingly violate any third-party rights or licenses.

By contributing, you confirm that you have the right to submit your contribution under the project's open-source license, regardless of whether AI tools were used. Use of AI tools does not change your responsibilities under the DCO.

See the full [DCO text](DCO.txt) for details.

## Reporting Issues

- Use [GitHub Issues](https://github.com/TruePersona/PersonaBench/issues) to report bugs or request features.
- Check existing issues before opening a new one to avoid duplicates.

## Submitting Pull Requests

1. Fork the repository and create a new branch from `main`.
2. Make your changes. Keep them focused; separate refactors from feature work.
3. Write clear commit messages. Include the DCO sign-off (`git commit -s`).
4. Open a pull request against `main`. Fill out the PR template.
5. For prompt modifications, include a before/after example of the prompt's output to demonstrate the effect.
6. Respond to review feedback promptly.

PRs that touch any prompt, generator, or sample data are reviewed under the bundle policy: prompt, data, and benchmark results approve or reject together. See [REVIEW.md](REVIEW.md) for what a reviewable bundle looks like and how cascading rejection works.

## License

By contributing to PersonaBench, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).
