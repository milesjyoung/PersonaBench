# DCO Enforcement Setup

This doc is for the repository maintainer. It explains how to activate DCO sign-off as a required status check on the `main` branch.

## What's in place

- `.github/workflows/dco.yml` — a GitHub Action that runs on every pull request and fails if any commit in the PR is missing a `Signed-off-by:` line.
- `DCO.txt` — the Developer Certificate of Origin 1.1 text.
- `CONTRIBUTING.md` — contributor-facing instructions: how to sign off, how to configure VS Code, what the DCO means, AI-assisted contribution expectations.

## One-time GitHub configuration

After this directory is pushed to the `TruePersona/PersonaBench` repository, the maintainer must configure the branch ruleset so the DCO workflow is required before merge.

### Steps (GitHub web UI)

1. Open the repository on GitHub.
2. Navigate to **Settings → Rules → Rulesets → New ruleset → New branch ruleset**.
3. **Ruleset name:** `main-protection`.
4. **Enforcement status:** Active.
5. **Target branches:** Include default branch (`main`) or add `main` by name.
6. Under **Rules**, enable:
   - **Require a pull request before merging**
     - Required approvals: 1 (adjust per team policy)
     - Dismiss stale approvals when new commits are pushed
   - **Require status checks to pass**
     - Select **`DCO / Signed-off-by required on every commit`** (appears after the workflow has run at least once)
     - Check "Require branches to be up to date before merging"
   - **Block force pushes**
   - **Require linear history** (optional but recommended)
7. **Bypass list:** leave empty. The maintainer is not exempt.
8. Save.

### Verifying the setup

- Open a test PR that includes a commit WITHOUT a `Signed-off-by:` line. The DCO workflow must fail, and the merge button must be disabled.
- Re-sign the commit (`git rebase HEAD~1 --signoff && git push --force-with-lease`). The workflow must pass and the merge button becomes available.

## Contributor quick reference

Every commit needs a sign-off. The easiest way:

```bash
git commit -s -m "Your message"
```

Or configure VS Code once and forget:

```json
{
  "git.alwaysSignOff": true
}
```

If a contributor forgets:

```bash
# Last commit only
git commit --amend -s --no-edit

# All commits in the current branch ahead of main
git rebase main --signoff
git push --force-with-lease
```

## Why DCO, not CLA

DCO is a lightweight certification on the commit itself — no external tooling, no agreement to sign outside of the commit log. It's the same mechanism used by the Linux kernel, Kubernetes, and vLLM. PersonaBench follows that convention so contributors from those communities have zero onboarding friction.
