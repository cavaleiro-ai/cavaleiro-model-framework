## Commands

- `make style` — ruff format + lint (required for code style checks).
- `make typing` — ty type checker and model structure rules.
- `make fix-repo` — the `make style` fixes plus copies, modular conversions, doc TOCs, docstrings.
- `make check-repo` — `make typing` plus consistency checks.
- `RUN_SLOW=1 pytest ...` — many tests are marked slow and skipped in CI.

Run `make style` (or `make fix-repo`) as the last step before opening a PR.

## Local agent setup

Hosted review agents read this file via the root `AGENTS.md` / `CLAUDE.md` symlinks. Local agents wire their own assets: `make codex` (→ `.agents/`), `make claude` (→ `.claude/`).

## Before opening a PR

- Coordinate on the matching issue first. Do not open a PR for someone else's issue without explicit approval from its author or a maintainer in the thread; if approval is unclear, ask instead of drafting.
- Check for overlapping work, and do not open a second PR for a fix already covered:

  ```bash
  gh issue view <issue_number> --repo cavaleiro-ai/cavaleiro-model-framework --comments
  gh pr list --repo cavaleiro-ai/cavaleiro-model-framework --state open --search "<issue_number> in:body"
  gh pr list --repo cavaleiro-ai/cavaleiro-model-framework --state open --search "<short area keywords>"
  ```

  If your approach is materially different, say why a second PR is needed in the issue.
- This repository is Cavaleiro Model Framework, a derivative of `huggingface/transformers`. Bugs that also affect upstream may already be tracked there: check `--repo huggingface/transformers` too, and follow upstream's own contribution policy if you contribute there (it asks first-time contributors not to submit agent-written PRs or issues, with a risk of being blocked).
- No one-off PRs for tiny edits (a single typo, an isolated lint fix). Mechanical cleanups are fine, but not as a first contribution.
- Every source file changed for the first time in this fork needs a `# Modified by Cavaleiro AI (<year>): <what changed>` line below its license header, and every change a line under **Unreleased** in `CHANGELOG.md` — see `CONTRIBUTING.md` and the PR template.

## Copies and modular models

Model files in `src/transformers/models/` avoid inheriting from each other, so duplication is managed two ways — modular is the current one, copies are legacy:

1. `# Copied from ...` marks a copied class or function. `make fix-repo` re-syncs it, so editing inside such a block is reverted — edit the source it copies from, or deliberately break the link. **Do not add new `# Copied from` statements**; write a modular file instead.
2. A `modular_<name>.py` **may** inherit from other models; `make fix-repo` generates the standalone `modeling_*.py` and friends from it. Never edit a generated file when a modular one exists. Guide: [modular_transformers.md](../docs/source/en/modular_transformers.md).

Two modular traps:

- **Other models inherit your modular file.** `modular_deepseek_vl.py` also generates `modeling_deepseek_vl_hybrid.py`. Run `make fix-repo` and check everything it rewrote; hand-editing only the generated file you had in mind leaves the rest stale and `Check repository consistency` red.
- **`attr = AttributeError()` deletes an inherited attribute** — it is an instruction to the converter, not a bug or placeholder. `raise AttributeError("...")` in a method body does the same for a method. Substituting a "real" value silently changes behaviour (`_no_split_modules = AttributeError()` → `[]` changes how the model may be sharded). See [Removing attributes](../docs/source/en/modular_transformers.md#removing-attributes) and [Deleting unused methods](../docs/source/en/modular_transformers.md#deleting-unused-methods).
