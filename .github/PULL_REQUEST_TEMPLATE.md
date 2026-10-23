## Summary

<!-- What changes, in one or two sentences. -->

## Why

<!-- The problem this solves. Link the issue: "Closes #123". -->

## What changed

<!-- Files and public behavior touched. Call out anything that changes a public API, a file format, or a default. -->

-

## How it was verified

<!-- Paste the relevant output. Tick only what you actually ran. -->

- [ ] A regression test that fails without the change and passes with it (name it below)
- [ ] `python -m pytest <the test files for the modules you touched>`
- [ ] `python -m pytest -n 4 --dist=loadfile tests/utils` (when you touched `src/transformers/utils/` or other shared code)
- [ ] `ruff check <changed files>` and `ruff format --check <changed files>`
- [ ] `python utils/checkers.py <relevant checkers>` (for example `copies,modular_conversion,inits,docstrings` when you touched models, `__init__` files, or docstrings)
- [ ] Tested on Windows (say which version) if the change touches paths, encodings, or subprocesses

## Compatibility

- [ ] The `transformers` package name, import paths, public APIs, and file formats stay compatible with the original project (or the break is explained above).
- [ ] Every source file changed for the first time in this fork has a `Modified by Cavaleiro AI (<year>): <what changed>` comment below its license header (Apache-2.0, Section 4(b)).
- [ ] A line was added under **Unreleased** in [CHANGELOG.md](../CHANGELOG.md).

## Risks and follow-ups

<!-- What could break, what was not tested (GPU, multi-device, Hub-dependent paths), and what remains pending. -->

## Checklist

- [ ] No secrets, Hub tokens, private URLs, or personal settings added.
- [ ] Generated files (`modeling_*.py` produced from a `modular_*.py`, `# Copied from` blocks) were changed through their source, not by hand.
- [ ] Documentation updated if behavior changed.
