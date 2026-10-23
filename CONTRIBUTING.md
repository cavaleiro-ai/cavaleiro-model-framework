<!---
Copyright 2020 The HuggingFace Team. All rights reserved.
Modified by Cavaleiro AI (2026): adapted as the Cavaleiro Model Framework contributing guide.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contributing to Cavaleiro Model Framework

Thanks for helping improve Cavaleiro Model Framework. This project focuses on **verified fixes to core, model-agnostic code** (generation, pipelines, configuration and serialization, tokenizer and processor base classes, Hub and file handling, the CLI) and on Windows support. It stays a drop-in replacement for `transformers`.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities privately, as described in the [security policy](SECURITY.md).

## Contents

1. [What belongs here](#what-belongs-here)
2. [Ground rules](#ground-rules)
3. [Set up](#set-up)
   - [Windows](#windows)
4. [Run the tests and checks](#run-the-tests-and-checks)
5. [Opening issues](#opening-issues)
6. [Pull requests](#pull-requests)
7. [Docs](#docs)
8. [Agentic contributions](#agentic-contributions)

## What belongs here

- **In scope:** bugs in Cavaleiro Model Framework changes, robustness and Windows fixes in shared code, CI, and documentation (including the Portuguese docs).
- **Out of scope:** new model architectures, model-specific fixes, and documentation translations.
- When a change touches models or their docs, the [model addition](docs/source/en/add_new_model.md), [modular](docs/source/en/modular_transformers.md), [auto-docstring](docs/source/en/auto_docstring.md), [testing](docs/source/en/testing.md), and [modeling rules](docs/source/en/modeling_rules.md) guides apply.

## Ground rules

1. **Compatibility is the contract.** Keep the `transformers` package name, import paths, public APIs, environment variables, file names, and checkpoint formats compatible with the original project. A deliberate behavior change needs a reason, a test, and a CHANGELOG entry.
2. **Prove the bug, then fix it.** Write a regression test that fails without your change and passes with it, and say how you ran it. Never weaken, skip, or delete an existing test or check to get a green result.
3. **Minimal diffs.** Match the surrounding code's style, naming, and comment density. No drive-by reformatting, no unrelated refactors, no new dependencies without a strong reason.
4. **Generated code is changed at its source.** When a `modular_<name>.py` exists, edit it and regenerate the `modeling_*.py` files; edit the source of a `# Copied from` block, not the copy (see [`.ai/AGENTS.md`](.ai/AGENTS.md)).
5. **License notices.** The project is Apache-2.0. The first time you change a source file in this fork, add one line below its license header: `# Modified by Cavaleiro AI (<year>): <what changed>` (Section 4(b) of the License). Keep every existing copyright line.

### How we think about contributions

- **Clear diagnosis of bugs.** Track down the exact cause of a problem instead of adding a workaround; finding the first commit where it appeared (for example with [git bisect](https://git-scm.com/docs/git-bisect)) is valuable.
- **Minimize the diff.** If your PR can be a 1-line fix, make it a 1-line fix. Add comments only when they are really necessary.
- **Reproduce the problem.** Many reports are caused by the reporter's environment or misdiagnose the cause. A small reproducer script that you ran yourself is worth more than a theory.
- **Compare against other models and modules.** The repository is large and many components do similar things; a fix that says "same approach as (other code)", with a link, is easier to trust.
- **Avoid busywork.** Small style changes and theoretical issues that never happen in practice are not worth a PR unless they can be exploited by an attacker.

## Set up

1. Fork [Cavaleiro Model Framework](https://github.com/cavaleiro-ai/cavaleiro-model-framework) with the **Fork** button on GitHub, then clone your fork.

   ```bash
   git clone git@github.com:<your GitHub handle>/cavaleiro-model-framework.git
   cd cavaleiro-model-framework
   ```

2. Add this repository as the `upstream` remote, so you can sync your `main` before opening a pull request.

   ```bash
   git remote add upstream https://github.com/cavaleiro-ai/cavaleiro-model-framework.git
   git checkout main && git pull upstream main && git push origin main
   git switch -c a-descriptive-name-for-my-branch
   ```

3. Create a project-local virtual environment and install the library in editable mode. `[testing]` covers most work; `[dev]` adds every optional backend; `[quality]` is enough for docs-only changes.

   ```bash
   uv venv --python 3.12 .venv           # or: python -m venv .venv
   source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
   uv pip install -e ".[testing,torch,vision,sentencepiece]"
   ```

   On Linux, install a CPU-only PyTorch first if you do not need CUDA: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`.

### Windows

- Configure git to keep `LF` line endings in this repository, unless you use [WSL](https://learn.microsoft.com/en-us/windows/wsl/):

  ```bash
  git config core.autocrlf input
  ```

- `make` is not needed: every Makefile target is a Python command you can run directly (see the next section). If you want `make`, one way is [MSYS2](https://www.msys2.org/): run `pacman -S make` in its shell and add `C:\msys64\usr\bin` to your `PATH`.
- Enable Developer Mode so the Hugging Face Hub cache can use symbolic links; otherwise some cache tests fail with `WinError 1314`.
- Tests that use `torch.compile` need the Visual Studio C++ compiler (`cl.exe`) on `PATH`; without it they fail with `InvalidCxxCompiler`. They are not related to your change.
- Run Python through the virtual environment's interpreter (`.venv\Scripts\python.exe`), or activate it first.

## Run the tests and checks

The full suite is very large. Run the tests of the modules you touch, plus `tests/utils` for shared code, and set `HF_HUB_DISABLE_TELEMETRY=1`. Tests use tiny models (for example `hf-internal-testing/tiny-random-*`) or configs built in code; do not add tests that download large checkpoints outside `@slow`.

```bash
export HF_HUB_DISABLE_TELEMETRY=1        # PowerShell: $env:HF_HUB_DISABLE_TELEMETRY = "1"
python -m pytest -n 4 --dist=loadfile tests/utils
python -m pytest tests/utils/test_hub_utils.py tests/pipelines/test_pipelines_common.py -k "your_test"
python utils/tests_fetcher.py             # lists the tests that cover your changed files
```

Style and consistency checks (the Makefile targets `style`, `check-code-quality`, and `check-repository-consistency` call the same script):

```bash
ruff check <changed files>
ruff format --check <changed files>
python utils/checkers.py ruff_check,ruff_format,init_isort,inits,copies,docstrings,dummies --keep-going
python utils/checkers.py all --fix --keep-going   # the `make fix-repo` equivalent: regenerates copies and modular files
```

The [Cavaleiro CI](.github/workflows/cavaleiro-ci.yml) workflow runs ruff and the tests of the modules this fork changed on every pull request.

## Opening issues

Use the [issue forms](.github/ISSUE_TEMPLATE/) to get started, and search existing issues first. The issue should be a bug in the library, not in your own code; if you are unsure, say so and include a minimal reproduction.

Include:

* Your *OS type and version*, *Python*, and *PyTorch* versions, and the Cavaleiro Model Framework commit or release. `transformers env` prints most of this.
* A short, self-contained code snippet that reproduces the bug, preferably with a tiny model.
* The *full* traceback if an exception is raised.
* Whether the bug also happens with the `transformers` release on PyPI.

For a feature request, describe the motivation, the feature in as much detail as possible, and a code snippet showing how it would be used.

## Pull requests

- Keep each pull request focused on one problem, and link the issue.
- Fill in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md), ticking only the checks you actually ran and pasting relevant output.
- Say what you could not test (GPUs, multiple devices, Hub-dependent paths, other operating systems).
- Add a line under **Unreleased** in the [changelog](CHANGELOG.md).

## Docs

Improvements to the docs, like missing content or unclear explanations, are welcome. Fork-specific documentation lives in the READMEs and [`docs/cavaleiro/`](docs/cavaleiro/); keep [README.md](README.md) and [README.pt-BR.md](README.pt-BR.md) aligned in the same change. The library documentation lives in [`docs/source/`](docs/source/); follow the [docs README](docs/README.md) when you change it. For API reference generated from source files, use the [`@auto_docstring`](docs/source/en/auto_docstring.md) decorator when it applies.

## Agentic contributions

AI-assisted contributions are welcome when they are coordinated, scoped, and verified:

- Do not submit "pure agent" PRs. The human submitter is responsible for reviewing all changed lines, validating behavior end to end, and running the relevant tests.
- If AI tools were used, say so in the PR description and include the test commands and results.
- Avoid one-off busywork PRs (a single typo, an isolated style cleanup); bundle mechanical cleanups into a clear, systematic scope.
- Agents should read [`.ai/AGENTS.md`](.ai/AGENTS.md) for the repository's commands and the copies/modular rules.

## License

By contributing, you agree that your contributions are licensed under the [Apache License, Version 2.0](LICENSE) of this repository.
