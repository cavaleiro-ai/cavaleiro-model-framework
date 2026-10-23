# Changelog

All notable changes to Cavaleiro Model Framework are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use [Semantic Versioning](https://semver.org/) for the Cavaleiro Model Framework release line.

The Python package that Cavaleiro Model Framework builds is still `transformers`, and its version string (`transformers.__version__`) is the development version of the original project it was taken from. This file lists only what Cavaleiro Model Framework changes.

## [Unreleased]

### Changed

- Project identity: the READMEs, community files, issue forms, pull request template, `setup.py` author, Copilot instructions, conda recipe metadata, and engineering notes refer only to Cavaleiro AI, and the README keeps a one-line credit to the original project under **License**. `CITATION.cff` was removed. The reviewer-routing file `.github/scripts/codeowners_for_review_action` assigns every path to `@cavaleiro-ai`.
- Inherited documentation (`docs/source/`, `docs/original-i18n/`, `docs/README.md`, `docs/TRANSLATING.md`): the product is named Cavaleiro Model Framework. Paper titles and abstracts, titles of external articles, historical notes about when a model was added to the original project, model contribution credits, and version requirements keep their original wording. Support and contact links (forum, Discord, maintainer handles, team email, Slack) point to this repository's issues or were removed, as were the commercial-support banners.

## Cavaleiro Model Framework 1.0.0 — 2026-10-17

First release of Cavaleiro Model Framework, based on commit `d4804f9d76b8c071b993d813e9e87067b780cbee` of the original project (`main`, package version `5.21.0.dev0`). Every fix below has a regression test that fails on that commit and passes here; the evidence is in [docs/cavaleiro/ENGINEERING.md](docs/cavaleiro/ENGINEERING.md).

### Added

- Cavaleiro CI workflow ([`.github/workflows/cavaleiro-ci.yml`](.github/workflows/cavaleiro-ci.yml)): ruff lint and format with the same version and targets as `utils/checkers.py`, plus the tests of the modules changed in this fork on CPU, with a read-only `GITHUB_TOKEN` only.
- [README.pt-BR.md](README.pt-BR.md), a complete Portuguese guide ([docs/cavaleiro/GUIA-PT-BR.md](docs/cavaleiro/GUIA-PT-BR.md)), and engineering notes with the architecture map, baseline, and findings ([docs/cavaleiro/ENGINEERING.md](docs/cavaleiro/ENGINEERING.md)).
- `NOTICE` file with the attribution notices of the original project and the list of Cavaleiro AI modifications (Apache-2.0, Section 4).
- `.github/CODEOWNERS` assigning the repository to `@cavaleiro-ai`.

### Changed

- Rebranded the project as **Cavaleiro Model Framework**: README, repository links, issue forms, pull request template, and the `url` in `setup.py` now point to `cavaleiro-ai/cavaleiro-model-framework`. The package name (`transformers`), import paths, CLI command (`transformers`), environment variables, cache locations, and file formats are unchanged.
- The translated READMEs of the original project moved from `i18n/` to [`docs/original-i18n/`](docs/original-i18n/README.md), unchanged apart from their link back to the main README.
- Issue forms rewritten for this project (bug report, improvement, and `config.yml` with private security reporting). The original forms for new models, documentation translations, and `pytorch-transformers` migration were removed, together with their lists of maintainers to mention.
- `CONTRIBUTING.md`, `SECURITY.md`, and `CODE_OF_CONDUCT.md` adapted for this fork (private advisories at `cavaleiro-ai/cavaleiro-model-framework`, enforcement contact through the Cavaleiro AI GitHub profile), keeping the original guidance.

### Fixed

- **Generation:** `generate(encoder_outputs=..., encoder_repetition_penalty=...)` without `input_ids` crashed with `RuntimeError: index -100 is out of bounds`, and `encoder_no_repeat_ngram_size` was silently computed on placeholder ids. Both encoder-side processors are now skipped with the existing warning when no encoder token ids are available (original issue #49695). `src/transformers/generation/utils.py`.
- **Pipelines:** `pipeline(..., model_kwargs={"code_revision": ...})` raised `TypeError: got multiple values for keyword argument 'code_revision'`, and a top-level `code_revision` was not forwarded to the model load. It is now accepted from either place (both at once raise a `ValueError`, like `device_map` and `dtype`) and forwarded to the model load (original issue #49272). `src/transformers/pipelines/__init__.py`.
- **Pipelines:** `pipeline()` wrote `dtype` and `device_map` into the caller's `model_kwargs` dictionary, so reusing the same dictionary for a second pipeline with an explicit `dtype` raised `ValueError: You cannot use both pipeline(... dtype=..., model_kwargs={"dtype":...})`. The function now works on a copy. `src/transformers/pipelines/__init__.py`.
- **Hub, all platforms:** `save_pretrained(..., push_to_hub=True)` derived the default repository name with `str.split(os.path.sep)`. A `pathlib.Path` raised `AttributeError` in configurations, feature extractors, image and video processors, and processors; a trailing separator produced an empty repository name; and on Windows a path written with `/` became the whole path. The name is now `os.path.basename(os.path.normpath(save_directory))` in the ten `save_pretrained` implementations (configuration, generation configuration, model, tokenizers, feature extractor, image processor, video processor, processor, Mistral tokenizer).
- **Hub, Windows:** files uploaded from a sub-folder by `push_to_hub` (for example `additional_chat_templates/`) were sent with a backslash in `path_in_repo`. Hub paths now always use `/`. `src/transformers/utils/hub.py`.
- **Windows:** `HfArgumentParser` read `.args` files with the locale encoding (cp1252 on many Windows systems), turning UTF-8 values such as `Ação` into `AÃ§Ã£o`. They are now read as UTF-8, like the JSON and YAML loaders. `src/transformers/hf_argparser.py`.
- **CLI:** `transformers add-new-model-like` ran its helper scripts with whatever `python` was first on `PATH` instead of the interpreter running the command. With a virtual environment that is not activated (common on Windows), the modular conversion ran in the wrong environment, failed silently, and no `modeling_*.py` was generated. The scripts now run with `sys.executable`; the two existing tests that failed for this reason pass. `src/transformers/cli/add_new_model_like.py`.
- **Auto classes:** `AutoVideoProcessor.register(config, processor, True)` (the legacy positional `exist_ok`) raised `TypeError: 'bool' object is not iterable` after `video_processor_classes` was inserted before `exist_ok`. A positional boolean is now read as `exist_ok` (original issue #49469). `src/transformers/models/auto/video_processing_auto.py`.

### Security

- Fork safety: 57 GitHub workflows of the original project (and its CI troubleshooting notes) moved to [`.github/workflows-disabled/`](.github/workflows-disabled/README.md). They publish to PyPI, Anaconda, Docker Hub, or the Hugging Face Hub, use secrets other than `GITHUB_TOKEN`, call the original project's reusable workflows and dashboards, run on self-hosted GPU runners, or automate the original project's triage (reviewer assignment, stale issues, anti-spam labels, bots). The README there explains how to re-enable each one with your own credentials and names. Kept active: `trufflehog.yml` (secret scanning, read-only) and `add-model-like.yml` (inert in this repository). The CircleCI configuration is documented as not to be connected as is.
- Private vulnerability reporting through [GitHub Security Advisories](https://github.com/cavaleiro-ai/cavaleiro-model-framework/security/advisories/new) (see [SECURITY.md](SECURITY.md)).

[Unreleased]: https://github.com/cavaleiro-ai/cavaleiro-model-framework/compare/v1.0.0...HEAD
