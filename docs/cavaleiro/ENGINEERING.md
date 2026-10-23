# Cavaleiro Model Framework — engineering notes

This document records how Cavaleiro Model Framework 1.0.0 was derived from the original project ("upstream" below): the architecture map, the test baseline, every finding and how it was verified, and what was deliberately left out. The Portuguese user guide is [GUIA-PT-BR.md](GUIA-PT-BR.md).

| | |
| --- | --- |
| Upstream | The original project, branch `main` |
| Upstream commit | `d4804f9d76b8c071b993d813e9e87067b780cbee` (2026-10-17, "Fix test selection for virtual module imports (#49690)") |
| Package version | `transformers` `5.21.0.dev0` (unchanged) |
| License | Apache-2.0 ([LICENSE](../../LICENSE) unchanged, [NOTICE](../../NOTICE) added) |
| Release | Cavaleiro Model Framework 1.0.0 — 2026-10-17 |

## 1. Architecture map

### Components

| Component | Location | Responsibility |
| --- | --- | --- |
| Lazy package root | `src/transformers/__init__.py`, `utils/import_utils.py` | Declares the public API as an import structure and imports submodules lazily on first attribute access; optional backends (torch, vision, audio, sentencepiece, ...) are probed by `is_*_available()` helpers. |
| Configuration | `configuration_utils.py` (`PreTrainedConfig`), `models/*/configuration_*.py` | Hyper-parameters of an architecture; JSON (de)serialization (`to_dict`, `to_diff_dict`, `save_pretrained`, `from_pretrained`); RoPE parameter standardization (`modeling_rope_utils.py`). |
| Models | `modeling_utils.py` (`PreTrainedModel`), `models/<name>/modeling_*.py` (533 model folders) | One self-contained file per architecture (generated from `modular_*.py` when present). `PreTrainedModel` handles loading, saving, sharding, device placement, attention backend selection, and `push_to_hub`. |
| Weight loading | `core_model_loading.py`, `conversion_mapping.py`, `safetensors_conversion.py`, `modeling_gguf_pytorch_utils.py` | Reads safetensors (or legacy) checkpoints, renames and converts tensors on the fly (`ConversionOps`), and materializes them on the target devices. |
| Auto classes | `models/auto/` (`auto_mappings.py`, `auto_factory.py`, `*_auto.py`) | Map a `model_type` to configuration, model, tokenizer, processor, image/video processor, and feature extractor classes; `register()` adds custom classes; remote code is resolved through `dynamic_module_utils.py`. |
| Tokenizers | `tokenization_utils_base.py` (shared API), `tokenization_utils_tokenizers.py` (Rust `tokenizers` backend), `tokenization_python.py`, `tokenization_utils_sentencepiece.py`, `tokenization_mistral_common.py`, `convert_slow_tokenizer.py` | Text to ids and back, padding and truncation, chat templates (`utils/chat_template_utils.py`), and save/load of tokenizer files. |
| Processors | `processing_utils.py`, `image_processing_*.py`, `video_processing_utils.py`, `feature_extraction_*.py`, `audio_utils.py` | Multimodal preprocessing, combining a tokenizer with image, video, or audio processors. |
| Generation | `generation/utils.py` (`GenerationMixin.generate`), `generation/configuration_utils.py` (`GenerationConfig`), `logits_process.py`, `stopping_criteria.py`, `candidate_generator.py`, `streamers.py`, `watermarking.py`, `continuous_batching/` | Decoding strategies (greedy, sampling, beam search, assisted decoding), logits processors, stopping criteria, KV caches (`cache_utils.py`), and continuous batching for serving. |
| Pipelines | `pipelines/__init__.py` (`pipeline()` factory), `pipelines/base.py`, one module per task | High-level inference: resolves task, model, and components, then runs preprocess → forward → postprocess. |
| Training | `trainer.py`, `training_args.py`, `trainer_*.py`, `optimization.py`, `hf_argparser.py` | The `Trainer` loop, its arguments, callbacks, and the dataclass-based CLI argument parser. |
| Integrations and quantization | `integrations/` (50 modules), `quantizers/` | Attention kernels (SDPA, Flash, flex, paged), DeepSpeed, FSDP, PEFT, tensor parallelism, Hub kernels, and 25+ quantization methods behind a common `HfQuantizer` interface. |
| Hub access | `utils/hub.py` | Download and cache resolution on top of `huggingface_hub` (`cached_file`, `resolve_revision`), the user agent, and `PushToHubMixin` (`push_to_hub`, `_upload_modified_files`). |
| CLI | `cli/transformers.py` (entry point `transformers`), `cli/chat.py`, `cli/serve.py`, `cli/download.py`, `cli/system.py`, `cli/add_new_model_like.py` | `transformers env|chat|serve|download|version|add-new-model-like`. |
| Repository tooling | `utils/*.py` (checkers, modular converter, test fetcher, release), `Makefile` | Style and consistency checks (`utils/checkers.py`), `# Copied from` and modular code generation, test selection for CI, release scripts. |

### Data and control flow

```text
 user code / CLI
      |
      v
 pipeline("task", model=...)            AutoModelFor*.from_pretrained(repo_or_dir)
      |  resolves task, revision,             |
      |  config, components                   v
      +-------------------------------> AutoConfig ---------> PreTrainedConfig (config.json)
                                              |  model_type -> class (auto_mappings / auto_map + remote code)
                                              v
                                       PreTrainedModel.from_pretrained
                                              |  utils/hub.py: cached_file / resolve_revision  <---> Hugging Face Hub
                                              |  core_model_loading: safetensors -> ConversionOps -> device_map
                                              v
 AutoTokenizer / AutoProcessor  --->  tokenizer / processor  ---> input_ids, pixel_values, ...
                                              |
                                              v
                                   model.forward  or  model.generate
                                              |  GenerationConfig -> logits processors, stopping criteria,
                                              |  KV cache (cache_utils), decoding loop
                                              v
                                    outputs -> postprocess (pipelines) -> user

 save_pretrained(dir, push_to_hub=True) -> files on disk -> PushToHubMixin._upload_modified_files -> Hub commit
```

### Extension points

- **Custom models and remote code:** `auto_map` in `config.json` plus `trust_remote_code=True` (`dynamic_module_utils.get_class_from_dynamic_module`), or `Auto*.register(config_class, model_class)` in-process.
- **Custom pipelines:** `PIPELINE_REGISTRY.register_pipeline(...)` or `custom_pipelines` in a config.
- **Attention backends:** `AttentionInterface.register(name, fn)` (`modeling_utils.py`), kernels from the Hub (`integrations/hub_kernels.py`).
- **Logits processors and stopping criteria:** pass `logits_processor=` / `stopping_criteria=` to `generate`, or a custom generation strategy from the Hub.
- **Quantizers:** `HfQuantizer` subclasses registered in `quantizers/auto.py`.
- **Trainer callbacks:** `TrainerCallback` subclasses.
- **Chat templates:** Jinja templates stored with the tokenizer (`chat_template.jinja`, `additional_chat_templates/`).

### Configuration

- Per-model: `config.json`, `generation_config.json`, `tokenizer_config.json`, `processor_config.json`, `preprocessor_config.json`.
- Environment: `HF_HOME` / `HF_HUB_CACHE` (cache), `HF_TOKEN`, `HF_HUB_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY`, `TRANSFORMERS_VERBOSITY`, `TRANSFORMERS_NO_ADVISORY_WARNINGS`; test-only `RUN_SLOW`, `RUN_PIPELINE_TESTS`.
- Telemetry: `utils/hub.http_user_agent` adds library, Python and torch versions, a session id, and usage context to Hub requests unless `HF_HUB_DISABLE_TELEMETRY=1`. Unchanged in this fork and documented in the README.

### Tests

- `tests/models/<model>/` — per-model tests built on shared mixins (`tests/test_modeling_common.py`, `test_tokenization_common.py`, `test_processing_common.py`, `generation/test_utils.py::GenerationTesterMixin`, `test_pipeline_mixin.py`).
- `tests/utils/` — shared utilities (hub, configuration, modeling utils, caches, masking, import structure, CLI helpers); `tests/generation/`, `tests/pipelines/`, `tests/tokenization/`, `tests/cli/`, `tests/trainer/`, and integration folders.
- `@slow` tests (real checkpoints) run only with `RUN_SLOW=1`; GPU, multi-GPU, and backend-specific tests are skipped when the hardware or package is missing. `utils/tests_fetcher.py` selects the tests affected by a change.

### Build, release, and CI (upstream)

- `setup.py` (setuptools) with extras (`torch`, `vision`, `audio`, `testing`, `quality`, `dev`, ...); `dependency_versions_table.py` is generated from it (`deps_table` checker).
- Releases: `utils/release.py` bumps versions; `.github/workflows/release.yml` publishes to PyPI with trusted publishing.
- CI: pull requests run on CircleCI (`.circleci/create_circleci_config.py` + `utils/tests_fetcher.py`) and on the original project's reusable CI workflows; scheduled GPU, AMD, and Gaudi suites run on self-hosted runners; documentation is built and hosted with `doc-builder`. In this fork those workflows are disabled (see [Fork safety](#5-rebrand-and-fork-safety)).

## 2. Environment and baseline

Environment: Windows 10 Pro 19045, 16 CPUs, Python 3.12.2 in a project-local `uv venv`, PyTorch 2.14.1+cpu (the PyPI wheel for Windows is CPU-only), `huggingface_hub` 2.2.0, ruff 0.14.10.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[testing,torch,vision,sentencepiece,tiktoken,chat_template]"
```

Each suite was run with `HF_HUB_DISABLE_TELEMETRY=1 PYTHONUTF8=1 PYTHONPATH=src TOKENIZERS_PARALLELISM=false` and:

```bash
python -m pytest -n 4 --dist=loadfile -p no:cacheprovider -q -rfE --timeout=900 <paths>
```

| Suite | Paths |
| --- | --- |
| utils | `tests/utils` |
| tokenization | `tests/tokenization` |
| generation | `tests/generation/test_logits_process.py test_stopping_criteria.py test_configuration_utils.py test_streamers.py test_candidate_generator.py test_watermarking.py` |
| pipelines_core | `tests/pipelines/test_pipelines_common.py test_pipelines_feature_extraction.py test_pipelines_text_classification.py test_pipelines_text_generation.py` |
| cli | `tests/cli` |
| video_auto | `tests/models/auto/test_video_processing_auto.py` |

The baseline ran on an untouched copy of the upstream tree (the `utils` suite on the derivative before any source change). Results:

| Suite | Passed | Failed | Skipped | Time |
| --- | ---: | ---: | ---: | ---: |
| utils | 667 | 173 (172 failures + 1 worker crash) | 151 (+1 xfail) | 30 min 27 s |
| tokenization | 43 | 0 | 4 | 1 min 7 s |
| generation | 149 | 1 (worker crash) | 5 | 37 min 5 s |
| pipelines_core | 63 | 7 | 13 | 1 min 37 s |
| cli | 87 | 0 | 131 | 50 s |
| video_auto | 16 | 0 | 0 | 27 s |

Causes of the baseline failures (none hidden; all reproduce on upstream):

| Count | Suite | Cause | Kind |
| ---: | --- | --- | --- |
| 151 | utils (146 `test_import_utils`, 5 `test_masking_utils`) | `torch.compile` needs the MSVC compiler: `InductorError: InvalidCxxCompiler: Compiler: cl is not found` | Environment (Windows without Build Tools) |
| 10 + 1 | utils (`test_audio_utils`), pipelines | Optional dependency `torchcodec` not installed | Environment |
| 6 | utils (`test_modeling_utils`) | `PermissionError: [WinError 5]` deleting a temporary directory while the loaded model still memory-maps its `.safetensors` files | Windows test hygiene (the test keeps the model alive inside `TemporaryDirectory`) |
| 2 | utils (`test_add_new_model_like`) | Helper scripts launched with `python` from `PATH`, a different interpreter without `libcst` — **fixed in this fork** (finding F7) | Bug |
| 1 | utils (`test_dynamic_module_utils`) | `WinError 1314`: creating a symlink needs Developer Mode (original issue #48527) | Environment |
| 1 | utils (`test_configuration_utils`) | `bert-base-uncased` could not be resolved from the shared cache/Hub during the run | Network/cache |
| 1 | utils (`test_modeling_utils`) | xdist worker crashed in `test_linear_attention_models_can_use_accelerate_hooks` | Native crash on Windows CPU |
| 1 | generation | xdist worker crashed in the SynthID watermark convergence test (very long CPU run) | Native crash / resources |
| 2 | pipelines | Deprecated-task warnings not seen by `assertWarns` because `huggingface_hub` emits its Windows symlink warning first | Windows test environment |
| 1 | pipelines | `cannot pickle 'generator' object` with `num_workers` under Windows `spawn` | Windows multiprocessing |
| 1 | pipelines | Gated repository needs a token | Credentials |
| 1 | pipelines | HEAD-request counting test depends on cache state | Network/cache |

## 3. Findings

| # | Finding | Severity | Status | Evidence |
| --- | --- | --- | --- | --- |
| F1 | `generate(encoder_outputs=...)` without `input_ids`: `encoder_repetition_penalty` crashes (`index -100 is out of bounds`), `encoder_no_repeat_ngram_size` silently uses placeholder ids (original issue #49695) | High (crash / silent wrong behavior) | Fixed | `tests/generation/test_utils.py::GenerationEncoderInputIdsTest` — before: 2 failed, 2 passed; after: 4 passed |
| F2 | `pipeline(model_kwargs={"code_revision": ...})` raises `TypeError` (duplicate keyword); top-level `code_revision` not forwarded to the model load (original issue #49272) | Medium (documented route unusable, revision pin lost) | Fixed (model load) | `test_pipelines_common.py::CommonPipelineTest::test_pipeline_code_revision` — before: `KeyError: 'code_revision'`; after: pass |
| F3 | `pipeline()` writes `dtype`/`device_map` into the caller's `model_kwargs`; reusing the dict with an explicit `dtype` raises `ValueError` | Medium (spurious error, hidden side effect) | Fixed | `test_pipeline_does_not_mutate_model_kwargs` — before: `ValueError: You cannot use both ...`; after: pass |
| F4 | Default `push_to_hub` repo id from `str.split(os.path.sep)`: `AttributeError` for `pathlib.Path` (5 classes), `''` for a trailing separator, the full path for `/` paths on Windows | Medium (crash / pushes to a wrong or invalid repo) | Fixed in 10 implementations | `tests/utils/test_hub_utils.py::PushToHubLocalPathTests::test_default_repo_id_*` — before: 5 failed (`AttributeError`, `'' != 'my-model'`); after: 5 passed |
| F5 | `_upload_modified_files` builds `path_in_repo` with `os.path.join` → backslashes on Windows for sub-folder files | Medium (wrong file names on the Hub, Windows) | Fixed | `test_upload_modified_files_uses_posix_path_in_repo` — before: `'additional_chat_templates\\tool_use.jinja'`; after: pass |
| F6 | `HfArgumentParser` reads `.args` files with the locale encoding (cp1252 on Windows) | Low-Medium (silent mojibake in arguments, Windows) | Fixed | `test_hf_argparser.py::HfArgumentParserTest::test_parse_args_file_utf8` — before: `'AÃ§Ã£o-...' != 'Ação-模型'`; after: pass |
| F7 | `add-new-model-like` runs helper scripts with `python` from `PATH`; in a non-activated venv the modular conversion fails silently and no `modeling_*.py` is generated | Medium (CLI silently produces an incomplete model) | Fixed | `test_add_new_model_like.py` — before: 3 failed (including the 2 upstream tests); after: 3 passed |
| F8 | `AutoVideoProcessor.register(config, cls, True)` raises `TypeError: 'bool' object is not iterable` (legacy positional `exist_ok`) (original issue #49469) | Low-Medium (backward-compatibility break) | Fixed | `test_video_processing_auto.py::AutoVideoProcessorTest::test_video_processor_registration_legacy_positional_exist_ok` — before: `TypeError`; after: pass |
| F9 | A `code_revision` passed to `AutoTokenizer.from_pretrained` for a native (non-remote) tokenizer is stored in `init_kwargs` and written to `tokenizer_config.json`; concrete model classes reject it | Low | Not fixed (see section 6) | Reproduced with a local tiny BERT |
| F10 | Assigning `config.rope_scaling = {...}` replaces `rope_parameters`, dropping `rope_theta` (original issue #49519) | Medium | Not fixed (see section 6) | Upstream PR #49515 open with 10 tests |
| F11 | `.circleci/create_circleci_config.py` embeds upstream's read-only Hub token for CI | Info (fork safety) | Documented | `.github/workflows-disabled/README.md` |
| F12 | Windows-only test failures (memory-mapped files in `TemporaryDirectory`, symlink warnings vs `assertWarns`, `spawn` pickling) | Low (tests, not library) | Documented | Baseline table above |

## 4. Improvements and how they were verified

Every fix was verified the same way: the new regression test was copied into an untouched upstream tree and run there (it must fail), then run in the derivative (it must pass), with the same interpreter and environment. Logs: `before-tree` vs derivative, `regressions/<label>.{before,after}.log` in the working notes. The before/after runs did **not** set `PYTHONUTF8`, so F6 is exercised with the real Windows default encoding.

### F1 — Encoder-side logits processors with `encoder_outputs` only

- **Root cause:** when `generate` receives `encoder_outputs` but no `input_ids`, `_maybe_initialize_input_ids_for_generation` builds placeholder ids filled with `-100` (a deliberate sanity value), and `generate` passes that tensor as `encoder_input_ids` to `_get_logits_processor`. The only guard was `ndim == 2`, so `EncoderRepetitionPenaltyLogitsProcessor` gathered scores at index `-100`, and `EncoderNoRepeatNGramLogitsProcessor` built n-grams from placeholders without any warning.
- **Fix:** `src/transformers/generation/utils.py:1447` computes `has_encoder_input_ids` once (2-D, integer dtype, no negative ids) and both processors use it; otherwise the existing "requires some form of `input_ids` … ignoring the argument" warning is emitted. Same approach as the open upstream PR #49696.
- **Test:** `GenerationEncoderInputIdsTest` (tiny T5 built from a config, no download): both processors with `encoder_outputs` only must warn and generate; with `input_ids` and `encoder_outputs` they must not warn.

### F2 and F3 — `pipeline()` argument handling

- **Root cause (F2):** `pipeline()` pops `code_revision` from its own keywords and passes `code_revision=code_revision` explicitly to `AutoConfig.from_pretrained(..., **model_kwargs)`, so a `code_revision` inside `model_kwargs` arrives twice; the resolved value was never passed to `load_model`.
- **Root cause (F3):** `model_kwargs` is the caller's dictionary; `pipeline()` stores `device_map` and `dtype` in it (`model_kwargs["dtype"] = dtype`), so the next call with the same dictionary sees a `dtype` "from the user" and raises the conflict error.
- **Fix:** `src/transformers/pipelines/__init__.py:878` copies `model_kwargs`; `:881` accepts `code_revision` from `model_kwargs` (both places at once raise a `ValueError`, as for `device_map` and `dtype`); `:1084` forwards it to `load_model` only when it is set.
- **Tests:** `test_pipeline_does_not_mutate_model_kwargs`, `test_pipeline_code_revision` (local tiny BERT; `load_model` wrapped with `mock` to observe the forwarded value).

### F4 — Default repository name for `push_to_hub`

- **Root cause:** ten `save_pretrained` implementations used `save_directory.split(os.path.sep)[-1]` (some with `str(...)`). This raises on `os.PathLike`, which the signatures accept; returns `''` for `"out/my-model/"`; and on Windows treats `"C:/x/my-model"` as a single component.
- **Fix:** `os.path.basename(os.path.normpath(save_directory))`, the idiom upstream already uses in `dynamic_module_utils.py:441`, in `configuration_utils.py:749`, `generation/configuration_utils.py:937`, `modeling_utils.py:3343`, `tokenization_utils_base.py:2041`, `tokenization_utils_tokenizers.py:607`, `tokenization_mistral_common.py:1641`, `feature_extraction_utils.py:413`, `image_processing_base.py:213`, `video_processing_utils.py:271`, `processing_utils.py:1142`.
- **Tests:** `PushToHubLocalPathTests` checks configuration, generation configuration, tokenizer, feature extractor, and model with three path forms each, with `hf_api` and the upload mocked (no network).

### F5 — Hub paths for files in sub-folders

- **Fix:** `src/transformers/utils/hub.py:789` uses `f"{file}/{f}"` for `path_in_repo` (`huggingface_hub` does not normalize backslashes).
- **Test:** `test_upload_modified_files_uses_posix_path_in_repo` inspects the `CommitOperationAdd` objects passed to a mocked `create_commit`.

### F6 — UTF-8 `.args` files

- **Fix:** `src/transformers/hf_argparser.py:334` reads with `encoding="utf-8"`, consistent with `parse_json_file` and `parse_yaml_file` in the same class.
- **Test:** `test_parse_args_file_utf8` writes `--foo 7 --baz Ação-模型` as UTF-8 and parses it.

### F7 — `add-new-model-like` interpreter

- **Fix:** `src/transformers/cli/add_new_model_like.py:621-627` runs `utils/check_doc_toc.py`, `utils/sort_auto_mappings.py`, and `utils/modular_model_converter.py` with `sys.executable`. The `ruff` calls are unchanged (they need the `ruff` executable, not a Python script).
- **Tests:** the new `test_helper_scripts_use_current_interpreter` (mocks `subprocess.run`, runs in its own copy of the fake repository) and the two upstream tests, which failed in the baseline for this reason and pass now.

### F8 — `AutoVideoProcessor.register` backward compatibility

- **Fix:** `src/transformers/models/auto/video_processing_auto.py:486` maps a boolean third argument to `exist_ok`, keeping the new parameter order and keyword use.
- **Test:** registering `LlavaOnevisionConfig` with `False` must raise the "already used" `ValueError`, and with `True` must register.

### Checks on the changed files

- `ruff check` and `ruff format --check` (ruff 0.14.10, the version pinned in `setup.py`) pass on all 23 changed Python files; formatting was applied only to lines this fork added.
- Repository consistency checkers (`python utils/checkers.py init_isort,inits,copies,docstrings,dummies,doc_toc,pipeline_typing,import_complexity,deps_table,noisy_comments --keep-going --no-cache`) were run on the derivative and on the untouched upstream tree; see section 7 for the comparison.

## 5. Rebrand and fork safety

- Renamed: README titles and text, the README banner (`docs/assets/cavaleiro-banner.svg`), repository links (clone URL, issues, advisories, CI badge), `setup.py` `url`, issue forms, pull request template, community files. Translated upstream READMEs moved to [`docs/original-i18n/`](../original-i18n/README.md).
- Kept (compatibility): the PyPI distribution and import name `transformers`, the `transformers` CLI, version string, environment variables, cache paths, config and checkpoint file names, Hub endpoints and telemetry (documented, not repointed), and in-product strings.
- Workflows: 57 upstream workflow files plus `TROUBLESHOOT.md` moved to [`.github/workflows-disabled/`](../../.github/workflows-disabled/README.md) (publishing, non-`GITHUB_TOKEN` secrets, the original project's reusable workflows and dashboards, self-hosted GPU runners, upstream bots). Active: `cavaleiro-ci.yml` (new), `trufflehog.yml`, `add-model-like.yml` (inert). CircleCI is documented, not connected.
- `CODEOWNERS` added (`* @cavaleiro-ai`); there was no `FUNDING.yml`. Upstream-only issue forms (new model, translation, legacy migration) removed, and maintainer mention lists dropped so fork issues do not notify upstream maintainers.
- Attribution: `LICENSE` byte-identical to upstream, `NOTICE` added, and a `Modified by Cavaleiro AI (2026): ...` line below the license header of each of the 23 changed source and test files (plus the README and contributing guide comments).
- Identity cleanup (after 1.0.0): `CITATION.cff` removed; the community files, issue forms, `setup.py` author, and reviewer-routing file name only Cavaleiro AI; the inherited docs in `docs/` name the product Cavaleiro Model Framework, keeping paper titles and abstracts, external article titles, historical notes, model contribution credits, and version requirements unchanged, with support links pointing to this repository's issues.

## 6. What was not done, and why

- **F9 (`code_revision` for tokenizers and processors):** forwarding `code_revision` to `AutoTokenizer` would store it in `tokenizer_config.json` for every native tokenizer, and concrete model classes reject it. The right fix is in `AutoTokenizer` and the processor auto classes (pop `code_revision` and pass it only to `get_class_from_dynamic_module`), which is a broader change; it is left as a next step. The model load, which is where remote modeling code runs, is fixed.
- **F10 (`rope_scaling` setter):** a correct fix must merge legacy keys into the per-layer-type `rope_parameters` structure and validate the result. Upstream PR #49515 already does this with 10 tests; taking it when it lands is safer than a divergent fix.
- **Windows-only test failures (F12)** were not changed: they are test-environment issues, and editing upstream tests only to pass on Windows would diverge from upstream without fixing library behavior.
- **No large-model runs:** no `RUN_SLOW` tests, no GPU tests, and no download above ~500 MB; generation and pipelines were verified with tiny models built in code or `hf-internal-testing` fixtures.
- **No dependency upgrades, no reformatting** of untouched code, no public API changes beyond the bug fixes listed in the changelog.

## 7. Final verification

The same suites and command as the baseline, on the derivative, with one difference: `tests/utils/test_modeling_utils.py::ModelUtilsTest::test_linear_attention_models_can_use_accelerate_hooks` was deselected from the `utils` suite. It crashed its pytest-xdist worker in the baseline as well; in the first final attempt it crashed two workers in a row and the xdist controller deadlocked. Run alone (no xdist) it **passes on both** the upstream tree and the derivative (99.96 s and 30.05 s), so the crash is environmental and unrelated to these changes.

| Suite | Baseline (passed / failed / skipped) | Final (passed / failed / skipped) | Difference |
| --- | --- | --- | --- |
| utils | 667 / 173 / 151 (+1 xfail) | 676 / 170 / 151 (+1 xfail, 1 deselected) | +8 new tests; 2 upstream tests fixed by F7; crash-prone test deselected. No test fails in final that passed in baseline. |
| tokenization | 43 / 0 / 4 | 43 / 0 / 4 | — |
| generation | 149 / 1 / 5 | 149 / 0 / 5 | Baseline worker crash in the SynthID convergence test did not recur. |
| pipelines_core | 63 / 7 / 13 | 65 / 7 / 13 | +2 new tests; the same 7 environmental failures. |
| cli | 87 / 0 / 131 | 87 / 0 / 131 | — |
| video_auto | 16 / 0 / 0 | 17 / 0 / 0 | +1 new test. |

The new generation tests live in `tests/generation/test_utils.py`, which is not part of the generation suite above (its model-mixin tests are very long on CPU); they were run on their own: `python -m pytest "tests/generation/test_utils.py::GenerationEncoderInputIdsTest"` → 4 passed (2 failed on upstream).

Static checks:

| Check | Upstream tree | Derivative |
| --- | --- | --- |
| `ruff check` / `ruff format --check` on the 23 changed Python files | — | pass |
| `utils/checkers.py`: `init_isort`, `inits`, `copies`, `dummies`, `doc_toc`, `pipeline_typing`, `import_complexity`, `deps_table` | OK | OK |
| `utils/checkers.py`: `noisy_comments` (report-only) | 339 findings | 339 findings (none in changed files) |
| `utils/checkers.py`: `docstrings` | fails: needs a git `main` ref (the derivative is an uncommitted repository by design; the copy of upstream has no `.git`) | same |
| `python utils/check_docstrings.py --check_all` | 1 pre-existing problem (redundant docstring in `models/idefics/image_processing_pil_idefics.py`) | identical output |
| `git diff --cached --check` on the files written for this fork | — | clean, except upstream's own Markdown line-break spaces in the moved `docs/original-i18n/README_{fa,ko,zh-hans}.md` (content unchanged) |
| Relative links and anchors in the 30 Markdown files written or moved | — | 476 links, 0 problems |

## 8. Known limitations

- The full upstream suite (tens of thousands of tests, many `@slow` or GPU-only) was not run; the targeted subset above covers the modules changed.
- `torch.compile`-dependent code paths could not be exercised on this machine (no MSVC).
- The fork carries upstream's version string; tools that compare `transformers.__version__` cannot tell the two distributions apart.
- Upstream moves fast (several commits a day); fixes here that upstream later merges in a different form should be replaced by the upstream version on the next rebase.

## 9. Recommended next steps

1. Rebase on upstream regularly; drop F1, F2, and F8 when upstream PRs #49696 / the #49272 fix / #49619 land, and take #49515 for F10.
2. Fix F9 in `AutoTokenizer` and the processor auto classes, then forward `code_revision` from `pipeline()` to all component loaders.
3. Make the Windows test failures in F12 robust upstream (release model references before temporary-directory cleanup, filter the symlink warning in `assertWarns` tests).
4. Add a Windows job to Cavaleiro CI for the targeted suite.
5. Offer F3–F7 upstream as small, focused pull requests with the tests from this fork, following upstream's contribution policy.
