# Disabled upstream workflows

These GitHub Actions workflows come from the original project's repository. GitHub only runs workflows stored in `.github/workflows/`, so moving them here turns them off without losing them.

They were disabled in Cavaleiro Model Framework because each one does at least one of the following, which a derivative must not do under the original project's identity:

- publishes packages, Docker images, or documentation (PyPI, Anaconda, Docker Hub, the Hugging Face Hub);
- needs secrets other than `GITHUB_TOKEN` (Hub tokens, Docker Hub credentials, Slack bots, Tailscale, GitHub Apps);
- calls Hugging Face infrastructure, such as the reusable workflows in `huggingface/transformers-ci`, `huggingface/doc-builder`, or `huggingface/security-workflows`, or the upstream CI dashboards;
- runs on self-hosted GPU, AMD, or Intel Gaudi runners, or on the benchmark fleet;
- automates upstream triage policy (reviewer assignment, stale issues, anti-spam labels, bot comments).

The fork's active CI lives in [`../workflows/`](../workflows/): [`cavaleiro-ci.yml`](../workflows/cavaleiro-ci.yml) (lint and targeted CPU tests, read-only `GITHUB_TOKEN`), plus the upstream [`trufflehog.yml`](../workflows/trufflehog.yml) secret scan and [`add-model-like.yml`](../workflows/add-model-like.yml), which upstream keeps switched off (it only triggers on a branch named `none`).

## Inventory

| Workflow | Why it is disabled |
| --- | --- |
| `release.yml` | Builds and publishes the `transformers` package to PyPI (trusted publishing, `pypi-release` environment). |
| `release-conda.yml` | Publishes to Anaconda with `ANACONDA_API_TOKEN`. |
| `build-docker-images.yml`, `build-ci-docker-images.yml`, `build-nightly-ci-docker-images.yml`, `build-past-ci-docker-images.yml` | Push `huggingface/*` images to Docker Hub with Docker Hub credentials; some run on self-hosted runners and post to Slack. |
| `build_documentation.yml`, `build_pr_documentation.yml`, `upload_pr_documentation.yml`, `pr_build_doc_with_comment.yml` | Build and upload the documentation to Hugging Face hosting through `huggingface/doc-builder`, with Hub and bot tokens. |
| `update_metdata.yml` | Pushes library metadata to the Hugging Face Hub with a personal Hub token. |
| `check_tiny_models.yml` | Creates and uploads tiny test models to the Hub with a bot token. |
| `pr-ci-caller.yml`, `pr-ci-security-gate.yml`, `pr-ci-post-dashboard-link.yml`, `nightly-integration-failure-triage-caller.yml`, `push-important-models.yml` | Call reusable workflows in `huggingface/transformers-ci` and the upstream CI dashboards. |
| `self-*.yml`, `model_jobs_intel_gaudi.yml`, `doctest_job.yml`, `doctests.yml`, `check_failed_tests.yml`, `collated-reports.yml`, `slack-report.yml`, `vllm-ci-caller.yml` | Scheduled and comment-triggered GPU, AMD, and Gaudi test suites on self-hosted runners, with Hub, Slack, and result-upload tokens. |
| `benchmark.yml`, `benchmark_v2.yml`, `benchmark_v2_a10_caller.yml`, `benchmark_v2_mi300_caller.yml` | Benchmarks on self-hosted GPU runners that push results to the Hub. |
| `ssh-runner.yml` | Opens an SSH session on a self-hosted GPU runner through Tailscale. |
| `extras-smoke-test.yml`, `new_model_pr_merged_notification.yml` | Post to upstream Slack channels. |
| `ai-review.yml`, `serge-verify-caller.yml` | Hugging Face's review bot (GitHub App secrets, self-hosted runner). |
| `trl-ci-bot.yml` | Dispatches TRL CI runs with a cross-repository token. |
| `pr-repo-consistency-bot.yml`, `pr_slow_ci_suggestion.yml`, `get-pr-info.yml`, `get-pr-number.yml` | Comment bots for upstream pull requests (the last two are helpers for them). |
| `assign-reviewers.yml`, `anti-slop.yml`, `stale.yml` | Upstream triage policy: reviewer assignment, spam labeling (`pull_request_target` with write access), stale-issue closing (guarded to `huggingface/transformers`). |
| `codeql.yml`, `check-workflow-permissions.yml` | Call `huggingface/security-workflows` reusable workflows. |
| `circleci-failure-summary-comment.yml.bak` | Already inactive upstream (`.bak`); uses a Hub write token. |
| `TROUBLESHOOT.md` | Upstream notes for the workflows above. |

`.github/scripts/` and `.github/conda/` are support files for the workflows above and stay where upstream keeps them.

## CircleCI

Upstream's main pull request CI runs on CircleCI ([`.circleci/`](../../.circleci/)). It is not a GitHub workflow and only runs if someone connects the repository to CircleCI. Do not connect it as it is: it expects Hugging Face's CircleCI contexts and Docker images, and `.circleci/create_circleci_config.py` embeds upstream's read-only Hub token for CI. Replace those with your own before enabling it.

## How to re-enable a workflow

1. Read the workflow and list every secret, environment, runner label, reusable workflow (`uses: owner/repo/...`), and Hub, Docker, or Slack target it references.
2. Replace every upstream identity with your own: package names, image names (`huggingface/...`), Hub repositories, documentation targets, Slack channels, and `github.repository == 'huggingface/transformers'` guards.
3. Create your own secrets and environments in the repository settings. Never reuse upstream credentials.
4. Replace self-hosted runner labels with runners you control, or with GitHub-hosted runners.
5. Keep `permissions:` minimal and keep third-party actions pinned to a full commit SHA, as upstream does.
6. Move the file back to `.github/workflows/` and run it once on a branch with `workflow_dispatch` before enabling its other triggers.

Publishing a package named `transformers`, or images and documentation under `huggingface/...`, would impersonate the upstream project. Choose your own names first.
