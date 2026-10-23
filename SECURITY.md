# Security Policy

Cavaleiro Model Framework downloads and runs models, tokenizers, processors, and sometimes code published by other people, so the guidance below matters as much as the reporting process.

## Reporting a vulnerability

**Please do not open a public issue.** Report privately through GitHub:

1. Go to the repository's **Security** tab.
2. Select **Report a vulnerability** (private vulnerability reporting), or open the form directly at <https://github.com/cavaleiro-ai/cavaleiro-model-framework/security/advisories/new>.

Include the affected file and line or API, the version or commit, steps to reproduce (a tiny model or a config built in code is ideal), and the impact you expect. Redact tokens and private data.

### Vulnerabilities in dependencies

Vulnerabilities in other projects (`huggingface_hub`, `tokenizers`, `safetensors`, PyTorch, ...) belong to those projects. If a vulnerability in inherited code also affects the `transformers` release on PyPI, report it here privately as well, so that it can be fixed in this repository.

### What to expect

Cavaleiro AI maintains this project on a best-effort basis:

- Acknowledgement of your report within **7 days**.
- An initial assessment, and a plan or a request for more information, within **14 days**.
- A fix on `main` and a published advisory crediting you (unless you prefer to stay anonymous) once the issue is resolved.

Only the latest release and the `main` branch receive security fixes.

## Hugging Face Hub, remote artefacts, and remote code

This library is tightly coupled to the Hugging Face Hub. You can use it offline with pre-downloaded model weights, but it also provides a very simple way to download, use, and manage models locally.

When downloading artefacts that have been uploaded by others on any platform, you expose yourself to risks. Please read below for the security recommendations in order to keep your runtime and local environment safe.

### Remote artefacts

Models uploaded on the Hugging Face Hub come in different formats. We heavily recommend uploading and downloading models in the [`safetensors`](https://github.com/huggingface/safetensors) format (which is the default prioritized by the library), as developed specifically to prevent arbitrary code execution on your system.

To avoid loading models from unsafe formats (e.g. [pickle](https://docs.python.org/3/library/pickle.html)), you should use the `use_safetensors` parameter. If doing so, in the event that no `.safetensors` file is present, the library will error when loading the model.

### Remote code

#### Modeling

The library supports many model architectures, but is also the bridge between your Python runtime and models that are stored in model repositories on the Hugging Face Hub.

These models require the `trust_remote_code=True` parameter to be set when using them; please **always** verify the content of the modeling files when using this argument. We recommend setting a revision (and `code_revision` when the code lives in another repository) in order to ensure you protect yourself from updates on the repository.

### Tokens and telemetry

- Never commit Hub tokens (`HF_TOKEN`) or paste them in issues. Use a fine-grained, read-only token unless you need to upload.
- Requests to the Hub carry a user agent with version information and usage context (see [Compatibility](README.md#compatibility)). Set `HF_HUB_DISABLE_TELEMETRY=1` to reduce it, or `HF_HUB_OFFLINE=1` to make no Hub requests.

## Scope

In scope: changes made in this repository (see [CHANGELOG.md](CHANGELOG.md)), the active GitHub workflows in `.github/workflows/`, and any vulnerability in the code shipped here, including inherited code.

Out of scope: models, datasets, and remote code published on the Hub by third parties; workflows in `.github/workflows-disabled/`, which do not run.
