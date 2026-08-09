# Contributing to Spec OCR Pipeline

Thank you for helping improve Spec OCR Pipeline. Contributions to extraction,
normalization, validation, exports, platform support, tests, and documentation
are welcome.

## Before You Start

- Search existing issues and pull requests before opening a duplicate.
- Use an issue to discuss large features, schema changes, new model runtimes,
  or changes that affect generated workbooks.
- Keep each pull request focused on one problem.
- Use the security process in [SECURITY.md](SECURITY.md) for vulnerabilities;
  do not report them in a public issue.

## Protect Documents and Credentials

This rule is non-negotiable. Never commit or upload:

- real client, customer, employer, or production documents;
- proprietary PDF, spreadsheet, image, template, archive, or model files;
- document text, OCR output, logs, screenshots, or metadata containing private
  project details or personal data;
- API keys, access tokens, passwords, cookies, authentication exports,
  private keys, or populated environment files;
- generated artifacts that reproduce confidential source content.

Use a small synthetic fixture that you created and may redistribute. Replace
names, addresses, identifiers, prices, contacts, and document metadata with
invented values. A redacted document is acceptable only when its license and
every remaining field are safe to publish; a synthetic reproducer is strongly
preferred.

If a credential is exposed, revoke or rotate it immediately and report the
incident privately as described in [SECURITY.md](SECURITY.md). Deleting it in
a later commit is not sufficient because Git retains history.

## Development Setup

Python 3.11–3.14 is required.

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Optional OCR and model backends have additional system and Python
dependencies. Install only the extras needed for the area you are testing,
and record them in the pull request.

## Validate Your Change

Run linting, formatting checks, and the test suite from the repository root:

```bash
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest
```

During development, a focused test is useful:

```bash
python -m pytest tests/test_cli.py -q
```

Add regression tests for bug fixes and tests for new behavior. Unit tests
should use fakes or mocks for network services, large models, and optional
hardware. Tests must not require a contributor's credentials or private
documents.

## Platform Checklist

Every pull request must state which checks were run and where. Mark untested
platforms explicitly instead of assuming compatibility.

- [ ] Record the operating system, version, architecture, and Python version.
- [ ] Run the relevant unit tests and `spec-ocr doctor` on each tested platform.
- [ ] For path, subprocess, encoding, or installer changes, check paths with
      spaces and non-ASCII characters.
- [ ] For Windows changes, test PowerShell or CMD behavior and Windows/WSL path
      conversion as applicable.
- [ ] For macOS changes, state whether the test used Apple Silicon or Intel and
      whether an MLX or Quartz path was involved.
- [ ] For Linux changes, state the distribution and whether the run used CPU,
      NVIDIA/CUDA, or another accelerator.
- [ ] For provider-specific changes, name the OCR/AI provider, model, runtime,
      and dependency extra used.
- [ ] Confirm that unsupported or untested combinations fail clearly and are
      documented honestly.

## Code and Compatibility Guidelines

- Preserve the normalized row schema unless the change has been discussed and
  includes migration notes and tests.
- Keep provider-specific behavior behind the existing OCR or AI client
  boundaries.
- Prefer deterministic validation and actionable error messages.
- Do not silently weaken quality gates or turn a failed extraction into a
  successful empty result.
- Keep optional dependencies optional; the base CLI should not import a heavy
  runtime merely to display help or run unrelated commands.
- Update user-facing documentation and `CHANGELOG.md` when behavior, output,
  installation, or compatibility changes.

## Pull Requests

In the pull request description:

1. Explain the problem and the chosen approach.
2. Link the related issue when one exists.
3. List the exact validation commands and platform results.
4. Describe compatibility, output-schema, dependency, and privacy impact.
5. Add screenshots only when they contain synthetic, redistributable data.

Maintainers may request that a change be split, receive additional tests, or
be rebased before merge. By submitting a contribution, you agree that it is
licensed under the repository's MIT License and that you have the right to
submit it.
