# Releasing

## Mandatory clean-history rule

**The first public push must use a clean repository or clean orphan history. Do not push the
existing development history.** Earlier commits contained customer PDFs/XLSX files and bundled
binaries. Deleting those files in a later commit does not remove their contents, names, metadata,
or credentials from Git history.

The preferred release process is to copy an audited allowlist of current files into a brand-new
directory with no `.git` directory, initialize a new repository there, and create the public root
commit. A carefully prepared orphan branch is acceptable only when reviewers can prove that no old
branch, tag, note, reflog-derived ref, release, LFS object, submodule, or remote ref can be pushed.
A fresh repository is safer.

Example initialization inside the separately audited public directory:

```text
git init --initial-branch="main"
git status --short
git add --all
git diff --cached --stat
git commit -m "Initial public release"
```

Do not add a public remote until the staged tree and the new repository's object graph have passed
the checks below. Never use the development repository's tags for the public release.

## Legal and asset review

Obtain explicit legal or authorized-owner approval for:

- the project license and copyright notices;
- every dependency and optional model license, including redistribution and commercial-use terms;
- workbook templates, fonts, icons, screenshots, documentation, and demo assets;
- names, trademarks, attribution, notices, and generated-content provenance;
- any sample claimed to be synthetic.

Customer PDFs, XLSX workbooks, OCR text, page images, filenames, document metadata, and derived
fixtures must not be included. Bundled executables, DLLs, model weights, package archives, and
language packs must not be included unless a separate, documented legal and security review
authorizes each exact artifact. Tesseract must remain a system dependency.

## Pre-push audit

Run these checks in the new public repository and review the output, not merely the exit code:

```text
git status --short
git log --oneline --decorate --all
git rev-list --objects --all
git ls-files
```

Also:

1. Run an approved secret scanner over every tracked file and every reachable object.
2. Search filenames and contents for customer names, email addresses, URLs, local paths, account
   identifiers, tokens, cookies, and document metadata.
3. Review large blobs and binary file signatures. Allowlist each required binary asset.
4. Confirm that `.gitignore` is only a guardrail; ignored files are not proof of deletion from
   history.
5. Inspect generated screenshots at full resolution for terminal history, window titles,
   notifications, and background applications.
6. Verify that examples are synthetic and reproducible from documented generators or source data.

If any prohibited object is reachable, discard the public repository and create a new clean one.
Do not rely on a follow-up deletion commit.

## Release verification

Before building, synchronize the release identity:

1. Set the SemVer value in `src/spec_pipeline/_version.py`.
2. Move the relevant `Unreleased` entries in `CHANGELOG.md` into a section named
   `## [<version>] - YYYY-MM-DD`.
3. Build from the exact reviewed commit and verify that `spec-ocr --version`, wheel and sdist
   filenames, and package metadata all report the same version.
4. Create the matching annotated `v<version>` tag only after every release gate passes. Never
   reuse or move a published tag.

From a clean checkout of the proposed public root commit:

```text
python -m pip install --upgrade "pip"
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check "src" "tests" "scripts"
python -m ruff format --check "src" "tests" "scripts"
python -m build
python -m twine check "dist/spec_ocr_pipeline-<version>-py3-none-any.whl" "dist/spec_ocr_pipeline-<version>.tar.gz"
spec-ocr --version
```

Replace `<version>` with the exact version produced by the build. Passing exact artifact paths is
portable and avoids shell-specific glob handling.

The required core CI matrix covers native Linux, macOS, and Windows and exercises the synthetic
searchable-PDF-to-XLSX flow. Test Tesseract with a system binary in separate jobs. Hardware jobs for
Apple Silicon MLX, Linux x86_64 NVIDIA, and WSL2 NVIDIA are separate from core CI and must not be
simulated by claiming an import-only check proves inference.

Install the built wheel into a new virtual environment and repeat `spec-ocr doctor`, the synthetic
demo, JSONL export, both XLSX exporters, and package-data discovery from outside the source tree.
Test supported Python versions and verify that a wheel contains only intended files.

## Publish

1. Record the approved commit and immutable test results.
2. Create checksums and, when the project adopts it, sign release artifacts and provenance.
3. Push only the new clean public branch and reviewed tags.
4. Publish artifacts built from a clean checkout, not from an untracked or dirty worktree.
5. Re-download the public release and repeat the smoke test.
6. Verify that no user credential, OpenCode session, API token, model cache, or private input is in
   the repository, CI logs, releases, or package archives.

See [Privacy](privacy.md) for credential and document-handling requirements.
