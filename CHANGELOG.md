# Changelog

All notable changes to Spec OCR Pipeline will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-09

### Added

- PDF text-layer, Tesseract, PaddleOCR-VL, and dots.mocr extraction paths.
- AI semantic mapping through OpenCode, MLX, and OpenAI-compatible backends.
- Normalized JSONL, quality and performance reports, plain Excel export, and
  template-based commercial workbook export.
- Local model management, hardware diagnostics, and Windows/WSL setup flows.
- A one-command, local-only `spec-ocr process` workflow for supported searchable
  PDFs, with explicit overwrite protection and run metadata.
- A redistributable synthetic PDF fixture, ground truth, expected workbooks,
  package-safe templates, and a reproducible project screenshot.
- Native Linux, macOS, and Windows CI coverage for Python 3.11-3.14, plus
  source-distribution inspection and installed-wheel smoke tests.
- Public contribution, conduct, security, support, issue, pull request, and
  dependency maintenance policies.

### Changed

- Package resources, manifests, output paths, diagnostics, subprocess encoding,
  Windows bootstrap behavior, and optional dependency groups are now portable
  across supported platforms and clean wheel installations.
- External provider authentication uses each user's own environment by default;
  repository-bundled sessions and credentials are neither required nor shared.
- `map-ocr` now fails closed with AI disabled unless a provider is selected explicitly; remote
  OpenAI-compatible endpoints must use HTTPS while loopback development servers may use HTTP.
- Experimental dots.mocr model downloads and model-supplied code execution now fail closed; users
  must provide an audited local snapshot and explicitly opt in to its code.
- WSL setup checks for a user-installed OpenCode CLI instead of executing a remote shell installer.

### Security

- Documented private vulnerability reporting and strict handling rules for
  credentials, source documents, OCR artifacts, logs, and generated outputs.
- Escaped untrusted OCR and AI strings before XLSX export to prevent spreadsheet
  formula injection while preserving formulas owned by the quote template.
- Added an automated public-tree gate that rejects private document locations,
  secrets, bundled runtimes, model archives, oversized files, and local home paths.

[Unreleased]: https://github.com/floze-the-genius/spec-ocr-pipeline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/floze-the-genius/spec-ocr-pipeline/releases/tag/v0.1.0
