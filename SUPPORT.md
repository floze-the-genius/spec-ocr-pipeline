# Support

## Where to Ask

- Use a bug report for reproducible defects in the project.
- Use a feature request for proposed behavior or platform support.
- Use GitHub Discussions for usage questions when Discussions are enabled.
- Follow [SECURITY.md](SECURITY.md) for vulnerabilities or accidental exposure
  of credentials or private documents.

Do not use public support channels for account tokens, authentication exports,
private model endpoints, or confidential documents.

## Before Opening a Report

1. Reproduce the problem with the latest available revision or release.
2. Run `spec-ocr doctor --json` and remove usernames, local paths, hostnames, and
   other sensitive values before sharing its output.
3. Record the operating system, architecture, Python version, installation
   method, command, OCR provider, AI backend, and model/runtime.
4. Reduce the problem to a synthetic, redistributable input whenever possible.
5. Search existing issues for the same error and workaround.

A useful report explains the expected result, actual result, exact reproduction
steps, and validation already attempted. Paste text logs rather than terminal
screenshots when possible, after sanitizing them.

## Data Safety

Never attach real client or production PDFs, spreadsheets, rendered pages, OCR
JSON, result archives, API keys, cookies, or authentication files. Document
metadata and generated Excel files can contain private information even when
the visible sample appears harmless.

If a problem occurs only with a private document, describe its structural
properties and create a minimal synthetic document with the same layout. If
that is impossible, report only the software symptoms and do not upload the
source.

## Support Scope

Community support is best effort. Maintainers can help with project code,
documented installation paths, and reproducible compatibility problems. They
cannot guarantee OCR or model accuracy for every document, restore third-party
accounts, grant access to hosted models, or support modified private builds
without a reproducible example.

Questions about a third-party model, runtime, service plan, or account may need
to be handled by that upstream provider.
