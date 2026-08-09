# Privacy and credentials

Engineering specifications, OCR text, normalized rows, page images, quality reports, prompts, and
Excel outputs may all contain confidential or regulated information. Treat every derived artifact
as sensitively as the source PDF.

## Know the data boundary

A local extraction provider is not proof that the entire run is local. For example, Tesseract can
produce OCR locally and a later mapping provider can still transmit that OCR to an external
service.

- `pdf-text` and system Tesseract operate locally.
- Apple Silicon MLX and a loopback OpenAI-compatible model server operate locally when correctly
  configured.
- `opencode` is configuration-dependent and may use an external account or service.
- `openai-compatible` becomes external when `--ai-external-openai-compatible` points to a remote
  API.
- An external mapping request can include OCR text. Page images may also be included depending on
  provider and flags.

`map-ocr` defaults to `--ai-client none`: it uses a supported deterministic table layer or stops.
An AI provider must be selected explicitly. This fail-closed default prevents an ordinary mapping
command from silently falling through to an authenticated external session.

`--no-image-attachments` suppresses page-image attachments. It does **not** make an external run
local and does **not** prevent OCR text from being transmitted.

Before an external run, obtain authorization from the document owner and review the provider's
terms, data region, retention, model-training policy, subprocessors, access controls, and deletion
process. The project cannot make those decisions for the user.

## Authentication is the user's responsibility

Users must authenticate with their own provider accounts and follow their organization's access
policy. Never share, publish, sell, or redistribute:

- API keys or `.env` files
- OpenCode or browser session files
- refresh tokens, cookies, or exported authentication state
- CI secrets, SSH keys, cloud credentials, or model-registry tokens

Pass external API credentials by naming an environment variable with `--ai-api-key-env`; do not
place the secret value in a CLI argument. Use an OS credential store or CI secret manager where
available. Remote OpenAI-compatible endpoints must use HTTPS; cleartext HTTP is accepted only for
loopback servers.

The `export-gpt-auth` command, if used for an authorized migration, creates a sensitive credential
artifact. It is never a distribution mechanism. Keep the result outside public packages and Git,
restrict its permissions, transfer it through an approved secret channel, and delete it when the
migration is complete. Never ask another user to send you their session or token.

## Local processing checklist

- Bind model servers to `127.0.0.1`, not a public interface.
- Restrict permissions on PDFs, rendered pages, caches, JSON, logs, and workbooks.
- Verify that diagnostics and crash reports do not upload content.
- Disable image attachments unless they are required and approved.
- Use encrypted storage and an approved backup policy.
- Define a retention period for source files, intermediate images, prompts, and outputs.
- Remove temporary artifacts securely according to the storage medium and organization policy.

Local model downloads still contact a model registry and can reveal IP address and model choice,
although they should not transmit the source document. Pin and verify approved model revisions.
Some model formats also include executable code. The experimental `dots-mocr` provider therefore
accepts only a local snapshot and requires the explicit `--dots-trust-model-code` opt-in after that
snapshot has been reviewed; the project does not download or trust it automatically.

## Logs, issues, and screenshots

Do not paste real OCR output, document paths, credentials, customer names, or workbook screenshots
into public issues. Reproduce defects with the synthetic demo or a newly generated minimal fixture.
Redaction must cover document content, metadata, file names, window titles, terminal history, and
background applications.

## Suspected exposure

Stop processing, revoke and rotate the credential, preserve appropriate audit evidence, and notify
the responsible security or privacy contact. Deleting a file from the latest Git commit is not
enough: secrets and customer data must also be purged from every reachable history, tag, release,
cache, fork, and artifact. If any sensitive history may have reached a public remote, treat it as a
disclosure and follow the host's purge procedure.

See [Releasing](releasing.md) for the clean-history requirement.
