# Security Policy

## Supported Versions

Security fixes are applied to the latest tagged release and the default branch.
Before the first tagged release, the default branch is the only supported
version.

| Version | Security support |
| --- | --- |
| Latest tagged release | Supported |
| Default branch | Pre-release fixes |
| Older releases | Unsupported unless explicitly announced |

## Report a Vulnerability Privately

Do not open a public issue, pull request, or discussion for a suspected
vulnerability.

Use GitHub Private Vulnerability Reporting when it is enabled for this
repository. If it is unavailable, contact an active maintainer privately
through a contact method available on their GitHub profile and ask for a secure
reporting channel. Do not send exploit details or credentials until a private
channel is established.

Include, when possible:

- the affected release or commit;
- the affected operating system, provider, and runtime;
- a concise impact assessment and attack scenario;
- minimal reproduction steps using synthetic data;
- logs with paths, tokens, document content, and personal data removed;
- any known workaround or mitigation.

Never attach real business documents, authentication exports, API keys, model
service credentials, or unredacted generated artifacts.

## Response Process

Maintainers aim to acknowledge a complete report within seven days. They will
validate the issue, agree on disclosure timing, prepare a fix, and credit the
reporter if requested and appropriate. Complex or upstream issues may take
longer, but material updates should be shared through the private report.

Please allow a reasonable remediation period before public disclosure. If the
issue belongs to an upstream OCR engine, model runtime, package, or hosted
provider, maintainers may coordinate with that project.

## Credential or Document Exposure

If a credential was exposed, revoke or rotate it immediately. Removing it from
the latest commit does not remove it from Git history, caches, forks, logs, or
release archives.

If a private document or generated artifact was exposed, stop distributing the
affected revision or release and report the incident privately. Treat OCR JSON,
rendered page images, spreadsheets, and diagnostic logs as potentially
sensitive because they may reproduce source document content.

## Security Boundaries

Model accuracy, an unavailable hosted model, and ordinary OCR mistakes are not
security vulnerabilities by themselves. Reports are in scope when they involve
issues such as credential disclosure, unsafe command execution, path traversal,
untrusted-file handling, dependency compromise, or unintended transmission of
document data.
