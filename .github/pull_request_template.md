## Summary

Describe the problem and the outcome of this change.

## Related Issue

Link the issue or explain why one is not needed.

## Changes

<!-- List the main implementation changes. -->

## Validation

List the exact commands that were run.

```text
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest -q
python scripts/check_public_tree.py
```

| Platform | OS and architecture | Python | Provider/runtime | Result |
| --- | --- | --- | --- | --- |
| Tested |  |  |  |  |
| Not tested |  |  |  |  |

## Compatibility and Output Impact

Describe changes to the CLI, dependencies, normalized schema, quality gates,
generated files, or supported platforms. Write "None" when not applicable.

## Screenshots

Include screenshots only for user-visible changes and only when every visible
value is synthetic and safe for public redistribution.

## Checklist

- [ ] The change is focused and includes regression tests or an explanation of
      why tests are not needed.
- [ ] Ruff and relevant pytest checks pass.
- [ ] Tested and untested platform combinations are recorded above.
- [ ] User-facing behavior, installation, and compatibility documentation is
      updated where needed.
- [ ] `CHANGELOG.md` is updated for a notable change.
- [ ] No real client or production documents, proprietary templates, private
      output, personal data, credentials, authentication files, or secrets are
      included in commits, logs, fixtures, or screenshots.
- [ ] New fixtures are synthetic or explicitly licensed for redistribution and
      have been checked for embedded metadata.
- [ ] Optional provider dependencies remain isolated from unrelated CLI paths.
- [ ] Schema or generated-output changes include compatibility notes.
