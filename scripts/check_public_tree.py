#!/usr/bin/env python3
"""Fail when the public source tree or distribution contains private or bulky payloads."""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024

PRIVATE_PREFIXES = (
    "samples/client_examples/",
    "customer-secrets/",
    ".customer-secrets/",
    ".private/",
    "tools/windows/wheelhouse/",
)

FORBIDDEN_BINARY_SUFFIXES = {
    ".7z",
    ".bin",
    ".bz2",
    ".dll",
    ".dmg",
    ".dylib",
    ".exe",
    ".gguf",
    ".gz",
    ".iso",
    ".key",
    ".msi",
    ".onnx",
    ".p12",
    ".pem",
    ".pfx",
    ".pickle",
    ".pkg",
    ".pkl",
    ".pt",
    ".pth",
    ".rar",
    ".safetensors",
    ".so",
    ".tar",
    ".tgz",
    ".whl",
    ".xz",
    ".zip",
}

RESTRICTED_DOCUMENT_SUFFIXES = {".pdf", ".xls", ".xlsx"}
ALLOWED_DOCUMENTS = {
    "examples/demo/specification.pdf",
    "examples/demo/expected-output.xlsx",
    "examples/demo/expected-quote.xlsx",
    "examples/kp-header-template.xlsx",
    "examples/kp-template.xlsx",
    "src/spec_pipeline/data/demo-specification.pdf",
    "src/spec_pipeline/data/kp-header-template.xlsx",
    "src/spec_pipeline/data/kp-template.xlsx",
    "spec_pipeline/data/demo-specification.pdf",
    "spec_pipeline/data/kp-header-template.xlsx",
    "spec_pipeline/data/kp-template.xlsx",
}

FORBIDDEN_FILENAMES = {
    ".env",
    "auth.json",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}

TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".cmd",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".lock",
    ".md",
    ".ndjson",
    ".ps1",
    ".py",
    ".rst",
    ".sh",
    ".svg",
    ".toml",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

PLACEHOLDER_USERS = {
    b"alice",
    b"bob",
    b"example",
    b"root",
    b"runner",
    b"user",
    b"username",
    b"you",
    b"your-name",
}

PERSONAL_PATH_PATTERNS = (
    re.compile(rb"/(?:Users|home)/(?P<user>[A-Za-z0-9._-]+)/"),
    re.compile(rb"[A-Za-z]:[\\/]+Users[\\/]+(?P<user>[^\\/\s\"']+)[\\/]", re.IGNORECASE),
    re.compile(rb"/mnt/[A-Za-z]/Users/(?P<user>[^/\s\"']+)/", re.IGNORECASE),
)

SECRET_PATTERNS = (
    (re.compile(rb"sk-[A-Za-z0-9_-]{20,}"), "possible API key"),
    (re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"), "possible GitHub token"),
    (re.compile(rb"AKIA[0-9A-Z]{16}"), "possible AWS access key"),
    (
        re.compile(b"-----BEGIN " + b"(?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
        "private key material",
    ),
)

EMAIL_PATTERN = re.compile(
    rb"(?<![A-Za-z0-9._%+-])(?P<local>[A-Za-z0-9._%+-]+)@(?P<domain>[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?![A-Za-z0-9.-])"
)
ALLOWED_EMAIL_DOMAINS = (
    b"example.com",
    b"example.invalid",
    b"example.net",
    b"example.org",
)
RUSSIAN_PHONE_PATTERN = re.compile(rb"(?<!\d)(?:\+7|8[\s().-]*800)(?:[\s().-]*\d){7,}(?!\d)")


@dataclass(frozen=True, order=True)
class Finding:
    source: str
    path: str
    reason: str


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check tracked/public files and optional source archives for private payloads."
    )
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        type=Path,
        help="Also inspect an sdist or wheel archive. Repeatable.",
    )
    return parser.parse_args(argv)


def _git_public_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    deleted_result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=D", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    )
    deleted_paths = {os.fsdecode(raw) for raw in deleted_result.stdout.split(b"\0") if raw}
    relative_paths = [Path(os.fsdecode(raw)) for raw in result.stdout.split(b"\0") if raw]
    return sorted(
        (
            path
            for path in relative_paths
            if path.as_posix() not in deleted_paths and (root / path).is_file()
        ),
        key=lambda path: path.as_posix().casefold(),
    )


def _path_findings(relative_path: PurePosixPath, size: int, source: str) -> list[Finding]:
    findings: list[Finding] = []
    display_path = relative_path.as_posix()
    folded_path = display_path.casefold()
    folded_name = relative_path.name.casefold()

    if relative_path.is_absolute() or ".." in relative_path.parts:
        return [Finding(source, display_path, "absolute or parent-traversing path")]

    if relative_path.parts and relative_path.parts[0] == "Examples":
        findings.append(
            Finding(source, display_path, "legacy private Examples directory is not public")
        )

    for prefix in PRIVATE_PREFIXES:
        matches_prefix = folded_path == prefix.rstrip("/").casefold() or folded_path.startswith(
            prefix.casefold()
        )
        if matches_prefix:
            findings.append(Finding(source, display_path, f"private path prefix: {prefix}"))

    if folded_name in FORBIDDEN_FILENAMES or (
        folded_name.startswith(".env.") and folded_name != ".env.example"
    ):
        findings.append(Finding(source, display_path, "credential or secret filename"))

    suffix = relative_path.suffix.casefold()
    if suffix in FORBIDDEN_BINARY_SUFFIXES:
        findings.append(Finding(source, display_path, f"forbidden binary/archive suffix: {suffix}"))
    if suffix in RESTRICTED_DOCUMENT_SUFFIXES and folded_path not in ALLOWED_DOCUMENTS:
        findings.append(Finding(source, display_path, "unapproved PDF/Excel payload"))
    if size > MAX_FILE_BYTES:
        findings.append(
            Finding(
                source,
                display_path,
                f"file is {size} bytes; public files are limited to {MAX_FILE_BYTES} bytes",
            )
        )

    return findings


def _content_findings(data: bytes, relative_path: PurePosixPath, source: str) -> list[Finding]:
    if relative_path.suffix.casefold() not in TEXT_SUFFIXES:
        return []

    findings: list[Finding] = []
    display_path = relative_path.as_posix()

    for pattern, reason in SECRET_PATTERNS:
        if pattern.search(data):
            findings.append(Finding(source, display_path, reason))

    for match in EMAIL_PATTERN.finditer(data):
        domain = match.group("domain").lower()
        if not any(
            domain == allowed or domain.endswith(b"." + allowed)
            for allowed in ALLOWED_EMAIL_DOMAINS
        ):
            address = match.group(0).decode("ascii", errors="replace")
            findings.append(
                Finding(source, display_path, f"non-placeholder email address: {address}")
            )

    for match in RUSSIAN_PHONE_PATTERN.finditer(data):
        number = match.group(0).decode("ascii", errors="replace")
        findings.append(Finding(source, display_path, f"possible real phone number: {number}"))

    is_test_fixture = bool(relative_path.parts) and relative_path.parts[0].casefold() == "tests"
    if not is_test_fixture:
        for pattern in PERSONAL_PATH_PATTERNS:
            for match in pattern.finditer(data):
                username = match.group("user").lower()
                if username not in PLACEHOLDER_USERS:
                    path = match.group(0).decode("utf-8", errors="replace")
                    findings.append(
                        Finding(source, display_path, f"machine-specific home path: {path}")
                    )

    return findings


def inspect_working_tree(root: Path) -> tuple[list[Finding], int]:
    findings: list[Finding] = []
    files = _git_public_files(root)
    total_size = 0

    for relative_path in files:
        absolute_path = root / relative_path
        if absolute_path.is_symlink():
            findings.append(Finding("working tree", relative_path.as_posix(), "symlink payload"))
            continue

        size = absolute_path.stat().st_size
        total_size += size
        public_path = PurePosixPath(relative_path.as_posix())
        findings.extend(_path_findings(public_path, size, "working tree"))
        if size <= MAX_FILE_BYTES:
            findings.extend(
                _content_findings(absolute_path.read_bytes(), public_path, "working tree")
            )

    if total_size > MAX_TOTAL_BYTES:
        findings.append(
            Finding(
                "working tree",
                "<all public files>",
                f"total size is {total_size} bytes; limit is {MAX_TOTAL_BYTES} bytes",
            )
        )

    return findings, len(files)


def _archive_root(names: list[str]) -> str | None:
    first_parts: set[str] = set()
    for name in names:
        path = PurePosixPath(name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            continue
        parts = tuple(part for part in path.parts if part not in {"", "."})
        if len(parts) < 2:
            return None
        first_parts.add(parts[0])
    return next(iter(first_parts)) if len(first_parts) == 1 else None


def _archive_member_path(name: str, root_name: str | None) -> PurePosixPath | None:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        return None
    parts = [part for part in path.parts if part not in {"", "."}]
    if root_name and parts and parts[0] == root_name:
        parts.pop(0)
    return PurePosixPath(*parts) if parts else None


def _inspect_tar(archive_path: Path) -> tuple[list[Finding], int]:
    source = f"archive {archive_path.name}"
    findings: list[Finding] = []
    checked = 0
    total_size = 0

    with tarfile.open(archive_path, mode="r:*") as archive:
        payload_members = [member for member in archive.getmembers() if not member.isdir()]
        root_name = _archive_root([member.name for member in payload_members])
        for member in payload_members:
            relative_path = _archive_member_path(member.name, root_name)
            if relative_path is None:
                findings.append(Finding(source, member.name, "unsafe archive member path"))
                continue
            if member.issym() or member.islnk():
                findings.append(Finding(source, relative_path.as_posix(), "archive link payload"))
                continue
            if not member.isfile():
                findings.append(
                    Finding(source, relative_path.as_posix(), "special archive payload")
                )
                continue

            checked += 1
            total_size += member.size
            findings.extend(_path_findings(relative_path, member.size, source))
            if member.size <= MAX_FILE_BYTES:
                extracted = archive.extractfile(member)
                if extracted is None:
                    findings.append(
                        Finding(source, relative_path.as_posix(), "archive member cannot be read")
                    )
                else:
                    findings.extend(_content_findings(extracted.read(), relative_path, source))

    if total_size > MAX_TOTAL_BYTES:
        findings.append(
            Finding(
                source,
                "<all archive files>",
                f"expanded size is {total_size} bytes; limit is {MAX_TOTAL_BYTES} bytes",
            )
        )
    return findings, checked


def _inspect_zip(archive_path: Path) -> tuple[list[Finding], int]:
    source = f"archive {archive_path.name}"
    findings: list[Finding] = []
    checked = 0
    total_size = 0

    with zipfile.ZipFile(archive_path) as archive:
        payload_members = [member for member in archive.infolist() if not member.is_dir()]
        root_name = _archive_root([member.filename for member in payload_members])
        for member in payload_members:
            relative_path = _archive_member_path(member.filename, root_name)
            if relative_path is None:
                findings.append(Finding(source, member.filename, "unsafe archive member path"))
                continue
            if stat.S_ISLNK(member.external_attr >> 16):
                findings.append(Finding(source, relative_path.as_posix(), "archive link payload"))
                continue

            checked += 1
            total_size += member.file_size
            findings.extend(_path_findings(relative_path, member.file_size, source))
            if member.file_size <= MAX_FILE_BYTES:
                findings.extend(_content_findings(archive.read(member), relative_path, source))

    if total_size > MAX_TOTAL_BYTES:
        findings.append(
            Finding(
                source,
                "<all archive files>",
                f"expanded size is {total_size} bytes; limit is {MAX_TOTAL_BYTES} bytes",
            )
        )
    return findings, checked


def inspect_archive(archive_path: Path) -> tuple[list[Finding], int]:
    if not archive_path.is_file():
        return [Finding("archive", str(archive_path), "archive does not exist")], 0
    if zipfile.is_zipfile(archive_path):
        return _inspect_zip(archive_path)
    if tarfile.is_tarfile(archive_path):
        return _inspect_tar(archive_path)
    return [Finding("archive", str(archive_path), "unsupported archive format")], 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        findings, checked = inspect_working_tree(PROJECT_ROOT)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"public-tree: unable to enumerate repository files: {exc}", file=sys.stderr)
        return 2

    archive_count = 0
    for archive_path in args.archive:
        archive_findings, archive_checked = inspect_archive(archive_path.resolve())
        findings.extend(archive_findings)
        archive_count += archive_checked

    unique_findings = sorted(set(findings))
    if unique_findings:
        print("public-tree: failed", file=sys.stderr)
        for finding in unique_findings:
            print(
                f"- [{finding.source}] {finding.path}: {finding.reason}",
                file=sys.stderr,
            )
        return 1

    print(f"public-tree: ok ({checked} repository files, {archive_count} archive files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
