from __future__ import annotations

import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DependencyRunner = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class LocalDependencyInstallResult:
    ok: bool
    platform: str
    extras: tuple[str, ...]
    command: list[str]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    skipped_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "platform": self.platform,
            "extras": list(self.extras),
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "skipped_reason": self.skipped_reason,
        }


def local_dependency_extras_for_platform(
    system: str | None = None, *, include_paddle: bool = True
) -> tuple[str, ...]:
    resolved = system or platform.system()
    if resolved == "Darwin":
        extras = ["local-models", "macos-local"]
        if include_paddle:
            extras.append("paddleocr-vl")
        return tuple(extras)
    return ()


def install_local_dependencies(
    *,
    project_root: Path,
    system: str | None = None,
    include_paddle: bool = True,
    runner: DependencyRunner = subprocess.run,
) -> LocalDependencyInstallResult:
    resolved_system = system or platform.system()
    extras = local_dependency_extras_for_platform(resolved_system, include_paddle=include_paddle)
    if not extras:
        return LocalDependencyInstallResult(
            ok=False,
            platform=resolved_system,
            extras=(),
            command=[],
            skipped_reason=(
                "setup-local dependency installation is supported on macOS; "
                "use setup-wsl for Windows/WSL2."
            ),
        )

    command = [sys.executable, "-m", "pip", "install", "-e", f".[{','.join(extras)}]"]
    completed = runner(
        command,
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return LocalDependencyInstallResult(
        ok=int(completed.returncode) == 0,
        platform=resolved_system,
        extras=extras,
        command=command,
        returncode=int(completed.returncode),
        stdout=str(completed.stdout),
        stderr=str(completed.stderr),
    )
