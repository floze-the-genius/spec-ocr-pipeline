from __future__ import annotations

from pathlib import Path

from spec_pipeline.local.deps import (
    install_local_dependencies,
    local_dependency_extras_for_platform,
)


def test_local_dependency_extras_are_macos_only() -> None:
    assert local_dependency_extras_for_platform("Darwin") == (
        "local-models",
        "macos-local",
        "paddleocr-vl",
    )
    assert local_dependency_extras_for_platform("Darwin", include_paddle=False) == (
        "local-models",
        "macos-local",
    )
    assert local_dependency_extras_for_platform("Windows") == ()
    assert local_dependency_extras_for_platform("Linux") == ()


def test_install_local_dependencies_runs_pip_for_macos(tmp_path: Path) -> None:
    calls = []

    class Completed:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return Completed()

    result = install_local_dependencies(project_root=tmp_path, system="Darwin", runner=runner)

    assert result.ok is True
    assert result.extras == ("local-models", "macos-local", "paddleocr-vl")
    assert calls[0][0][-1] == ".[local-models,macos-local,paddleocr-vl]"
    assert calls[0][1]["cwd"] == str(tmp_path)


def test_install_local_dependencies_points_windows_to_setup_wsl(tmp_path: Path) -> None:
    result = install_local_dependencies(project_root=tmp_path, system="Windows")

    assert result.ok is False
    assert "setup-wsl" in str(result.skipped_reason)
    assert result.command == []
