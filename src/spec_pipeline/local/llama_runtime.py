from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LlamaCppRuntimeInfo:
    installed: bool
    version: str | None = None
    supports_gpu_offload: bool = False
    error: str | None = None

    @property
    def cuda_ready(self) -> bool:
        return self.installed and self.supports_gpu_offload

    def to_dict(self) -> dict[str, object]:
        return {
            "installed": self.installed,
            "version": self.version,
            "supports_gpu_offload": self.supports_gpu_offload,
            "cuda_ready": self.cuda_ready,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class LlamaCppCudaLibraryLinkInfo:
    llama_lib_dir: str | None
    nvidia_lib_dirs: tuple[str, ...]
    linked: tuple[str, ...]
    copied: tuple[str, ...]
    skipped: tuple[str, ...]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.llama_lib_dir is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "llama_lib_dir": self.llama_lib_dir,
            "nvidia_lib_dirs": list(self.nvidia_lib_dirs),
            "linked": list(self.linked),
            "copied": list(self.copied),
            "skipped": list(self.skipped),
            "error": self.error,
        }


def link_llama_cpp_cuda_libraries(*, prefix: Path | None = None) -> LlamaCppCudaLibraryLinkInfo:
    """Expose PyPI NVIDIA CUDA libraries beside libllama.so for reliable dlopen."""
    root = Path(prefix or sys.prefix)
    site_roots = sorted((root / "lib").glob("python*/site-packages"))
    llama_lib_dirs = [path for site in site_roots for path in (site / "llama_cpp").glob("lib")]
    if not llama_lib_dirs:
        return LlamaCppCudaLibraryLinkInfo(
            llama_lib_dir=None,
            nvidia_lib_dirs=(),
            linked=(),
            copied=(),
            skipped=(),
            error="llama_cpp/lib directory was not found in this Python environment",
        )

    llama_lib_dir = llama_lib_dirs[0]
    nvidia_lib_dirs = sorted(
        path for site in site_roots for path in (site / "nvidia").glob("*/lib") if path.is_dir()
    )
    if not nvidia_lib_dirs:
        return LlamaCppCudaLibraryLinkInfo(
            llama_lib_dir=str(llama_lib_dir),
            nvidia_lib_dirs=(),
            linked=(),
            copied=(),
            skipped=(),
            error="NVIDIA CUDA runtime library directories were not found in this Python environment",
        )

    linked: list[str] = []
    copied: list[str] = []
    skipped: list[str] = []
    for source_dir in nvidia_lib_dirs:
        for source in sorted(source_dir.glob("*.so*")):
            target = llama_lib_dir / source.name
            if target.exists():
                skipped.append(source.name)
                continue
            try:
                target.symlink_to(source)
                linked.append(source.name)
            except OSError:
                shutil.copy2(source, target)
                copied.append(source.name)

    return LlamaCppCudaLibraryLinkInfo(
        llama_lib_dir=str(llama_lib_dir),
        nvidia_lib_dirs=tuple(str(path) for path in nvidia_lib_dirs),
        linked=tuple(linked),
        copied=tuple(copied),
        skipped=tuple(skipped),
    )


def inspect_llama_cpp_runtime() -> LlamaCppRuntimeInfo:
    try:
        import llama_cpp  # type: ignore[import-not-found]
    except Exception as exc:
        return LlamaCppRuntimeInfo(installed=False, error=str(exc))

    version = str(getattr(llama_cpp, "__version__", "unknown"))
    try:
        supports_gpu_offload = _llama_supports_gpu_offload(llama_cpp)
    except Exception as exc:
        return LlamaCppRuntimeInfo(
            installed=True,
            version=version,
            supports_gpu_offload=False,
            error=str(exc),
        )
    return LlamaCppRuntimeInfo(
        installed=True,
        version=version,
        supports_gpu_offload=supports_gpu_offload,
    )


def _llama_supports_gpu_offload(llama_cpp_module: Any) -> bool:
    checker = getattr(llama_cpp_module, "llama_supports_gpu_offload", None)
    if callable(checker):
        return bool(checker())

    low_level = getattr(llama_cpp_module, "llama_cpp", None)
    checker = getattr(low_level, "llama_supports_gpu_offload", None)
    if callable(checker):
        return bool(checker())

    try:
        from llama_cpp import llama_cpp as imported_low_level  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "Cannot inspect llama.cpp GPU offload support: low-level module is unavailable"
        ) from exc

    checker = getattr(imported_low_level, "llama_supports_gpu_offload", None)
    if callable(checker):
        return bool(checker())
    raise RuntimeError("llama.cpp runtime does not expose llama_supports_gpu_offload")
