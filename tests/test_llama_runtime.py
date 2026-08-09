from __future__ import annotations

import sys
import types

from spec_pipeline.local.llama_runtime import (
    inspect_llama_cpp_runtime,
    link_llama_cpp_cuda_libraries,
)


def test_inspect_llama_cpp_runtime_reports_gpu_offload(monkeypatch) -> None:
    module = types.ModuleType("llama_cpp")
    setattr(module, "__version__", "0.3.21")
    setattr(module, "llama_supports_gpu_offload", lambda: True)
    monkeypatch.setitem(sys.modules, "llama_cpp", module)

    info = inspect_llama_cpp_runtime()

    assert info.installed is True
    assert info.version == "0.3.21"
    assert info.cuda_ready is True


def test_inspect_llama_cpp_runtime_reports_cpu_wheel(monkeypatch) -> None:
    module = types.ModuleType("llama_cpp")
    low_level = types.SimpleNamespace(llama_supports_gpu_offload=lambda: False)
    setattr(module, "__version__", "0.3.21")
    setattr(module, "llama_cpp", low_level)
    monkeypatch.setitem(sys.modules, "llama_cpp", module)

    info = inspect_llama_cpp_runtime()

    assert info.installed is True
    assert info.cuda_ready is False
    assert info.supports_gpu_offload is False


def test_link_llama_cpp_cuda_libraries_exposes_nvidia_libs(tmp_path) -> None:
    site_packages = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    llama_lib = site_packages / "llama_cpp" / "lib"
    cuda_lib = site_packages / "nvidia" / "cuda_runtime" / "lib"
    llama_lib.mkdir(parents=True)
    cuda_lib.mkdir(parents=True)
    source = cuda_lib / "libcudart.so.12"
    source.write_text("cuda", encoding="utf-8")

    info = link_llama_cpp_cuda_libraries(prefix=tmp_path / "venv")

    assert info.ok is True
    assert info.llama_lib_dir == str(llama_lib)
    assert (llama_lib / "libcudart.so.12").exists()
    assert "libcudart.so.12" in set(info.linked) | set(info.copied)
