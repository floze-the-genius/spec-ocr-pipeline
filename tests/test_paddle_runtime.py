from __future__ import annotations

import sys
import types

from spec_pipeline.ocr.paddle_runtime import inspect_paddle_runtime, wants_cuda_device


def test_wants_cuda_device_detects_gpu_aliases() -> None:
    assert wants_cuda_device("gpu") is True
    assert wants_cuda_device("gpu:0") is True
    assert wants_cuda_device("cuda:0") is True
    assert wants_cuda_device("cpu") is False
    assert wants_cuda_device(None) is False


def test_inspect_paddle_runtime_reports_cpu_build(monkeypatch) -> None:
    paddle = types.ModuleType("paddle")
    setattr(paddle, "__version__", "3.2.1")
    setattr(paddle, "is_compiled_with_cuda", lambda: False)
    monkeypatch.setitem(sys.modules, "paddle", paddle)

    info = inspect_paddle_runtime()

    assert info.installed is True
    assert info.version == "3.2.1"
    assert info.compiled_with_cuda is False
    assert info.cuda_ready is False


def test_inspect_paddle_runtime_smoke_tests_cuda(monkeypatch) -> None:
    paddle = types.ModuleType("paddle")
    setattr(paddle, "__version__", "3.2.1")
    setattr(paddle, "is_compiled_with_cuda", lambda: True)
    setattr(paddle, "get_device", lambda: "gpu:0")

    class FakeCuda:
        @staticmethod
        def device_count() -> int:
            return 1

    class FakeTensor:
        def numpy(self):
            return [1.0]

    setattr(paddle, "device", types.SimpleNamespace(cuda=FakeCuda()))
    setattr(paddle, "set_device", lambda device: None)
    setattr(paddle, "to_tensor", lambda values, dtype=None: FakeTensor())
    monkeypatch.setitem(sys.modules, "paddle", paddle)

    info = inspect_paddle_runtime(device="gpu:0")

    assert info.cuda_ready is True
    assert info.selected_device == "gpu:0"
    assert info.smoke_test_ok is True
