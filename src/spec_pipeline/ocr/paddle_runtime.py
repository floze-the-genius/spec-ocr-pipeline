from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PaddleRuntimeInfo:
    installed: bool
    version: str | None = None
    compiled_with_cuda: bool = False
    cuda_device_count: int = 0
    selected_device: str | None = None
    smoke_test_ok: bool | None = None
    error: str | None = None

    @property
    def cuda_ready(self) -> bool:
        return (
            self.installed
            and self.compiled_with_cuda
            and self.cuda_device_count > 0
            and self.smoke_test_ok is not False
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "installed": self.installed,
            "version": self.version,
            "compiled_with_cuda": self.compiled_with_cuda,
            "cuda_device_count": self.cuda_device_count,
            "selected_device": self.selected_device,
            "smoke_test_ok": self.smoke_test_ok,
            "cuda_ready": self.cuda_ready,
            "error": self.error,
        }


def inspect_paddle_runtime(*, device: str = "gpu:0", smoke_test: bool = True) -> PaddleRuntimeInfo:
    try:
        import paddle  # type: ignore[import-not-found]
    except Exception as exc:
        return PaddleRuntimeInfo(installed=False, error=str(exc))

    version = str(getattr(paddle, "__version__", "unknown"))
    compiled_with_cuda = _safe_bool_call(paddle, "is_compiled_with_cuda")
    device_count = 0
    selected_device: str | None = None
    smoke_test_ok: bool | None = None
    error: str | None = None

    if compiled_with_cuda:
        try:
            cuda = getattr(getattr(paddle, "device", None), "cuda", None)
            if cuda is not None and hasattr(cuda, "device_count"):
                device_count = int(cuda.device_count())
        except Exception as exc:
            error = f"cannot query CUDA device count: {exc}"

    if compiled_with_cuda and device_count > 0 and smoke_test:
        try:
            set_device = getattr(paddle, "set_device")
            get_device = getattr(paddle, "get_device", None)
            set_device(device)
            selected_device = str(get_device()) if callable(get_device) else device
            # Force a tiny CUDA op so missing runtime libraries fail before OCR starts.
            paddle.to_tensor([1.0], dtype="float32").numpy()
            smoke_test_ok = True
        except Exception as exc:
            smoke_test_ok = False
            error = str(exc)

    return PaddleRuntimeInfo(
        installed=True,
        version=version,
        compiled_with_cuda=compiled_with_cuda,
        cuda_device_count=device_count,
        selected_device=selected_device,
        smoke_test_ok=smoke_test_ok,
        error=error,
    )


def ensure_paddle_cuda_available(*, device: str = "gpu:0") -> PaddleRuntimeInfo:
    info = inspect_paddle_runtime(device=device, smoke_test=True)
    if not info.installed:
        raise RuntimeError("PaddlePaddle is not installed. Install paddlepaddle-gpu for CUDA OCR.")
    if not info.compiled_with_cuda:
        raise RuntimeError(
            "PaddlePaddle is installed without CUDA. Remove CPU paddlepaddle and install "
            "paddlepaddle-gpu==3.2.1 from the cu126 index."
        )
    if info.cuda_device_count < 1:
        raise RuntimeError(
            "PaddlePaddle CUDA build is installed, but no CUDA device is visible in WSL. "
            "Check NVIDIA driver, WSL GPU support, and nvidia-smi inside WSL."
        )
    if info.smoke_test_ok is False:
        raise RuntimeError(f"PaddlePaddle CUDA smoke test failed: {info.error}")
    return info


def wants_cuda_device(device: str | None) -> bool:
    if device is None:
        return False
    normalized = device.strip().lower()
    return normalized == "gpu" or normalized.startswith("gpu:") or normalized.startswith("cuda")


def _safe_bool_call(module: Any, name: str) -> bool:
    func = getattr(module, name, None)
    if not callable(func):
        return False
    try:
        return bool(func())
    except Exception:
        return False
