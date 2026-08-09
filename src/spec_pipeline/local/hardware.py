from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(slots=True)
class HardwareReport:
    os: str
    machine: str
    python: str
    nvidia_smi: bool
    nvidia_gpu: str | None
    mlx_available: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "os": self.os,
            "machine": self.machine,
            "python": self.python,
            "nvidia_smi": self.nvidia_smi,
            "nvidia_gpu": self.nvidia_gpu,
            "mlx_available": self.mlx_available,
        }


def collect_hardware_report() -> HardwareReport:
    nvidia_smi_path = shutil.which("nvidia-smi")
    gpu_name = None
    if nvidia_smi_path:
        try:
            completed = subprocess.run(
                [
                    nvidia_smi_path,
                    "--query-gpu=name",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if completed.returncode == 0:
                gpu_name = (
                    completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else None
                )
        except (OSError, subprocess.TimeoutExpired):
            gpu_name = None

    return HardwareReport(
        os=f"{platform.system()} {platform.release()}",
        machine=platform.machine(),
        python=platform.python_version(),
        nvidia_smi=nvidia_smi_path is not None,
        nvidia_gpu=gpu_name,
        mlx_available=is_module_available("mlx"),
    )


def is_module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, OSError, ValueError):
        return False
