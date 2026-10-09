import platform
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import psutil


@dataclass(frozen=True, slots=True)
class GPUInfo:
    name: str
    vram_total_mb: int


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    python_version: str
    os: str
    os_version: str
    architecture: str
    cpu: str
    logical_cpu_count: int
    ram_total_mb: int
    gpu: str | None
    vram_total_mb: int | None
    cuda_available: bool


def query_nvidia_gpu() -> GPUInfo | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first_line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    if not first_line or "," not in first_line:
        return None
    name, raw_memory = (part.strip() for part in first_line.rsplit(",", maxsplit=1))
    try:
        memory = int(raw_memory)
    except ValueError:
        return None
    return GPUInfo(name=name, vram_total_mb=memory)


def collect_hardware_info(
    gpu_probe: Callable[[], GPUInfo | None] = query_nvidia_gpu,
) -> HardwareInfo:
    gpu = gpu_probe()
    cpu_name = platform.processor().strip() or platform.machine()
    return HardwareInfo(
        python_version=platform.python_version(),
        os=platform.system(),
        os_version=platform.version(),
        architecture=platform.machine(),
        cpu=cpu_name,
        logical_cpu_count=psutil.cpu_count(logical=True) or 1,
        ram_total_mb=round(psutil.virtual_memory().total / (1024 * 1024)),
        gpu=gpu.name if gpu else None,
        vram_total_mb=gpu.vram_total_mb if gpu else None,
        cuda_available=gpu is not None,
    )
