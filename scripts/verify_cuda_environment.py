"""Emit an auditable CUDA readiness report for the isolated training venv."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _nvidia_smi() -> dict[str, Any]:
    """Return the driver inventory when NVIDIA's command line tool is present."""
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, TimeoutError) as error:
        return {"available": False, "error": str(error)}

    devices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {"available": True, "devices": devices}


def build_report() -> dict[str, Any]:
    """Collect only local, reproducible hardware and PyTorch facts."""
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "nvidia_smi": _nvidia_smi(),
    }
    try:
        import torch
    except ImportError as error:
        report.update(
            {
                "torch_installed": False,
                "cuda_ready": False,
                "error": str(error),
            }
        )
        return report

    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is None:
        report.update(
            {
                "torch_installed": False,
                "cuda_ready": False,
                "error": "PyTorch import is incomplete: torch.cuda is unavailable.",
            }
        )
        return report

    cuda_ready = cuda_module.is_available()
    report.update(
        {
            "torch_installed": True,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_ready": cuda_ready,
            "cuda_device_count": cuda_module.device_count(),
        }
    )
    if cuda_ready:
        device = torch.device("cuda:0")
        sample = torch.arange(1024, device=device, dtype=torch.float32)
        report["cuda_device"] = cuda_module.get_device_name(device)
        report["cuda_capability"] = list(cuda_module.get_device_capability(device))
        report["cuda_smoke_sum"] = float((sample * sample).sum().item())
        report["cuda_peak_allocated_bytes"] = cuda_module.max_memory_allocated(device)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional UTF-8 JSON output path.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit nonzero unless PyTorch can execute the CUDA smoke operation.",
    )
    args = parser.parse_args()
    report = build_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if report["cuda_ready"] or not args.require_cuda else 1


if __name__ == "__main__":
    raise SystemExit(main())
