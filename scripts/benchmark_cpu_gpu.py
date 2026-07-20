"""Run the paired CPU/CUDA neural-training microbenchmark for issue #67.

Run this from the isolated CUDA environment only.  The command exits nonzero
instead of producing a misleading CPU fallback when CUDA is unavailable.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from periodic_table_battleship_rl.experiments.device_benchmark import (
    CudaReadinessError,
    DeviceBenchmarkConfig,
    run_cpu_cuda_benchmark,
)
from periodic_table_battleship_rl.topology import get_topology


ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="periodic-table-battleship")
    parser.add_argument("--seed", type=int, default=8601)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--measured-iterations", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "v0.6-cuda-benchmark" / "cpu-gpu-microbenchmark.json",
    )
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()
    except subprocess.CalledProcessError:
        return "unavailable"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    arguments = _arguments()
    config = DeviceBenchmarkConfig(
        seed=arguments.seed,
        batch_size=arguments.batch_size,
        warmup_iterations=arguments.warmup_iterations,
        measured_iterations=arguments.measured_iterations,
        cpu_threads=arguments.cpu_threads,
    )
    try:
        report = run_cpu_cuda_benchmark(get_topology(arguments.scenario), config)
    except CudaReadinessError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    report["provenance"] = {
        "git_commit": _git_commit(),
        "uv_lock_sha256": _sha256(ROOT / "uv.lock"),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python_executable": sys.executable,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
