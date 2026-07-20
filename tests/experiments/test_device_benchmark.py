"""Tests for the CUDA gate and deterministic benchmark report shape."""

from __future__ import annotations

import pytest

from periodic_table_battleship_rl.experiments.device_benchmark import (
    BENCHMARK_ARCHITECTURES,
    CudaReadinessError,
    DeviceBenchmarkConfig,
    benchmark_architecture,
    require_cuda,
)
from periodic_table_battleship_rl.topology import BATTLESHIP


class _UnavailableCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _CpuOnlyTorch:
    __version__ = "test+cpu"

    class version:
        cuda = None

    cuda = _UnavailableCuda()


def test_cuda_gate_rejects_cpu_only_torch_without_fallback() -> None:
    with pytest.raises(CudaReadinessError, match="never falls back to CPU"):
        require_cuda(_CpuOnlyTorch())


@pytest.mark.parametrize("architecture", BENCHMARK_ARCHITECTURES)
def test_cpu_workload_reports_throughput_and_no_fictitious_vram(architecture: str) -> None:
    pytest.importorskip("torch")
    result = benchmark_architecture(
        architecture,
        BATTLESHIP,
        DeviceBenchmarkConfig(batch_size=2, warmup_iterations=1, measured_iterations=1),
        device="cpu",
    )

    assert result["device"] == "cpu"
    assert result["parameter_count"] > 0
    assert result["iterations_per_second"] > 0
    assert result["examples_per_second"] > 0
    assert result["peak_allocated_bytes"] is None


def test_unknown_device_and_architecture_are_rejected() -> None:
    config = DeviceBenchmarkConfig()
    with pytest.raises(ValueError, match="unknown architecture"):
        benchmark_architecture("transformer", BATTLESHIP, config, device="cpu")
    with pytest.raises(ValueError, match="device must be"):
        benchmark_architecture("cnn", BATTLESHIP, config, device="mps")
