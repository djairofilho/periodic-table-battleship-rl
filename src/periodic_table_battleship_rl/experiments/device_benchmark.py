"""Reproducible CPU/CUDA neural-training microbenchmarks.

This module deliberately measures one forward/backward/Adam update for the
CNN, DQN and GNN architectures used by this project.  It does *not* claim to
measure end-to-end environment throughput or agent quality; those include
Python environment stepping, rollout collection and evaluation and belong to
their own campaign reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import platform
import time
from typing import Any, Callable

from periodic_table_battleship_rl.topology import Topology
from periodic_table_battleship_rl.training.dqn import _MlpQNetworkFactory, _require_torch
from periodic_table_battleship_rl.training.gnn import TopologyGraphQNetwork


DEVICE_BENCHMARK_SCHEMA_VERSION = "device-training-microbenchmark-v1"
BENCHMARK_ARCHITECTURES = ("cnn", "dqn", "gnn")


class CudaReadinessError(RuntimeError):
    """Raised before any benchmark can accidentally fall back to CPU."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceBenchmarkConfig:
    """Frozen workload shared by the CPU and CUDA halves of one measurement."""

    seed: int = 8601
    batch_size: int = 32
    warmup_iterations: int = 5
    measured_iterations: int = 20
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for field_name in (
            "batch_size",
            "warmup_iterations",
            "measured_iterations",
            "cpu_threads",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")

    def public_dict(self) -> dict[str, int]:
        return asdict(self)


def require_cuda(torch: Any) -> None:
    """Reject CPU-only wheels and unavailable drivers with an actionable error."""
    if not bool(torch.cuda.is_available()):
        version = getattr(torch, "__version__", "unknown")
        runtime = getattr(getattr(torch, "version", None), "cuda", None)
        raise CudaReadinessError(
            "CUDA is not ready for the CPU/GPU benchmark: "
            f"torch={version!s}, torch.version.cuda={runtime!s}, "
            "torch.cuda.is_available()=False. Run "
            "scripts/verify_cuda_environment.py --require-cuda from the isolated "
            ".venv-cuda first; this benchmark never falls back to CPU."
        )


def _cnn_network(torch: Any, topology: Topology) -> Any:
    """Build the v0.5 CNN backbone plus a trainable action head.

    The convolution widths and adaptive pool match ``training.cnn``.  The
    action head turns the feature extractor into a self-contained update
    workload without depending on rollout collection from Stable-Baselines3.
    """

    class SpatialCnnQNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(4, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((2, 3)),
                torch.nn.Flatten(),
                torch.nn.Linear(64 * 2 * 3, 128),
                torch.nn.ReLU(),
            )
            self.head = torch.nn.Linear(128, topology.action_count)

        def forward(self, observations: Any) -> Any:
            return self.head(self.features(observations))

    return SpatialCnnQNetwork()


def _network_builders(topology: Topology) -> dict[str, Callable[[Any], Any]]:
    return {
        "cnn": lambda torch: _cnn_network(torch, topology),
        "dqn": lambda _torch: _MlpQNetworkFactory.create(
            (4, topology.rows, topology.columns), topology.action_count, hidden_dim=128
        ),
        "gnn": lambda _torch: TopologyGraphQNetwork.create(
            topology, observation_channels=4, hidden_dim=64, message_passing_steps=2
        ),
    }


def _cuda_hardware(torch: Any) -> dict[str, object]:
    if not bool(torch.cuda.is_available()):
        return {"available": False}
    properties = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": int(properties.total_memory),
        "runtime_version": torch.version.cuda,
    }


def _one_update(model: Any, observations: Any, targets: Any, optimizer: Any) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = (model(observations) - targets).square().mean()
    loss.backward()
    optimizer.step()


def _synchronize(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def benchmark_architecture(
    architecture: str,
    topology: Topology,
    config: DeviceBenchmarkConfig,
    *,
    device: str,
) -> dict[str, object]:
    """Measure a deterministic update workload on exactly one device."""
    if architecture not in BENCHMARK_ARCHITECTURES:
        raise ValueError(f"unknown architecture: {architecture!r}")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")

    torch = _require_torch()
    if device == "cuda":
        require_cuda(torch)
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    # Inputs are generated on CPU then copied.  That keeps the data and initial
    # weights equal across CPU/CUDA while excluding setup and transfers from the
    # timed neural-update region.
    observations_cpu = torch.randn(
        (config.batch_size, 4, topology.rows, topology.columns), dtype=torch.float32
    )
    targets_cpu = torch.randn(
        (config.batch_size, topology.action_count), dtype=torch.float32
    )
    model = _network_builders(topology)[architecture](torch).to(device)
    observations = observations_cpu.to(device)
    targets = targets_cpu.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)
        baseline_allocated = int(torch.cuda.memory_allocated(0))
    else:
        baseline_allocated = None

    for _ in range(config.warmup_iterations):
        _one_update(model, observations, targets, optimizer)
    _synchronize(torch, device)

    start = time.perf_counter()
    for _ in range(config.measured_iterations):
        _one_update(model, observations, targets, optimizer)
    _synchronize(torch, device)
    duration_seconds = time.perf_counter() - start
    if duration_seconds <= 0:
        raise RuntimeError("benchmark clock returned a non-positive duration")

    result: dict[str, object] = {
        "device": device,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "duration_seconds": duration_seconds,
        "iterations": config.measured_iterations,
        "iterations_per_second": config.measured_iterations / duration_seconds,
        "examples_per_second": config.batch_size * config.measured_iterations / duration_seconds,
    }
    if device == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(0))
        result.update(
            {
                "baseline_allocated_bytes": baseline_allocated,
                "peak_allocated_bytes": peak_allocated,
                "incremental_peak_allocated_bytes": peak_allocated - int(baseline_allocated),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            }
        )
    else:
        result["baseline_allocated_bytes"] = None
        result["peak_allocated_bytes"] = None
        result["incremental_peak_allocated_bytes"] = None
        result["peak_reserved_bytes"] = None
    return result


def run_cpu_cuda_benchmark(
    topology: Topology,
    config: DeviceBenchmarkConfig,
) -> dict[str, object]:
    """Run the complete paired benchmark, refusing partial CPU-only reports."""
    torch = _require_torch()
    require_cuda(torch)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    try:
        architectures: dict[str, dict[str, object]] = {}
        for architecture in BENCHMARK_ARCHITECTURES:
            cpu = benchmark_architecture(architecture, topology, config, device="cpu")
            cuda = benchmark_architecture(architecture, topology, config, device="cuda")
            architectures[architecture] = {
                "cpu": cpu,
                "cuda": cuda,
                "cuda_speedup": float(cuda["iterations_per_second"])
                / float(cpu["iterations_per_second"]),
            }
    finally:
        torch.set_num_threads(original_threads)

    return {
        "schema_version": DEVICE_BENCHMARK_SCHEMA_VERSION,
        "kind": "paired-neural-training-microbenchmark",
        "scope": (
            "Forward/backward/Adam updates only; excludes environment stepping, "
            "rollout collection, evaluation and disk I/O."
        ),
        "topology": {
            "name": topology.name,
            "rows": topology.rows,
            "columns": topology.columns,
            "action_count": topology.action_count,
            "valid_cell_count": topology.valid_cell_count,
        },
        "configuration": config.public_dict(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or "unreported",
            "torch_version": torch.__version__,
            "cuda": _cuda_hardware(torch),
        },
        "architectures": architectures,
    }
