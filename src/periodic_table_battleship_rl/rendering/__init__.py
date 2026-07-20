"""Deterministic, privacy-aware renderers for benchmark diagnostics."""

from periodic_table_battleship_rl.rendering.attack import (
    AttackEpisodeTrace,
    AttackFrame,
    AttackTraceRecorder,
    AttackTraceStep,
    capture_attack_frame,
    render_attack_frame,
    render_episode_trace,
    render_topology,
)

__all__ = [
    "AttackEpisodeTrace",
    "AttackFrame",
    "AttackTraceRecorder",
    "AttackTraceStep",
    "capture_attack_frame",
    "render_attack_frame",
    "render_episode_trace",
    "render_topology",
]
