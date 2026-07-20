"""Reproducible non-learning policies for Battleship attack experiments."""

from .baselines import (
    NoValidActionError,
    hunt_target_action,
    random_masked_action,
)

__all__ = [
    "NoValidActionError",
    "hunt_target_action",
    "random_masked_action",
]
