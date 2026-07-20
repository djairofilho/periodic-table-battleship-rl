from __future__ import annotations

import numpy as np
import pytest

from periodic_table_battleship_rl.evaluation.schemas import EpisodeResult
from periodic_table_battleship_rl.experiments.belief_validation import (
    BayesianValidationProtocol,
    paired_seed_comparison,
)


def _result(run_id: str, seed: int, shots: int) -> EpisodeResult:
    return EpisodeResult(
        episode_id=f"{run_id}-{seed}",
        run_id=run_id,
        seed=seed,
        scenario="battleship",
        valid_cells=100,
        valid_shots=shots,
        invalid_attempts=0,
        hit_segments=17,
        sunk_ship_lengths=(5, 4, 3, 3, 2),
        won=True,
        truncated=False,
        auc_discovery=0.5,
        first_hit_shot=1,
        first_sunk_shot=3,
    )


def test_protocol_is_validation_only_and_exposes_fixed_seed_inventory() -> None:
    protocol = BayesianValidationProtocol(seed_start=100, seed_count=3)
    assert protocol.seeds == (100, 101, 102)
    assert protocol.as_dict()["split"] == "validation"
    with pytest.raises(ValueError, match="validation-only"):
        BayesianValidationProtocol(split="test")


def test_paired_comparison_aggregates_by_common_seed_and_is_reproducible() -> None:
    candidate = (_result("candidate", 11, 10), _result("candidate", 12, 20))
    reference = (_result("reference", 11, 15), _result("reference", 12, 25))
    first = paired_seed_comparison(
        candidate,
        reference,
        metric="valid_shots",
        direction="lower",
        bootstrap_resamples=100,
        rng=np.random.default_rng(1),
    )
    second = paired_seed_comparison(
        candidate,
        reference,
        metric="valid_shots",
        direction="lower",
        bootstrap_resamples=100,
        rng=np.random.default_rng(1),
    )
    assert first == second
    assert first["candidate_minus_reference_mean"] == -5.0
    assert first["improves_reference_at_95"]


def test_paired_comparison_rejects_different_seed_inventory() -> None:
    with pytest.raises(ValueError, match="identical seed inventory"):
        paired_seed_comparison(
            (_result("candidate", 1, 10),),
            (_result("reference", 2, 20),),
            metric="valid_shots",
            direction="lower",
            bootstrap_resamples=10,
            rng=np.random.default_rng(1),
        )
