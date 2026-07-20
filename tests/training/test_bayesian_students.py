"""Regression coverage for public Bayesian neural distillation."""

from __future__ import annotations

import numpy as np
import pytest

from periodic_table_battleship_rl.topology import BATTLESHIP
from periodic_table_battleship_rl.training.bayesian_distillation import (
    BayesianDemonstrationConfig,
    generate_bayesian_demonstrations,
    load_bayesian_demonstrations,
)
from periodic_table_battleship_rl.training.bayesian_students import (
    BayesianStudentTrainingConfig,
    build_bayesian_student,
    evaluate_bayesian_student,
    load_bayesian_student_policy,
    teacher_action_agreement,
    train_bayesian_student,
)


@pytest.mark.parametrize("architecture", ("cnn", "gnn"))
def test_public_students_train_load_and_respect_masks(
    tmp_path, architecture: str
) -> None:
    pytest.importorskip("torch")
    dataset = generate_bayesian_demonstrations(
        BATTLESHIP,
        BayesianDemonstrationConfig(
            dataset_id="teacher",
            seeds=(9711,),
            output_directory=tmp_path,
            sample_count=2,
            max_nodes_per_sample=4_096,
        ),
    )
    artifact = train_bayesian_student(
        BATTLESHIP,
        BayesianStudentTrainingConfig(
            run_id=f"student-{architecture}",
            architecture=architecture,  # type: ignore[arg-type]
            seed=3,
            dataset_path=dataset.data_path,
            checkpoint_directory=tmp_path,
            epochs=2,
            batch_size=32,
            hidden_dim=8,
        ),
    )
    policy = load_bayesian_student_policy(BATTLESHIP, artifact.checkpoint_path)
    demos = load_bayesian_demonstrations(dataset.data_path)
    action_mask = np.zeros(BATTLESHIP.action_count, dtype=np.bool_)
    action_mask[7] = True

    assert artifact.metadata_path.exists()
    assert 0.0 <= artifact.training_action_agreement <= 1.0
    assert teacher_action_agreement(policy, demos) == pytest.approx(
        artifact.training_action_agreement
    )
    assert policy.select_action(demos.observations[0], action_mask) == 7


def test_cnn_student_output_matches_canvas_action_space() -> None:
    torch = pytest.importorskip("torch")
    student = build_bayesian_student(
        BATTLESHIP, architecture="cnn", observation_channels=4, hidden_dim=8
    )
    assert student(torch.zeros((2, 4, 10, 18))).shape == (2, BATTLESHIP.action_count)


def test_student_validation_refuses_one_seed_and_never_accepts_test_split(
    tmp_path,
) -> None:
    pytest.importorskip("torch")
    dataset = generate_bayesian_demonstrations(
        BATTLESHIP,
        BayesianDemonstrationConfig(
            dataset_id="teacher",
            seeds=(9712,),
            output_directory=tmp_path,
            sample_count=2,
            max_nodes_per_sample=4_096,
        ),
    )
    artifact = train_bayesian_student(
        BATTLESHIP,
        BayesianStudentTrainingConfig(
            run_id="student",
            architecture="cnn",
            seed=4,
            dataset_path=dataset.data_path,
            checkpoint_directory=tmp_path,
            epochs=1,
            batch_size=32,
            hidden_dim=8,
        ),
    )
    policy = load_bayesian_student_policy(BATTLESHIP, artifact.checkpoint_path)
    with pytest.raises(ValueError, match="at least two"):
        evaluate_bayesian_student(BATTLESHIP, policy, seeds=(9713,))
