from __future__ import annotations

import json

import pytest

from periodic_table_battleship_rl.protocol import (
    PROTOCOL_VERSION,
    ArtifactProvenance,
    ArtifactRecord,
    CandidateRegistration,
    CheckpointPlan,
    ExperimentProtocol,
    PromotionDecision,
    SeedInventory,
    TestConfirmation,
)


def _inventory() -> SeedInventory:
    return SeedInventory(
        train=(101, 102),
        validation=(201, 202),
        test=(301, 302),
        demonstration=(401,),
    )


def _registration() -> CandidateRegistration:
    return CandidateRegistration(
        record_id="selection-e3-cnn-001",
        candidate_id="ppo-cnn-v1",
        control_id="ppo-mlp-v0.4",
        selected_checkpoint_step=20_000,
        metric="valid_shots",
        selection_seeds=(201, 202),
    )


def _artifact() -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id="learning-curve",
        kind="figure",
        relative_path="artifacts/e3/learning-curve.png",
        sha256="c" * 64,
        provenance=ArtifactProvenance(
            run_id="e3-cnn-battleship-seed101",
            git_commit="a" * 40,
            uv_lock_sha256="b" * 64,
            config_sha256="d" * 64,
            hardware={"cpu": "i5-9300H", "accelerator": "none"},
        ),
    )


def _protocol(**changes: object) -> ExperimentProtocol:
    values: dict[str, object] = {
        "experiment_id": "e3-ppo-cnn",
        "algorithm": "maskable-ppo",
        "architecture": {"encoder": "cnn", "channels": 32},
        "observation": {"public_board": True, "action_mask": True},
        "reward": {"name": "efficiency-v0"},
        "seeds": _inventory(),
        "checkpoints": CheckpointPlan(
            steps=(10_000, 20_000, 30_000),
            metric="valid_shots",
            direction="minimize",
        ),
        "registration": _registration(),
        "artifacts": (_artifact(),),
    }
    values.update(changes)
    return ExperimentProtocol(**values)  # type: ignore[arg-type]


def test_protocol_persists_selection_before_the_blind_confirmation() -> None:
    confirmation = TestConfirmation(
        candidate_id="ppo-cnn-v1",
        selection_record_id="selection-e3-cnn-001",
        test_seeds=(301, 302),
    )
    protocol = _protocol(
        confirmation=confirmation,
        decision=PromotionDecision(
            candidate_id="ppo-cnn-v1",
            decision="promoted",
            reason="Venceu o controle no intervalo pré-registrado.",
            confirmation=confirmation,
        ),
    )

    serialized = protocol.to_dict()

    assert serialized["protocol_version"] == PROTOCOL_VERSION
    assert serialized["registration"]["selection_split"] == "validation"
    assert serialized["confirmation"]["split"] == "test"
    assert serialized["artifacts"][0]["provenance"]["uv_lock_sha256"] == "b" * 64
    assert json.loads(json.dumps(serialized))["algorithm"] == "maskable-ppo"


def test_protocol_rejects_test_seed_during_candidate_selection() -> None:
    registration = CandidateRegistration(
        record_id="selection-e3-cnn-001",
        candidate_id="ppo-cnn-v1",
        control_id="ppo-mlp-v0.4",
        selected_checkpoint_step=20_000,
        metric="valid_shots",
        selection_seeds=(201, 301),
    )

    with pytest.raises(ValueError, match="only validation seeds"):
        _protocol(registration=registration)


@pytest.mark.parametrize(
    ("confirmation", "message"),
    [
        (
            TestConfirmation(
                candidate_id="ppo-cnn-v1",
                selection_record_id="selection-e3-cnn-001",
                test_seeds=(301,),
            ),
            "complete fixed test inventory",
        ),
        (
            TestConfirmation(
                candidate_id="ppo-cnn-v1",
                selection_record_id="another-selection",
                test_seeds=(301, 302),
            ),
            "persisted selection record",
        ),
    ],
)
def test_protocol_rejects_untraceable_blind_confirmation(
    confirmation: TestConfirmation, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _protocol(confirmation=confirmation)


def test_seed_inventory_keeps_demonstrations_outside_every_evaluation_split() -> None:
    with pytest.raises(ValueError, match="demonstration seeds overlap"):
        SeedInventory(
            train=(101,),
            validation=(201,),
            test=(301,),
            demonstration=(301,),
        )


def test_each_artifact_requires_full_provenance() -> None:
    with pytest.raises(ValueError, match="uv_lock_sha256"):
        ArtifactProvenance(
            run_id="run",
            git_commit="a" * 40,
            uv_lock_sha256="not-a-hash",
            config_sha256="d" * 64,
            hardware={"cpu": "test"},
        )
