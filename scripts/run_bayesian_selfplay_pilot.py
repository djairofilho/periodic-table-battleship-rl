"""Run one validation-only placement update against a frozen Bayesian attacker.

This is an integration pilot, not a performance campaign.  It produces a
portable league ledger and frozen-suite evidence while deliberately refusing
to open the blind attack-test inventory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt

from periodic_table_battleship_rl.evaluation import sha256_file
from periodic_table_battleship_rl.placement.baselines import RandomLegalPlacementPolicy
from periodic_table_battleship_rl.placement.defensive import (
    FrozenDefensiveMixture,
    HuntTargetEvaluator,
)
from periodic_table_battleship_rl.selfplay import (
    BayesianAttackEvaluator,
    CoupledSelfPlayRunner,
    CoupledTrainingOutput,
    FrozenEvaluationSuite,
    PlacementPolicyFleetSampler,
    SelfPlayCampaignConfig,
    SelfPlayCampaignRecord,
    SnapshotLeague,
    SnapshotProvenance,
    ValidationFrozenSuiteEvaluator,
)
from periodic_table_battleship_rl.topology import BATTLESHIP
from periodic_table_battleship_rl.training.placement import (
    PLACEMENT_POLICY_ID,
    PlacementTrainingConfig,
    load_placement_policy,
    train_placement_policy,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = ROOT / "runs" / "v0.6-bayesian-selfplay-validation"
ARTIFACT_DIRECTORY = ROOT / "artifacts" / "v0.6-bayesian-selfplay-validation"
LOCAL_CHECKPOINT_DIRECTORY = ROOT / ".local-runs" / "v0.6-bayesian-selfplay"
CAMPAIGN_ID = "v06-bayesian-selfplay-placement-pilot"
VALIDATION_SEEDS = (8611, 8612, 8613)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=64)
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


class _BayesianPlacementPilotTrainer:
    """Train the one registered placer update against the frozen Bayes policy."""

    def __init__(self, checkpoint_directory: Path) -> None:
        self.checkpoint_directory = checkpoint_directory

    def train_attacker(self, plan, environment):
        del plan, environment
        raise RuntimeError("the one-round pilot only trains a placement policy")

    def train_placer(self, plan, evaluator) -> CoupledTrainingOutput:
        expected = "belief_probability_mc-v1"
        if evaluator.evaluator_id != expected:
            raise ValueError(f"pilot requires the frozen {expected} evaluator")
        mixture = FrozenDefensiveMixture(
            evaluators=(evaluator,),
            weights=(1.0,),
            evaluator_id="frozen-bayesian-probability-v1",
        )
        artifact = train_placement_policy(
            BATTLESHIP,
            PlacementTrainingConfig(
                run_id=f"{CAMPAIGN_ID}-round-{plan.round_index:03d}",
                seed=plan.training_seed,
                total_timesteps=plan.timesteps,
                checkpoint_directory=self.checkpoint_directory,
                n_steps=min(32, plan.timesteps),
                batch_size=min(32, plan.timesteps),
            ),
            defensive_mixture=mixture,
        )
        policy = load_placement_policy(artifact.checkpoint_path)
        return CoupledTrainingOutput(
            checkpoint_path=artifact.checkpoint_path,
            source_run_id=artifact.metadata_path.parent.name,
            runtime_opponent=PlacementPolicyFleetSampler(
                policy=policy,
                sampler_id=f"{CAMPAIGN_ID}-placer-round-{plan.round_index:03d}",
            ),
            policy_id=PLACEMENT_POLICY_ID,
        )


def main() -> None:
    args = _arguments()
    if args.timesteps < 32:
        raise ValueError("timesteps must be at least 32 for the registered PPO pilot")
    if args.sample_count <= 0:
        raise ValueError("sample-count must be positive")
    base_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    working_tree_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()
    )
    git_commit = f"{base_commit}-dirty" if working_tree_dirty else base_commit
    bayes = BayesianAttackEvaluator(BATTLESHIP, sample_count=args.sample_count)
    random_placer = PlacementPolicyFleetSampler(
        policy=RandomLegalPlacementPolicy(BATTLESHIP, seed=202606),
        sampler_id="random-legal-placement-v1",
    )
    suite = FrozenEvaluationSuite(
        attacker_evaluator_ids=(bayes.evaluator_id, "hunt-target-v1"),
        placement_policy_ids=(random_placer.sampler_id,),
    )
    config = SelfPlayCampaignConfig(
        campaign_id=CAMPAIGN_ID,
        scenario=BATTLESHIP.name,
        seed=202606,
        round_count=1,
        attacker_timesteps=args.timesteps,
        placer_timesteps=args.timesteps,
        first_learner="placer",
        frozen_evaluation=suite,
    )
    bayes_contract_path, placer_contract_path = _write_bootstrap_contracts(
        args.sample_count, git_commit
    )
    record = SelfPlayCampaignRecord(
        config=config,
        initial_league=SnapshotLeague(
            scenario=BATTLESHIP.name,
            snapshots=(
                SnapshotProvenance(
                    snapshot_id="bayes-probability-bootstrap",
                    role="attacker",
                    policy_id=bayes.evaluator_id,
                    scenario=BATTLESHIP.name,
                    source_run_id="v06-bayes-planner-validation",
                    checkpoint_sha256=sha256_file(bayes_contract_path),
                    training_round=0,
                ),
                SnapshotProvenance(
                    snapshot_id="random-placement-bootstrap",
                    role="placer",
                    policy_id=PLACEMENT_POLICY_ID,
                    scenario=BATTLESHIP.name,
                    source_run_id="v06-random-placement-bootstrap",
                    checkpoint_sha256=sha256_file(placer_contract_path),
                    training_round=0,
                ),
            ),
        ),
    )
    frozen_suite = ValidationFrozenSuiteEvaluator(
        topology=BATTLESHIP,
        validation_seeds=VALIDATION_SEEDS,
        attacker_evaluators={
            bayes.evaluator_id: bayes,
            "hunt-target-v1": HuntTargetEvaluator(BATTLESHIP),
        },
        placement_samplers={random_placer.sampler_id: random_placer},
    )
    bootstrap_scores = dict(
        frozen_suite.evaluate(
            role="placer",
            runtime_opponent=random_placer,
            target_ids=suite.attacker_evaluator_ids,
        )
    )
    runner = CoupledSelfPlayRunner(
        record=record,
        topology=BATTLESHIP,
        trainer=_BayesianPlacementPilotTrainer(LOCAL_CHECKPOINT_DIRECTORY),
        frozen_suite=frozen_suite,
        runtime_opponents={
            "bayes-probability-bootstrap": bayes,
            "random-placement-bootstrap": random_placer,
        },
        output_directory=RUN_DIRECTORY,
    )
    snapshot = runner.run_next_round()
    if snapshot is None or runner.run_next_round() is not None:
        raise RuntimeError("one-round Bayesian self-play pilot did not complete exactly once")
    _write_report(
        runner,
        frozen_suite,
        snapshot,
        bootstrap_scores,
        args,
        git_commit,
        working_tree_dirty,
    )


def _write_bootstrap_contracts(sample_count: int, git_commit: str) -> tuple[Path, Path]:
    """Persist independent hashable contracts for both bootstrap snapshots."""

    LOCAL_CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    bayes_path = LOCAL_CHECKPOINT_DIRECTORY / "bayes-probability-bootstrap.json"
    bayes_path.write_text(
        json.dumps(
            {
                "policy_id": "belief_probability_mc-v1",
                "strategy": "probability",
                "sample_count": sample_count,
                "source_commit": git_commit,
                "private_fleet_access": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    placer_path = LOCAL_CHECKPOINT_DIRECTORY / "random-placement-bootstrap.json"
    placer_path.write_text(
        json.dumps(
            {
                "policy_id": PLACEMENT_POLICY_ID,
                "strategy": "random-legal-placement-v1",
                "seed": 202606,
                "source_commit": git_commit,
                "private_attack_access": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return bayes_path, placer_path


def _write_report(
    runner: CoupledSelfPlayRunner,
    frozen_suite: ValidationFrozenSuiteEvaluator,
    snapshot: SnapshotProvenance,
    bootstrap_scores: dict[str, float],
    args: argparse.Namespace,
    git_commit: str,
    working_tree_dirty: bool,
) -> None:
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    round_audit = json.loads((RUN_DIRECTORY / "round-000.json").read_text(encoding="utf-8"))
    report = {
        "campaign": CAMPAIGN_ID,
        "split": "validation",
        "blind_test_used": False,
        "git_commit": git_commit,
        "working_tree_dirty": working_tree_dirty,
        "planner": {
            "policy_id": "belief_probability_mc-v1",
            "sample_count": args.sample_count,
            "posterior_exact": False,
        },
        "training": {"timesteps": args.timesteps, "role": "placer"},
        "frozen_suite": frozen_suite.public_dict(),
        "snapshot": snapshot.to_dict(),
        "bootstrap_frozen_evaluation": bootstrap_scores,
        "round_audit": round_audit,
        "ledger": runner.record.to_dict(),
        "promotion": {"status": "not-decided", "blind_test_used": False},
    }
    (ARTIFACT_DIRECTORY / "bayesian-selfplay-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    scores = round_audit["frozen_evaluation"]
    _plot_scores(bootstrap_scores, scores)
    lines = [
        "# Piloto de self-play Bayesiano v0.6",
        "",
        "Este piloto executa uma atualização de posicionamento contra o planejador",
        "Bayesiano de maior probabilidade congelado. Ele usa somente validação e",
        "não abre o teste cego.",
        "",
        f"- Seeds de validação: `{list(VALIDATION_SEEDS)}`",
        f"- Amostras Monte Carlo por decisão: `{args.sample_count}`",
        f"- Budget de treino do posicionador: `{args.timesteps}` passos",
        f"- Snapshot produzido: `{snapshot.snapshot_id}`",
        "",
        "| Atacante congelado | Bootstrap aleatório | Posicionador treinado | Diferença |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| `{target}` | {bootstrap_scores[target]:.2f} | {score:.2f} | "
        f"{score - bootstrap_scores[target]:+.2f} |"
        for target, score in scores.items()
    )
    lines.extend(
        [
            "",
            "![Avaliação congelada do piloto](bayesian-selfplay-frozen-scores.png)",
            "",
            "Os valores são evidência de validação de uma única atualização, não",
            "uma decisão de promoção. A proveniência da liga e os hashes estão no",
            "arquivo JSON ao lado deste relatório.",
        ]
    )
    (ARTIFACT_DIRECTORY / "bayesian-selfplay-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _plot_scores(bootstrap_scores: dict[str, float], scores: dict[str, float]) -> None:
    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    labels = list(scores)
    positions = list(range(len(labels)))
    width = 0.36
    axis.bar(
        [position - width / 2 for position in positions],
        [bootstrap_scores[label] for label in labels],
        width=width,
        label="bootstrap aleatório",
        color="#94a3b8",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        [scores[label] for label in labels],
        width=width,
        label="posicionador treinado",
        color="#2563eb",
    )
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylabel("Tiros válidos médios (maior é melhor para o posicionador)")
    axis.set_title("Piloto v0.6: posicionador contra atacantes congelados")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(ARTIFACT_DIRECTORY / "bayesian-selfplay-frozen-scores.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
