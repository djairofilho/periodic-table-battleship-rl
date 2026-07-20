"""Run the v0.4 attack ablation against random legal fleets.

The protocol changes one factor at a time relative to the v0.3 control:
``exploration-reward`` reduces only the miss penalty and ``available-channel``
adds only a public availability plane.  All arms use identical PPO settings,
training seeds, validation checkpoints, and held-out test seeds.

Run the complete study with::

    uv run --extra train --extra visual python scripts/run_attack_ablation.py

Use ``--smoke`` to validate the same flow with a reduced budget.  A smoke run
is labelled a pilot and must not be used as the release conclusion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Sequence

from periodic_table_battleship_rl.analysis.campaign import CampaignObservation, summarize_policies
from periodic_table_battleship_rl.evaluation import EpisodeResult, RunConfig
from periodic_table_battleship_rl.evaluation.storage import write_json_atomic
from periodic_table_battleship_rl.experiments import (
    AttackAblationArm,
    AttackAblationSchedule,
    compare_ablation_arms,
    default_periodic_ablation_arms,
    run_ppo_attack_evaluation,
)
from periodic_table_battleship_rl.experiments.attack_baselines import ENVIRONMENT_VERSION
from periodic_table_battleship_rl.topology import PERIODIC_TABLE_BATTLESHIP
from periodic_table_battleship_rl.training import (
    ATTACK_POLICY_ID,
    AttackCheckpointArtifact,
    AttackTrainingArtifact,
    AttackTrainingConfig,
    AttackValidationConfig,
    load_attack_policy,
    train_attack_policy,
)
from periodic_table_battleship_rl.visualization import (
    write_attack_results_csv,
    write_attack_summary_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "v0.4-attack-ablation"


@dataclass(frozen=True, slots=True)
class SelectedModel:
    """A checkpoint selected only with this arm's validation episodes."""

    arm: AttackAblationArm
    artifact: AttackTrainingArtifact
    checkpoint: AttackCheckpointArtifact


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="render the comparison figure from an already completed report",
    )
    return parser.parse_args()


def _schedule(smoke: bool) -> AttackAblationSchedule:
    if smoke:
        return AttackAblationSchedule(
            training_seeds=(6201,),
            validation_seeds=(6301, 6302),
            test_seeds=(6401, 6402, 6403),
            total_timesteps=1_024,
            checkpoint_steps=(512, 1_024),
        )
    return AttackAblationSchedule(
        training_seeds=(6201, 6202, 6203),
        validation_seeds=tuple(range(6301, 6311)),
        test_seeds=tuple(range(6401, 6501)),
        total_timesteps=20_000,
        checkpoint_steps=(10_000, 20_000),
    )


def _configure_torch() -> None:
    """Avoid CPU oversubscription in this one-environment experiment."""
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _select_checkpoint(artifact: AttackTrainingArtifact) -> AttackCheckpointArtifact:
    if not artifact.checkpoints:
        raise ValueError("ablation training must retain validation checkpoints")
    return min(
        artifact.checkpoints,
        key=lambda item: (item.mean_valid_shots, item.training_step),
    )


def _train(
    arms: Sequence[AttackAblationArm],
    schedule: AttackAblationSchedule,
    *,
    local_models: Path,
) -> tuple[SelectedModel, ...]:
    validation = AttackValidationConfig(
        seeds=schedule.validation_seeds,
        checkpoint_steps=schedule.checkpoint_steps,
    )
    selected: list[SelectedModel] = []
    for arm in arms:
        for seed in schedule.training_seeds:
            run_id = f"v04-{arm.arm_id}-s{seed}"
            artifact = train_attack_policy(
                PERIODIC_TABLE_BATTLESHIP,
                AttackTrainingConfig(
                    run_id=run_id,
                    seed=seed,
                    total_timesteps=schedule.total_timesteps,
                    checkpoint_directory=local_models / arm.arm_id,
                    n_steps=256,
                    batch_size=64,
                    learning_rate=1e-4,
                    device="cpu",
                    environment_config=arm.environment_config,
                ),
                validation=validation,
            )
            selected.append(
                SelectedModel(arm=arm, artifact=artifact, checkpoint=_select_checkpoint(artifact))
            )
    return tuple(selected)


def _evaluate(
    models: Sequence[SelectedModel],
    schedule: AttackAblationSchedule,
    *,
    runs: Path,
    commit: str,
) -> dict[str, tuple[EpisodeResult, ...]]:
    results: dict[str, list[EpisodeResult]] = {}
    for model in models:
        seed = model.artifact.seed
        run_id = f"v04-{model.arm.arm_id}-s{seed}-test"
        evaluation = run_ppo_attack_evaluation(
            RunConfig(
                run_id=run_id,
                experiment="attack",
                scenario=PERIODIC_TABLE_BATTLESHIP.name,
                environment_version=ENVIRONMENT_VERSION,
                policy_id=ATTACK_POLICY_ID,
                split="test",
                seeds=schedule.test_seeds,
                episodes_per_seed=1,
                parameters={
                    "campaign": CAMPAIGN_ID,
                    "arm_id": model.arm.arm_id,
                    "training_seed": seed,
                    "selected_checkpoint_step": model.checkpoint.training_step,
                    "comparison_split": "blind-test",
                    "environment_config": model.arm.environment_config.public_dict(),
                },
            ),
            PERIODIC_TABLE_BATTLESHIP,
            load_attack_policy(model.checkpoint.checkpoint_path, device="cpu"),
            runs / model.arm.arm_id / run_id,
            checkpoint_path=model.checkpoint.checkpoint_path,
            training_metadata_path=model.artifact.metadata_path,
            git_commit=commit,
            uv_lock_path=ROOT / "uv.lock",
        )
        results.setdefault(model.arm.arm_id, []).extend(evaluation.results)
    return {arm_id: tuple(records) for arm_id, records in results.items()}


def _report(
    arms: Sequence[AttackAblationArm],
    schedule: AttackAblationSchedule,
    models: Sequence[SelectedModel],
    results: dict[str, tuple[EpisodeResult, ...]],
    *,
    output: Path,
    smoke: bool,
    commit: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    comparisons = compare_ablation_arms(
        scenario=PERIODIC_TABLE_BATTLESHIP.name,
        results_by_arm=results,
    )
    observations = tuple(
        CampaignObservation(
            episode_id=result.episode_id,
            policy_id=arm_id,
            seed=result.seed,
            scenario=result.scenario,
            metric="valid_shots",
            value=float(result.valid_shots),
        )
        for arm_id, records in sorted(results.items())
        for result in records
    )
    summaries = summarize_policies(
        observations,
        experiment="attack-ablation",
        direction="lower",
    )
    all_results = [result for records in results.values() for result in records]
    run_labels = {
        result.run_id: arm_id for arm_id, records in results.items() for result in records
    }
    write_attack_results_csv(all_results, output / "attack-test-episodes.csv", policy_by_run=run_labels)
    write_attack_summary_markdown(
        all_results, output / "attack-test-summary.md", policy_by_run=run_labels
    )
    report = {
        "campaign": CAMPAIGN_ID,
        "mode": "pilot-smoke" if smoke else "full",
        "git_commit": commit,
        "scenario": PERIODIC_TABLE_BATTLESHIP.name,
        "protocol": {
            "one_factor_at_a_time": True,
            "ppo": {"n_steps": 256, "batch_size": 64, "learning_rate": 1e-4},
            "schedule": schedule.public_dict(),
            "selection": "lowest validation mean valid_shots; earlier checkpoint breaks ties",
            "test": "blind, public observation and action mask only",
            "statistics": "seed-level paired percentile bootstrap, 10,000 resamples",
        },
        "arms": [arm.public_dict() for arm in arms],
        "selected_checkpoints": [
            {
                "arm_id": model.arm.arm_id,
                "training_seed": model.artifact.seed,
                "checkpoint_step": model.checkpoint.training_step,
                "validation_mean_valid_shots": model.checkpoint.mean_valid_shots,
            }
            for model in models
        ],
        "summaries": [asdict(item) for item in summaries],
        "comparisons": [asdict(item) for item in comparisons],
    }
    write_json_atomic(output / "ablation-report.json", report)
    _write_markdown(report, output / "ablation-summary.md")


def _write_markdown(report: dict[str, object], destination: Path) -> None:
    comparisons = report["comparisons"]
    assert isinstance(comparisons, list)
    lines = [
        "# Ablação v0.4 de ataque",
        "",
        "Menos `valid_shots` é melhor. A unidade estatística é o seed cego, após",
        "a média das políticas treinadas no mesmo seed. As intervenções não revelam",
        "a frota: usam somente observação pública e máscara de ações.",
        "",
        "| Braço | Hipótese |",
        "| --- | --- |",
    ]
    for arm in report["arms"]:
        assert isinstance(arm, dict)
        lines.append(f"| {arm['arm_id']} | {arm['hypothesis']} |")
    lines.extend(
        [
            "",
            "## Comparações cegas contra o controle",
            "",
            "| Candidata − controle | Diferença | IC 95% | Conclusão |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for item in comparisons:
        assert isinstance(item, dict)
        lines.append(
            "| {candidate_policy} − {reference_policy} | {candidate_minus_reference_mean:+.2f} | "
            "[{lower_95:+.2f}; {upper_95:+.2f}] | {conclusion} |".format(**item)
        )
    lines.extend(
        [
            "",
            "Um intervalo abaixo de zero favorece a candidata; um intervalo que cruza",
            "zero é inconclusivo. A conclusão aplica-se ao orçamento e aos seeds deste",
            "protocolo, não constitui ajuste posterior nem prova de generalização.",
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def _plot_comparisons(comparisons: Sequence[dict[str, object]], destination: Path) -> None:
    """Render the seed-level effects without retraining any policy."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = [str(item["candidate_policy"]) for item in comparisons]
    means = [float(item["candidate_minus_reference_mean"]) for item in comparisons]
    lower = [float(item["lower_95"]) for item in comparisons]
    upper = [float(item["upper_95"]) for item in comparisons]
    figure, axis = plt.subplots(figsize=(8.2, 3.4))
    try:
        for index, (mean, low, high) in enumerate(zip(means, lower, upper, strict=True)):
            axis.errorbar(
                mean,
                index,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                color="#1f77b4",
                capsize=4,
            )
        axis.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        axis.set_yticks(range(len(labels)), labels=labels)
        axis.set_xlabel("Braço − controle em tiros válidos (← candidata favorecida)")
        axis.set_title("Ablação de ataque: efeito cego por seed com IC 95%")
        axis.grid(axis="x", alpha=0.3)
        figure.tight_layout()
        figure.savefig(destination, dpi=160, metadata={"Date": None})
    finally:
        plt.close(figure)


def _render_existing_report(artifacts: Path) -> None:
    report = json.loads((artifacts / "ablation-report.json").read_text(encoding="utf-8"))
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list) or not all(
        isinstance(item, dict) for item in comparisons
    ):
        raise ValueError("ablation report must contain comparison objects")
    _plot_comparisons(comparisons, artifacts / "ablation-comparison.png")


def main() -> None:
    arguments = _arguments()
    schedule = _schedule(arguments.smoke)
    arms = default_periodic_ablation_arms()
    suffix = "-smoke" if arguments.smoke else ""
    local_models = ROOT / ".local-runs" / f"{CAMPAIGN_ID}{suffix}"
    runs = ROOT / "runs" / f"{CAMPAIGN_ID}{suffix}"
    artifacts = ROOT / "artifacts" / f"{CAMPAIGN_ID}{suffix}"
    if arguments.render_existing:
        _render_existing_report(artifacts)
        return
    _configure_torch()
    commit = _commit()
    models = _train(arms, schedule, local_models=local_models)
    results = _evaluate(models, schedule, runs=runs, commit=commit)
    _report(arms, schedule, models, results, output=artifacts, smoke=arguments.smoke, commit=commit)
    _render_existing_report(artifacts)
    print(json.dumps(json.loads((artifacts / "ablation-report.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
