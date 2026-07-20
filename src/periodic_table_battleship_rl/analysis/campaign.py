"""Reproducible, seed-level analysis for fixed-opponent RL campaigns.

The campaign CSV files contain several observations for a held-out seed: one
per trained PPO policy and, for placement, one per attacker component.  This
module averages those observations *within* each seed before estimating an
uncertainty interval.  Consequently, the bootstrap unit remains the blind
seed rather than an artificially inflated count of correlated episodes.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import fmean, stdev
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np

from .statistics import bootstrap_mean_interval


MetricDirection = Literal["higher", "lower"]


@dataclass(frozen=True, slots=True)
class CampaignObservation:
    """One public held-out episode reduced to the metric under analysis."""

    episode_id: str
    policy_id: str
    seed: int
    scenario: str
    metric: str
    value: float
    attacker_id: str | None = None

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must not be empty")
        if not self.policy_id:
            raise ValueError("policy_id must not be empty")
        if not self.scenario:
            raise ValueError("scenario must not be empty")
        if not self.metric:
            raise ValueError("metric must not be empty")
        if not isfinite(self.value):
            raise ValueError("value must be finite")


@dataclass(frozen=True, slots=True)
class PolicySeedSummary:
    """A policy's performance after reducing repeated data within each seed."""

    experiment: str
    scenario: str
    attacker_id: str | None
    policy_id: str
    metric: str
    direction: MetricDirection
    seed_count: int
    episode_count: int
    mean: float
    seed_standard_deviation: float
    lower_95: float
    upper_95: float
    bootstrap_resamples: int


@dataclass(frozen=True, slots=True)
class PolicyComparison:
    """Paired candidate-minus-reference comparison at the blind-seed level."""

    experiment: str
    scenario: str
    attacker_id: str | None
    candidate_policy: str
    reference_policy: str
    metric: str
    direction: MetricDirection
    seed_count: int
    candidate_minus_reference_mean: float
    lower_95: float
    upper_95: float
    bootstrap_resamples: int
    conclusion: str


def load_campaign_csv(
    path: str | Path,
    *,
    metric: str,
    attacker_column: str | None = None,
) -> tuple[CampaignObservation, ...]:
    """Load and validate an exported campaign CSV without hidden game state."""

    source = Path(path)
    required = {"episode_id", "policy_id", "seed", "scenario", metric}
    if attacker_column is not None:
        required.add(attacker_column)

    with source.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(f"{source} is missing required columns: {missing}")
        rows = tuple(reader)

    if not rows:
        raise ValueError(f"{source} must contain at least one observation")
    observed_ids: set[str] = set()
    observations: list[CampaignObservation] = []
    for row in rows:
        episode_id = row["episode_id"]
        if episode_id in observed_ids:
            raise ValueError(f"{source} contains duplicate episode_id: {episode_id}")
        observed_ids.add(episode_id)
        try:
            seed = int(row["seed"])
            value = float(row[metric])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source} has an invalid seed or {metric!r} value") from error
        observations.append(
            CampaignObservation(
                episode_id=episode_id,
                policy_id=row["policy_id"],
                seed=seed,
                scenario=row["scenario"],
                metric=metric,
                value=value,
                attacker_id=None if attacker_column is None else row[attacker_column],
            )
        )
    return tuple(observations)


def summarize_policies(
    observations: Sequence[CampaignObservation],
    *,
    experiment: str,
    direction: MetricDirection,
    resamples: int = 10_000,
    bootstrap_seed: int = 20_260_720,
) -> tuple[PolicySeedSummary, ...]:
    """Summarize every scenario/policy/attacker stratum at the seed level."""

    _validate_direction(direction)
    grouped: dict[tuple[str, str, str | None, str, str], list[CampaignObservation]] = {}
    for observation in observations:
        key = (
            experiment,
            observation.scenario,
            observation.attacker_id,
            observation.policy_id,
            observation.metric,
        )
        grouped.setdefault(key, []).append(observation)

    summaries: list[PolicySeedSummary] = []
    for key in sorted(grouped, key=_summary_sort_key):
        records = grouped[key]
        seed_means = _seed_means(records)
        interval = bootstrap_mean_interval(
            tuple(seed_means.values()),
            rng=np.random.default_rng(_derived_seed(bootstrap_seed, *key)),
            resamples=resamples,
        )
        values = tuple(seed_means.values())
        summaries.append(
            PolicySeedSummary(
                experiment=key[0],
                scenario=key[1],
                attacker_id=key[2],
                policy_id=key[3],
                metric=key[4],
                direction=direction,
                seed_count=len(seed_means),
                episode_count=len(records),
                mean=interval.mean,
                seed_standard_deviation=0.0 if len(values) == 1 else stdev(values),
                lower_95=interval.lower,
                upper_95=interval.upper,
                bootstrap_resamples=resamples,
            )
        )
    return tuple(summaries)


def compare_policies(
    observations: Sequence[CampaignObservation],
    *,
    experiment: str,
    candidate_policy: str,
    reference_policies: Iterable[str],
    direction: MetricDirection,
    resamples: int = 10_000,
    bootstrap_seed: int = 20_260_720,
) -> tuple[PolicyComparison, ...]:
    """Compare a policy with references, paired by scenario, attacker, and seed.

    A policy may emit repeated observations for a seed, such as five separately
    trained PPO policies.  Those are averaged before pairing with the reference
    policy.  Different seed inventories are rejected instead of silently
    treating an unpaired difference as a controlled comparison.
    """

    _validate_direction(direction)
    references = tuple(reference_policies)
    if not references:
        raise ValueError("reference_policies must not be empty")
    candidate_strata = _strata_for_policy(observations, candidate_policy)
    comparisons: list[PolicyComparison] = []
    for scenario, attacker_id, metric in candidate_strata:
        candidate = _filter_stratum(
            observations, candidate_policy, scenario, attacker_id, metric
        )
        candidate_by_seed = _seed_means(candidate)
        for reference_policy in references:
            reference = _filter_stratum(
                observations, reference_policy, scenario, attacker_id, metric
            )
            if not reference:
                continue
            reference_by_seed = _seed_means(reference)
            if candidate_by_seed.keys() != reference_by_seed.keys():
                raise ValueError(
                    "candidate and reference must cover identical seeds for "
                    f"scenario={scenario!r}, attacker_id={attacker_id!r}"
                )
            differences = tuple(
                candidate_by_seed[seed] - reference_by_seed[seed]
                for seed in sorted(candidate_by_seed)
            )
            interval = bootstrap_mean_interval(
                differences,
                rng=np.random.default_rng(
                    _derived_seed(
                        bootstrap_seed,
                        experiment,
                        scenario,
                        attacker_id or "",
                        candidate_policy,
                        reference_policy,
                        metric,
                    )
                ),
                resamples=resamples,
            )
            comparisons.append(
                PolicyComparison(
                    experiment=experiment,
                    scenario=scenario,
                    attacker_id=attacker_id,
                    candidate_policy=candidate_policy,
                    reference_policy=reference_policy,
                    metric=metric,
                    direction=direction,
                    seed_count=len(differences),
                    candidate_minus_reference_mean=interval.mean,
                    lower_95=interval.lower,
                    upper_95=interval.upper,
                    bootstrap_resamples=resamples,
                    conclusion=_comparison_conclusion(interval.lower, interval.upper, direction),
                )
            )
    return tuple(sorted(comparisons, key=_comparison_sort_key))


def write_campaign_analysis(
    *,
    attack_csv: str | Path,
    placement_csv: str | Path,
    destination: str | Path,
    resamples: int = 10_000,
    bootstrap_seed: int = 20_260_720,
) -> tuple[tuple[PolicySeedSummary, ...], tuple[PolicyComparison, ...]]:
    """Write portable CSV, JSON, and Markdown artifacts for a v0.3-style run."""

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    attack = load_campaign_csv(attack_csv, metric="valid_shots")
    placement = load_campaign_csv(
        placement_csv,
        metric="valid_shots_to_sink",
        attacker_column="attacker_id",
    )
    attack_summaries = summarize_policies(
        attack,
        experiment="attack",
        direction="lower",
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
    )
    placement_summaries = summarize_policies(
        placement,
        experiment="placement",
        direction="higher",
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
    )
    attack_comparisons = compare_policies(
        attack,
        experiment="attack",
        candidate_policy="MaskablePPO (multi-seed)",
        reference_policies=("Hunt-target", "Random masked"),
        direction="lower",
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
    )
    placement_comparisons = compare_policies(
        placement,
        experiment="placement",
        candidate_policy="MaskablePPO placement (multi-seed)",
        reference_policies=(
            "dispersion-placement-v1",
            "hunt-target-resistant-placement-v1",
            "random-legal-placement-v1",
        ),
        direction="higher",
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
    )
    summaries = attack_summaries + placement_summaries
    comparisons = attack_comparisons + placement_comparisons
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "seed-summary.csv", (asdict(summary) for summary in summaries))
    _write_csv(
        output / "policy-comparisons.csv", (asdict(comparison) for comparison in comparisons)
    )
    report = {
        "analysis": "v0.3 seed-level bootstrap",
        "method": {
            "unit": "held-out seed after within-seed averaging",
            "interval": "two-sided 95% percentile bootstrap interval",
            "bootstrap_resamples": resamples,
            "bootstrap_seed": bootstrap_seed,
            "directions": {
                "attack": "lower valid_shots is better",
                "placement": "higher valid_shots_to_sink is better",
            },
        },
        "summaries": [asdict(summary) for summary in summaries],
        "comparisons": [asdict(comparison) for comparison in comparisons],
    }
    (output / "analysis-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "analysis-summary.md").write_text(
        analysis_markdown(summaries, comparisons), encoding="utf-8"
    )
    return summaries, comparisons


def analysis_markdown(
    summaries: Sequence[PolicySeedSummary], comparisons: Sequence[PolicyComparison]
) -> str:
    """Return a concise Markdown rendering, keeping interpretation auditable."""

    lines = [
        "# Análise v0.3 por seed",
        "",
        "A unidade de reamostragem é o seed cego. Observações repetidas de uma",
        "mesma política ou dos cinco PPOs são reduzidas à média dentro do seed",
        "antes do bootstrap. Os intervalos são percentis bootstrap bilaterais de",
        "95%; eles descrevem estes seeds avaliados e não provam generalização além",
        "do protocolo.",
        "",
        "## Comparações pareadas",
        "",
        "| Experimento | Cenário | Atacante/escopo | Comparação | Diferença | IC 95% | Leitura |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for comparison in comparisons:
        scope = comparison.attacker_id or "—"
        lines.append(
            "| {experiment} | {scenario} | {scope} | {candidate} − {reference} | "
            "{mean:+.2f} | [{lower:+.2f}; {upper:+.2f}] | {conclusion} |".format(
                experiment=comparison.experiment,
                scenario=comparison.scenario,
                scope=scope,
                candidate=comparison.candidate_policy,
                reference=comparison.reference_policy,
                mean=comparison.candidate_minus_reference_mean,
                lower=comparison.lower_95,
                upper=comparison.upper_95,
                conclusion=_conclusion_label(comparison.conclusion),
            )
        )
    lines.extend(
        [
            "",
            "## Resumos por política",
            "",
            "| Experimento | Cenário | Atacante/escopo | Política | Seeds | Episódios | Média | IC 95% |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in summaries:
        scope = summary.attacker_id or "—"
        lines.append(
            "| {experiment} | {scenario} | {scope} | {policy} | {seeds} | {episodes} | "
            "{mean:.2f} | [{lower:.2f}; {upper:.2f}] |".format(
                experiment=summary.experiment,
                scenario=summary.scenario,
                scope=scope,
                policy=summary.policy_id,
                seeds=summary.seed_count,
                episodes=summary.episode_count,
                mean=summary.mean,
                lower=summary.lower_95,
                upper=summary.upper_95,
            )
        )
    return "\n".join(lines) + "\n"


def plot_primary_comparisons(
    comparisons: Sequence[PolicyComparison], path: str | Path
) -> Path:
    """Render the attack and frozen-mixture comparisons as a readable forest plot.

    Component-specific placement results remain in the CSV and Markdown table.
    The figure deliberately limits itself to the six attack comparisons and the
    six comparisons against the pre-declared frozen mixture, avoiding a chart
    whose labels would obscure the evidence.
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    selected = [
        comparison
        for comparison in comparisons
        if comparison.experiment == "attack"
        or (
            comparison.experiment == "placement"
            and comparison.attacker_id is not None
            and comparison.attacker_id.endswith("-random-hunt-frozen-ppo")
        )
    ]
    grouped = {
        experiment: sorted(
            (item for item in selected if item.experiment == experiment),
            key=_comparison_sort_key,
        )
        for experiment in ("attack", "placement")
    }
    if not all(grouped.values()):
        raise ValueError("comparisons must include attack and frozen-mixture placement")
    figure, axes = plt.subplots(nrows=2, figsize=(10.8, 7.8), squeeze=False)
    try:
        for axis, experiment in zip(axes.flat, ("attack", "placement"), strict=True):
            entries = grouped[experiment]
            labels = [
                f"{_short_scenario(item.scenario)} · {_short_policy(item.reference_policy)}"
                for item in entries
            ]
            palette = sns.color_palette("colorblind", n_colors=len(entries))
            for index, (item, color) in enumerate(zip(entries, palette, strict=True)):
                axis.errorbar(
                    item.candidate_minus_reference_mean,
                    index,
                    xerr=[
                        [item.candidate_minus_reference_mean - item.lower_95],
                        [item.upper_95 - item.candidate_minus_reference_mean],
                    ],
                    fmt="o",
                    color=color,
                    capsize=3,
                )
            axis.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
            axis.set_yticks(range(len(entries)), labels=labels, fontsize=9)
            axis.set_xlabel(
                "PPO menos baseline em tiros para afundar\n(← baseline favorecida | PPO favorecido →)"
                if experiment == "placement"
                else "PPO menos baseline em tiros válidos\n(← PPO favorecido | baseline favorecida →)"
            )
            axis.set_title(
                "Posicionamento: mistura congelada" if experiment == "placement" else "Ataque"
            )
            axis.grid(axis="x", alpha=0.3)
        figure.tight_layout()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160, metadata={"Date": None})
        return destination
    finally:
        plt.close(figure)


def _seed_means(
    observations: Sequence[CampaignObservation],
) -> dict[int, float]:
    by_seed: dict[int, list[float]] = {}
    for observation in observations:
        by_seed.setdefault(observation.seed, []).append(observation.value)
    if not by_seed:
        raise ValueError("observations must not be empty")
    return {seed: fmean(values) for seed, values in sorted(by_seed.items())}


def _strata_for_policy(
    observations: Sequence[CampaignObservation], policy_id: str
) -> tuple[tuple[str, str | None, str], ...]:
    return tuple(
        sorted(
            {
                (observation.scenario, observation.attacker_id, observation.metric)
                for observation in observations
                if observation.policy_id == policy_id
            },
            key=lambda item: (item[0], item[1] or "", item[2]),
        )
    )


def _filter_stratum(
    observations: Sequence[CampaignObservation],
    policy_id: str,
    scenario: str,
    attacker_id: str | None,
    metric: str,
) -> tuple[CampaignObservation, ...]:
    return tuple(
        observation
        for observation in observations
        if (
            observation.policy_id == policy_id
            and observation.scenario == scenario
            and observation.attacker_id == attacker_id
            and observation.metric == metric
        )
    )


def _write_csv(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    rows = tuple(records)
    if not rows:
        raise ValueError("records must not be empty")
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _comparison_conclusion(lower: float, upper: float, direction: MetricDirection) -> str:
    if direction == "higher":
        if lower > 0.0:
            return "candidate_favored"
        if upper < 0.0:
            return "reference_favored"
    else:
        if upper < 0.0:
            return "candidate_favored"
        if lower > 0.0:
            return "reference_favored"
    return "inconclusive"


def _conclusion_label(conclusion: str) -> str:
    return {
        "candidate_favored": "favorece candidata",
        "reference_favored": "favorece referência",
        "inconclusive": "inconclusivo",
    }[conclusion]


def _derived_seed(base_seed: int, *parts: object) -> int:
    joined = "\x1f".join(str(part) for part in parts)
    digest = sha256(f"{base_seed}\x1e{joined}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _validate_direction(direction: MetricDirection) -> None:
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be either 'higher' or 'lower'")


def _summary_sort_key(
    key: tuple[str, str, str | None, str, str]
) -> tuple[str, str, str, str, str]:
    return key[0], key[1], key[2] or "", key[3], key[4]


def _comparison_sort_key(
    comparison: PolicyComparison,
) -> tuple[str, str, str, str, str]:
    return (
        comparison.experiment,
        comparison.scenario,
        comparison.attacker_id or "",
        comparison.candidate_policy,
        comparison.reference_policy,
    )


def _short_scenario(scenario: str) -> str:
    return {
        "battleship": "clássico",
        "dense-118": "dense-118",
        "periodic-table-battleship": "periódico",
    }.get(scenario, scenario)


def _short_policy(policy_id: str) -> str:
    return {
        "Hunt-target": "hunt-target",
        "Random masked": "aleatório",
        "dispersion-placement-v1": "dispersão",
        "hunt-target-resistant-placement-v1": "resistente",
        "random-legal-placement-v1": "aleatório legal",
    }.get(policy_id, policy_id)
