from periodic_table_battleship_rl import __version__
from periodic_table_battleship_rl.envs import AttackEnv, PlacementEnv
from periodic_table_battleship_rl.experiments import run_ppo_attack_evaluation
from periodic_table_battleship_rl.experiments import run_placement_evaluation
from periodic_table_battleship_rl.placement import FrozenPPOEvaluator
from periodic_table_battleship_rl.training import train_placement_policy
from periodic_table_battleship_rl.visualization import plot_attack_comparison


def test_version_is_defined() -> None:
    assert __version__ == "0.9.0"


def test_environment_public_exports_are_available() -> None:
    assert AttackEnv.__name__ == "AttackEnv"
    assert PlacementEnv.__name__ == "PlacementEnv"
    assert FrozenPPOEvaluator.__name__ == "FrozenPPOEvaluator"
    assert run_ppo_attack_evaluation.__name__ == "run_ppo_attack_evaluation"
    assert run_placement_evaluation.__name__ == "run_placement_evaluation"
    assert train_placement_policy.__name__ == "train_placement_policy"
    assert plot_attack_comparison.__name__ == "plot_attack_comparison"
