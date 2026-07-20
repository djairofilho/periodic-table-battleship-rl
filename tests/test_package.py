from periodic_table_battleship_rl import __version__
from periodic_table_battleship_rl.envs import AttackEnv, PlacementEnv


def test_version_is_defined() -> None:
    assert __version__ == "0.1.0"


def test_environment_public_exports_are_available() -> None:
    assert AttackEnv.__name__ == "AttackEnv"
    assert PlacementEnv.__name__ == "PlacementEnv"
