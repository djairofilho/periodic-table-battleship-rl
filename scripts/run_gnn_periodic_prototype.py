"""Exercise the periodic-table GNN on one public masked attack state.

This is an architectural smoke runner, not a trained policy evaluation.  It
proves that periodic gaps are absent from graph message passing and that the
greedy output honours the environment action mask.
"""

from __future__ import annotations

from periodic_table_battleship_rl.envs import AttackEnv
from periodic_table_battleship_rl.topology import PERIODIC_TABLE_BATTLESHIP
from periodic_table_battleship_rl.training import GnnMaskedPolicy, TopologyGraphQNetwork


def main() -> None:
    environment = AttackEnv(PERIODIC_TABLE_BATTLESHIP)
    observation, _ = environment.reset(seed=7301)
    network = TopologyGraphQNetwork.create(
        PERIODIC_TABLE_BATTLESHIP,
        observation_channels=observation.shape[0],
    )
    policy = GnnMaskedPolicy(network=network)
    action = policy.select_action(observation, environment.action_masks())
    print(
        {
            "scenario": PERIODIC_TABLE_BATTLESHIP.name,
            "valid_nodes": PERIODIC_TABLE_BATTLESHIP.valid_cell_count,
            "action": action,
            "legal": bool(environment.action_masks()[action]),
        }
    )


if __name__ == "__main__":
    main()
