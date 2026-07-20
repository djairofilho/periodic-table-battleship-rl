"""Tests for the dependency-light periodic topology GNN prototype."""

from __future__ import annotations

import numpy as np
import pytest

from periodic_table_battleship_rl.topology import PERIODIC_TABLE_BATTLESHIP
from periodic_table_battleship_rl.training.gnn import (
    GnnMaskedPolicy,
    TopologyGraph,
    TopologyGraphQNetwork,
)


def test_graph_has_only_valid_nodes_and_valid_neighbour_edges() -> None:
    graph = TopologyGraph.from_topology(PERIODIC_TABLE_BATTLESHIP)
    index = {action: node for node, action in enumerate(graph.actions)}

    assert len(graph.actions) == 118
    assert graph.adjacency.shape == (118, 118)
    for action in graph.actions:
        connected = set(np.flatnonzero(graph.adjacency[index[action]]))
        expected = {index[action], *(index[item] for item in PERIODIC_TABLE_BATTLESHIP.neighbors(action))}
        assert connected == expected


def test_gnn_output_and_policy_preserve_canvas_mask() -> None:
    torch = pytest.importorskip("torch")
    network = TopologyGraphQNetwork.create(
        PERIODIC_TABLE_BATTLESHIP, observation_channels=4, hidden_dim=8
    )
    values = network(torch.zeros((2, 4, 10, 18)))
    mask = np.zeros(180, dtype=np.bool_)
    legal_action = min(PERIODIC_TABLE_BATTLESHIP.valid_actions)
    mask[legal_action] = True

    assert values.shape == (2, 180)
    assert GnnMaskedPolicy(network=network).select_action(
        np.zeros((4, 10, 18), dtype=np.uint8), mask
    ) == legal_action
