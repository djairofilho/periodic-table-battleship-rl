"""Small PyTorch message-passing prototype for periodic-table attack states.

No graph framework is required: valid cells are graph nodes and only valid
orthogonal topology neighbours exchange messages.  Canvas gaps therefore do
not become artificial pixels or edges.  This module is intentionally a model
prototype, not a promoted attacker or a completed training campaign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from periodic_table_battleship_rl.topology import Topology

from .dqn import DQN_ATTACK_POLICY_ID, _require_torch, masked_argmax


@dataclass(frozen=True, slots=True)
class TopologyGraph:
    """Stable graph projection of one topology's valid canvas cells."""

    actions: tuple[int, ...]
    adjacency: np.ndarray
    coordinates: np.ndarray

    @classmethod
    def from_topology(cls, topology: Topology) -> "TopologyGraph":
        """Create a directed, self-looped mean-aggregation graph."""
        actions = tuple(sorted(topology.valid_actions))
        action_to_node = {action: index for index, action in enumerate(actions)}
        adjacency = np.eye(len(actions), dtype=np.float32)
        for action in actions:
            source = action_to_node[action]
            for neighbour in topology.neighbors(action):
                adjacency[source, action_to_node[neighbour]] = 1.0
        adjacency /= adjacency.sum(axis=1, keepdims=True)
        coordinates = np.asarray(
            [topology.coordinate_for(action) for action in actions], dtype=np.float32
        )
        coordinates[:, 0] /= max(topology.rows - 1, 1)
        coordinates[:, 1] /= max(topology.columns - 1, 1)
        return cls(actions=actions, adjacency=adjacency, coordinates=coordinates)


class TopologyGraphQNetwork:
    """Lazy PyTorch module factory for message passing on a ``TopologyGraph``."""

    @staticmethod
    def create(
        topology: Topology,
        *,
        observation_channels: int,
        hidden_dim: int = 64,
        message_passing_steps: int = 2,
    ) -> Any:
        """Build a Q-network whose output remains aligned to canvas actions."""
        if observation_channels <= 0 or hidden_dim <= 0 or message_passing_steps <= 0:
            raise ValueError("channels, hidden_dim and message_passing_steps must be positive")
        torch = _require_torch()
        graph = TopologyGraph.from_topology(topology)

        class _Network(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "node_actions", torch.as_tensor(graph.actions, dtype=torch.long)
                )
                self.register_buffer("adjacency", torch.as_tensor(graph.adjacency))
                self.register_buffer("coordinates", torch.as_tensor(graph.coordinates))
                self.input_layer = torch.nn.Linear(observation_channels + 2, hidden_dim)
                self.self_layers = torch.nn.ModuleList(
                    torch.nn.Linear(hidden_dim, hidden_dim)
                    for _ in range(message_passing_steps)
                )
                self.message_layers = torch.nn.ModuleList(
                    torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
                    for _ in range(message_passing_steps)
                )
                self.head = torch.nn.Linear(hidden_dim, 1)
                self.action_count = topology.action_count

            def forward(self, observations: Any) -> Any:
                if observations.ndim != 4:
                    raise ValueError("observations must have shape (batch, channels, rows, columns)")
                if observations.shape[1] != observation_channels:
                    raise ValueError("observation channel count does not match the network")
                flattened = observations.flatten(start_dim=2)
                public_nodes = flattened.index_select(2, self.node_actions).transpose(1, 2)
                positions = self.coordinates.unsqueeze(0).expand(observations.shape[0], -1, -1)
                features = torch.cat((public_nodes, positions), dim=-1)
                nodes = torch.relu(self.input_layer(features))
                for self_layer, message_layer in zip(
                    self.self_layers, self.message_layers, strict=True
                ):
                    neighbours = torch.matmul(self.adjacency, nodes)
                    nodes = torch.relu(self_layer(nodes) + message_layer(neighbours))
                node_values = self.head(nodes).squeeze(-1)
                values = torch.zeros(
                    (observations.shape[0], self.action_count),
                    dtype=node_values.dtype,
                    device=node_values.device,
                )
                return values.scatter(1, self.node_actions.unsqueeze(0).expand_as(node_values), node_values)

        return _Network()


@dataclass(frozen=True, slots=True)
class GnnMaskedPolicy:
    """Greedy public-state policy adapter for a graph Q-network prototype."""

    network: Any
    device: str = "cpu"
    policy_id: str = DQN_ATTACK_POLICY_ID

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> int:
        """Choose a legal action without exposing fleet placement to the model."""
        del deterministic
        torch = _require_torch()
        if action_mask.dtype != np.bool_:
            raise TypeError("action_mask must have dtype bool")
        if not action_mask.any():
            raise ValueError("cannot select action from an empty mask")
        self.network.eval()
        with torch.no_grad():
            values = self.network(
                torch.as_tensor(observation, device=self.device).unsqueeze(0).float()
            )
            mask = torch.as_tensor(action_mask, device=self.device).unsqueeze(0)
            return int(masked_argmax(values, mask).item())
