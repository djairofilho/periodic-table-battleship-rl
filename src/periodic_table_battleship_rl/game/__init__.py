"""Pure Battleship domain objects and fleet sampling."""

from .fleet import (
    CANONICAL_FLEET,
    CandidatePlacement,
    Fleet,
    FleetSamplingError,
    Orientation,
    ShipPlacement,
    ShipSpec,
    TopologyProtocol,
    candidate_placements,
    is_legal_fleet,
    sample_random_legal_fleet,
)

__all__ = [
    "CANONICAL_FLEET",
    "CandidatePlacement",
    "Fleet",
    "FleetSamplingError",
    "Orientation",
    "ShipPlacement",
    "ShipSpec",
    "TopologyProtocol",
    "candidate_placements",
    "is_legal_fleet",
    "sample_random_legal_fleet",
]
