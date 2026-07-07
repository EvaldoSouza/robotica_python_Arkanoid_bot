from typing import Optional, List
from dataclasses import dataclass

@dataclass(frozen=True)
class Coordinate:
    """Sub-pixel precision coordinate for physical entities."""
    x_pos: float
    y_pos: float

@dataclass(frozen=True)
class FramePerception:
    """Standardized visual state passed from Vision to RL and Display."""
    ball: Optional[Coordinate]
    paddle: Optional[Coordinate]
    velocity: Optional[Coordinate]
    intercept_x: float
    block_count: int
    block_density: int = 1

@dataclass
class TelemetryHistory:
    """Global episodic tracking metrics."""
    episodes: List[int]
    rewards: List[float]
    survival_frames: List[int]
    blocks_destroyed: List[int]
    paddle_hits: List[int]
    epsilons: List[float]
    jitters: List[int]
    avg_max_q: List[float]
    alpha: float = 0.0
    gamma: float = 0.0