"""MDP terms specific to fixed-height Pikachu locomotion."""

import torch

TARGET_TRUNK_HEIGHT_M = 0.45
MIN_TRUNK_HEIGHT_M = 0.33


def trunk_height(env) -> torch.Tensor:
    """Return Trunk-link origin height relative to each environment origin."""
    return env.scene["robot"].data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]


def height_error(env, target_height: float = TARGET_TRUNK_HEIGHT_M) -> torch.Tensor:
    """Signed fixed-height error used by both the critic and reward."""
    return trunk_height(env) - target_height


def height_tracking(env, target_height: float = TARGET_TRUNK_HEIGHT_M, std: float = 0.025) -> torch.Tensor:
    """Reward a fixed Trunk height without prescribing joint angles or tilt."""
    return torch.exp(-height_error(env, target_height).square() / std**2)


def height_observation(env, target_height: float = TARGET_TRUNK_HEIGHT_M) -> torch.Tensor:
    """Expose the fixed-height error to the critic only."""
    return height_error(env, target_height).unsqueeze(-1)


def low_height(env, min_height: float = MIN_TRUNK_HEIGHT_M) -> torch.Tensor:
    """Terminate only after substantial height loss; do not constrain roll/pitch."""
    return trunk_height(env) < min_height
