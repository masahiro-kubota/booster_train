from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StateProjection:
    projection: torch.Tensor
    pseudoinverse: torch.Tensor
    raw_state_dim: int
    projected_state_dim: int
    root_indices: list[int]
    random_dim: int
    root_weight: float
    seed: int


def root_emphasis_indices(state_layout: dict[str, list[int]]) -> list[int]:
    indices: list[int] = []
    for key in ("root_pos_rel", "root_rot_rel_6d", "root_lin_vel_rel", "root_ang_vel_rel"):
        start, end = state_layout[key]
        indices.extend(range(int(start), int(end)))
    return indices


def build_state_projection(
    raw_state_dim: int,
    root_indices: list[int],
    *,
    random_dim: int = 0,
    root_weight: float = 6.0,
    seed: int = 1,
) -> StateProjection:
    """Build the paper-style emphasis projection P and pseudo-inverse P^-1.

    P is stacked as [A, B, I]^T where A is a Gaussian random projection and B
    is a diagonal root-emphasis matrix. The raw state is projected by
    ``projected = raw @ P.T``.
    """

    raw_state_dim = int(raw_state_dim)
    random_dim = int(random_dim)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    chunks: list[torch.Tensor] = []
    if random_dim > 0:
        chunks.append(torch.randn(random_dim, raw_state_dim, generator=generator) / raw_state_dim**0.5)

    emphasis = torch.zeros(raw_state_dim, raw_state_dim, dtype=torch.float32)
    if root_indices:
        idx = torch.tensor(root_indices, dtype=torch.long)
        emphasis[idx, idx] = float(root_weight)
    chunks.append(emphasis)
    chunks.append(torch.eye(raw_state_dim, dtype=torch.float32))
    projection = torch.cat(chunks, dim=0).contiguous()
    pseudoinverse = torch.linalg.pinv(projection).contiguous()
    return StateProjection(
        projection=projection,
        pseudoinverse=pseudoinverse,
        raw_state_dim=raw_state_dim,
        projected_state_dim=int(projection.shape[0]),
        root_indices=list(root_indices),
        random_dim=random_dim,
        root_weight=float(root_weight),
        seed=int(seed),
    )


def save_state_projection(state_projection: StateProjection, path: str) -> None:
    torch.save(
        {
            "projection": state_projection.projection,
            "pseudoinverse": state_projection.pseudoinverse,
            "raw_state_dim": state_projection.raw_state_dim,
            "projected_state_dim": state_projection.projected_state_dim,
            "root_indices": state_projection.root_indices,
            "random_dim": state_projection.random_dim,
            "root_weight": state_projection.root_weight,
            "seed": state_projection.seed,
        },
        path,
    )
