from __future__ import annotations

import torch
import torch.nn.functional as F

from .schema import paper_state_layout


def yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def yaw_rotation_matrix(yaw: torch.Tensor) -> torch.Tensor:
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)
    row0 = torch.stack((cos_yaw, -sin_yaw, zeros), dim=-1)
    row1 = torch.stack((sin_yaw, cos_yaw, zeros), dim=-1)
    row2 = torch.stack((zeros, zeros, ones), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def matrix_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    quat = F.normalize(quat, dim=-1)
    qw, qx, qy, qz = quat.unbind(dim=-1)
    two_s = 2.0
    return torch.stack(
        (
            1.0 - two_s * (qy * qy + qz * qz),
            two_s * (qx * qy - qz * qw),
            two_s * (qx * qz + qy * qw),
            two_s * (qx * qy + qz * qw),
            1.0 - two_s * (qx * qx + qz * qz),
            two_s * (qy * qz - qx * qw),
            two_s * (qx * qz - qy * qw),
            two_s * (qy * qz + qx * qw),
            1.0 - two_s * (qx * qx + qy * qy),
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def rotation_6d_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2].reshape(matrix.shape[:-2] + (6,))


def build_character_yaw_state(
    *,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    joint_pos_rel: torch.Tensor,
    joint_vel: torch.Tensor,
    last_action: torch.Tensor,
    anchor_root_pos_w: torch.Tensor | None = None,
    anchor_root_quat_w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the paper-style state in a character-yaw frame.

    The root pose is expressed relative to an anchor yaw frame. Per-body
    positions and velocities are expressed in the current root yaw frame.
    """

    if body_pos_w.dim() != 3 or body_lin_vel_w.dim() != 3:
        raise ValueError("body tensors must have shape [B, num_bodies, 3]")

    anchor_root_pos_w = root_pos_w if anchor_root_pos_w is None else anchor_root_pos_w
    anchor_root_quat_w = root_quat_w if anchor_root_quat_w is None else anchor_root_quat_w
    anchor_yaw = yaw_from_quat_wxyz(anchor_root_quat_w)
    anchor_rot_w = yaw_rotation_matrix(anchor_yaw)
    current_yaw = yaw_from_quat_wxyz(root_quat_w)
    current_rot_w = yaw_rotation_matrix(current_yaw)

    root_pos_rel = torch.matmul(
        (root_pos_w - anchor_root_pos_w).unsqueeze(-2),
        anchor_rot_w,
    ).squeeze(-2)

    root_rot_w = matrix_from_quat_wxyz(root_quat_w)
    root_rot_rel = torch.matmul(anchor_rot_w.transpose(-1, -2), root_rot_w)
    root_rot_rel_6d = rotation_6d_from_matrix(root_rot_rel)

    root_lin_vel_rel = torch.matmul(root_lin_vel_w.unsqueeze(-2), current_rot_w).squeeze(-2)
    root_ang_vel_rel = torch.matmul(root_ang_vel_w.unsqueeze(-2), current_rot_w).squeeze(-2)
    body_pos_local = torch.matmul((body_pos_w - root_pos_w.unsqueeze(-2)), current_rot_w).reshape(root_pos_w.shape[0], -1)
    body_vel_local = torch.matmul(body_lin_vel_w, current_rot_w).reshape(root_pos_w.shape[0], -1)
    gravity_w = torch.zeros_like(root_pos_w)
    gravity_w[..., 2] = -1.0
    projected_gravity = torch.matmul(gravity_w.unsqueeze(-2), current_rot_w).squeeze(-2)

    state = torch.cat(
        (
            root_pos_rel,
            root_rot_rel_6d,
            root_lin_vel_rel,
            root_ang_vel_rel,
            body_pos_local,
            body_vel_local,
            projected_gravity,
            joint_pos_rel,
            joint_vel,
            last_action,
        ),
        dim=-1,
    )
    expected_dim = paper_state_layout(body_pos_w.shape[1], last_action.shape[-1])["last_action"][1]
    if state.shape[-1] != expected_dim:
        raise RuntimeError(f"character state dim mismatch: built {state.shape[-1]}, expected {expected_dim}")
    return state
