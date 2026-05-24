from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

STATE_KEY = "state"
ACTION_KEY = "action"
ROOT_POSE_KEY = "root_pose"
ROOT_VELOCITY_KEY = "root_velocity"
BODY_POS_KEY = "body_pos_w"
BODY_VELOCITY_KEY = "body_velocity_w"
JOINT_STATE_KEY = "joint_state"
LAST_ACTION_KEY = "last_action"
TEACHER_ID_KEY = "teacher_id"
COMMAND_KEY = "command"


@dataclass(frozen=True)
class DatasetManifest:
    """Metadata shared by rollout shards, training, export, and deploy."""

    version: int
    state_dim: int
    action_dim: int
    action_joint_names: list[str]
    state_layout: dict[str, list[int]]
    policy_dt: float = 0.02
    source: str = "unknown"
    body_names: list[str] | None = None
    num_bodies: int | None = None
    raw_state_dim: int | None = None
    projected_state_dim: int | None = None
    latent_dim: int | None = None
    history_length: int | None = None
    current_index: int | None = None
    horizon: int | None = None
    window_length: int | None = None
    denoising_steps: int | None = None
    projection_artifact: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "DatasetManifest":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    def to_file(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
            f.write("\n")


def default_state_layout(action_dim: int) -> dict[str, list[int]]:
    """State layout used by the v1 joystick diffusion controller.

    State is:
    root_lin_vel_b(3), root_ang_vel_b(3), projected_gravity_b(3),
    joint_pos_rel(action_dim), joint_vel(action_dim), last_action(action_dim).
    """

    joint_pos_start = 9
    joint_vel_start = joint_pos_start + action_dim
    last_action_start = joint_vel_start + action_dim
    return {
        "root_lin_vel_b": [0, 3],
        "root_ang_vel_b": [3, 6],
        "projected_gravity_b": [6, 9],
        "joint_pos_rel": [joint_pos_start, joint_vel_start],
        "joint_vel": [joint_vel_start, last_action_start],
        "last_action": [last_action_start, last_action_start + action_dim],
        "planar_velocity": [0, 2],
        "yaw_rate": [5, 6],
    }


def paper_state_layout(num_bodies: int, action_dim: int) -> dict[str, list[int]]:
    """Character-yaw-frame state layout used by the paper-style diffusion model.

    State is:
    root_relative_position(3), root_relative_rotation_6d(6),
    root_linear_velocity_yaw(3), root_angular_velocity_yaw(3),
    body_local_position(num_bodies * 3), body_local_velocity(num_bodies * 3),
    projected_gravity_yaw(3), joint_pos_rel(action_dim), joint_vel(action_dim),
    last_action(action_dim).
    """

    root_pos_start = 0
    root_rot_start = root_pos_start + 3
    root_lin_vel_start = root_rot_start + 6
    root_ang_vel_start = root_lin_vel_start + 3
    body_pos_start = root_ang_vel_start + 3
    body_vel_start = body_pos_start + num_bodies * 3
    gravity_start = body_vel_start + num_bodies * 3
    joint_pos_start = gravity_start + 3
    joint_vel_start = joint_pos_start + action_dim
    last_action_start = joint_vel_start + action_dim
    return {
        "root_pos_rel": [root_pos_start, root_rot_start],
        "root_rot_rel_6d": [root_rot_start, root_lin_vel_start],
        "root_lin_vel_rel": [root_lin_vel_start, root_ang_vel_start],
        "root_ang_vel_rel": [root_ang_vel_start, body_pos_start],
        "body_pos_local": [body_pos_start, body_vel_start],
        "body_velocity_local": [body_vel_start, gravity_start],
        "projected_gravity": [gravity_start, joint_pos_start],
        "joint_pos_rel": [joint_pos_start, joint_vel_start],
        "joint_vel": [joint_vel_start, last_action_start],
        "last_action": [last_action_start, last_action_start + action_dim],
        "planar_velocity": [root_lin_vel_start, root_lin_vel_start + 2],
        "yaw_rate": [root_ang_vel_start + 2, root_ang_vel_start + 3],
    }


def validate_manifest_dict(manifest: dict[str, Any]) -> None:
    required = {"version", "state_dim", "action_dim", "action_joint_names", "state_layout"}
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"Manifest is missing required keys: {missing}")

    if int(manifest["state_dim"]) <= 0:
        raise ValueError("Manifest state_dim must be positive")
    if int(manifest["action_dim"]) <= 0:
        raise ValueError("Manifest action_dim must be positive")
    if len(manifest["action_joint_names"]) != int(manifest["action_dim"]):
        raise ValueError("Manifest action_joint_names length must match action_dim")
