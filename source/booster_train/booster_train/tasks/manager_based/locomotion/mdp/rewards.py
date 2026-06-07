from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

try:
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
    from isaaclab.utils.math import yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def k1_joint_torques_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(asset.data.applied_torque[:, asset_cfg.joint_ids].square(), dim=-1)


def k1_joint_energy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_power = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    )
    return torch.norm(joint_power, dim=-1)


def action_l2_subset(env, action_indices: list[int] | tuple[int, ...]) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action[:, action_indices]), dim=-1)


def action_rate_l2_subset(env, action_indices: list[int] | tuple[int, ...]) -> torch.Tensor:
    action = env.action_manager.action[:, action_indices]
    prev_action = env.action_manager.prev_action[:, action_indices]
    return torch.sum(torch.square(action - prev_action), dim=-1)


def k1_body_force(env, sensor_cfg: SceneEntityCfg, threshold: float, max_reward: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward = torch.where(reward < threshold, torch.zeros_like(reward), reward - threshold)
    return reward.clamp(min=0.0, max=max_reward)


def k1_feet_stumble(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    horizontal_force = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=-1)
    vertical_force = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    return torch.any(horizontal_force > 5.0 * vertical_force, dim=-1).float()


def k1_feet_too_near_humanoid(
    env,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0.0)


def k1_yaw_frame_lin_vel_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    return vel_yaw[:, :2]


def k1_command_direction_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    cap_to_command: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    command_norm = torch.linalg.norm(command_xy, dim=1)
    command_dir = command_xy / torch.clamp(command_norm.unsqueeze(1), min=1.0e-6)
    velocity = torch.sum(k1_yaw_frame_lin_vel_xy(env, asset_cfg) * command_dir, dim=1)
    if cap_to_command:
        velocity = torch.minimum(velocity, command_norm)
    return torch.where(command_norm > command_threshold, velocity, torch.zeros_like(velocity))


def k1_flat_forward_metrics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Log per-episode motion metrics without affecting rewards."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_x = asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    elapsed_time = torch.clamp(env.episode_length_buf[env_ids].to(torch.float32) * env.step_dt, min=env.step_dt)
    speed = root_x / elapsed_time
    command_speed = k1_command_direction_velocity(
        env,
        command_name=command_name,
        cap_to_command=False,
        asset_cfg=asset_cfg,
    )[env_ids]
    yaw_velocity = k1_yaw_frame_lin_vel_xy(env, asset_cfg)[env_ids]
    return {
        "distance_m": torch.mean(root_x),
        "speed_mps": torch.mean(speed),
        "command_direction_speed_mps": torch.mean(command_speed),
        "yaw_frame_vel_x_mps": torch.mean(yaw_velocity[:, 0]),
        "root_body_vel_x_mps": torch.mean(asset.data.root_lin_vel_b[env_ids, 0]),
    }
