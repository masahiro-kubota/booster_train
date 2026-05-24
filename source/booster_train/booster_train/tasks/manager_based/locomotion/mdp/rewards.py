# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


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
