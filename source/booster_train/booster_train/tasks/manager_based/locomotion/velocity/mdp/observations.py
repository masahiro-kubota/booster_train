# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.observations import base_lin_vel
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def k1_deploy_locomotion_observation(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    obs_dof_vel_scale: float,
) -> torch.Tensor:
    """Single-frame observation compatible with the K1 deploy controller."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids

    joint_pos_rel = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    joint_vel_scaled = asset.data.joint_vel[:, joint_ids] * obs_dof_vel_scale

    return torch.cat(
        (
            asset.data.root_ang_vel_b,
            asset.data.projected_gravity_b,
            env.command_manager.get_command(command_name),
            joint_pos_rel,
            joint_vel_scaled,
            env.action_manager.action,
        ),
        dim=-1,
    )


def k1_privileged_locomotion_observation(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    obs_dof_vel_scale: float,
    contact_threshold: float,
) -> torch.Tensor:
    """Single-frame privileged critic observation for K1 locomotion training."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > contact_threshold
    )

    policy_obs = k1_deploy_locomotion_observation(
        env=env,
        command_name=command_name,
        asset_cfg=asset_cfg,
        obs_dof_vel_scale=obs_dof_vel_scale,
    )
    return torch.cat((policy_obs, base_lin_vel(env, asset_cfg), contacts.float()), dim=-1)
