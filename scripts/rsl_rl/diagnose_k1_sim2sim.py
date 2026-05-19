# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a TorchScript K1 policy under current or legacy Isaac Lab K1 settings.

This is a small diagnostic utility for checking whether a policy was trained
against the current BoosterDelayedPDActuator K1 setup or the older
DelayedImplicitActuator setup used by the deploy examples.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Diagnose exported K1 TorchScript policy compatibility.")
parser.add_argument("--task", type=str, default="Booster-K1-Fight_001-v0-Play", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--max_steps", type=int, default=500, help="Number of simulation steps to run.")
parser.add_argument(
    "--k1_profile",
    choices=("current", "legacy"),
    default="current",
    help="Use the current K1 actuator/action scale or the legacy deploy-compatible profile.",
)
parser.add_argument(
    "--disable_randomization",
    action="store_true",
    default=False,
    help="Disable startup and reset randomization for a cleaner deploy-style comparison.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.checkpoint is None:
    parser.error("--checkpoint must point to an exported TorchScript policy.")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.robots.actuator import DelayedImplicitActuatorCfg

import isaaclab_tasks  # noqa: F401
import booster_train.tasks  # noqa: F401


LEGACY_K1_ACTION_SCALE = {
    ".*_Hip_Pitch": 0.09375,
    ".*_Hip_Roll": 0.109375,
    ".*_Hip_Yaw": 0.0625,
    ".*_Knee_Pitch": 0.125,
    ".*_Ankle_Pitch": 1.0 / 6.0,
    ".*_Ankle_Roll": 1.0 / 6.0,
    ".*_Shoulder_Pitch": 0.875,
    ".*_Shoulder_Roll": 0.875,
    ".*_Elbow_Pitch": 0.875,
    ".*_Elbow_Yaw": 0.875,
    ".*Head.*": 0.375,
}


def make_legacy_k1_cfg() -> ArticulationCfg:
    """Return the pre-2026-04-02 K1 actuator configuration used by deploy."""

    armature_6416 = 0.095625
    armature_4310 = 0.0282528
    armature_6408 = 0.0478125
    armature_4315 = 0.0339552

    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=False,
            asset_path=f"{BOOSTER_ASSETS_DIR}/robots/K1/K1_22dof.urdf",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.57),
            joint_pos={
                "Left_Shoulder_Roll": -1.3,
                "Right_Shoulder_Roll": 1.3,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": DelayedImplicitActuatorCfg(
                max_delay=8,
                min_delay=2,
                joint_names_expr=[
                    ".*_Hip_Pitch",
                    ".*_Hip_Roll",
                    ".*_Hip_Yaw",
                    ".*_Knee_Pitch",
                ],
                effort_limit_sim={
                    ".*_Hip_Pitch": 30.0,
                    ".*_Hip_Roll": 35.0,
                    ".*_Hip_Yaw": 20.0,
                    ".*_Knee_Pitch": 40.0,
                },
                velocity_limit_sim={
                    ".*_Hip_Pitch": 8.0,
                    ".*_Hip_Roll": 12.9,
                    ".*_Hip_Yaw": 18.0,
                    ".*_Knee_Pitch": 12.5,
                },
                stiffness={
                    ".*_Hip_Pitch": 80.0,
                    ".*_Hip_Roll": 80.0,
                    ".*_Hip_Yaw": 80.0,
                    ".*_Knee_Pitch": 80.0,
                },
                damping={
                    ".*_Hip_Pitch": 2.0,
                    ".*_Hip_Roll": 2.0,
                    ".*_Hip_Yaw": 2.0,
                    ".*_Knee_Pitch": 2.0,
                },
                armature={
                    ".*_Hip_Pitch": armature_6408,
                    ".*_Hip_Roll": armature_4315,
                    ".*_Hip_Yaw": armature_4310,
                    ".*_Knee_Pitch": armature_6416,
                },
            ),
            "feet": DelayedImplicitActuatorCfg(
                max_delay=8,
                min_delay=2,
                effort_limit_sim=20.0,
                velocity_limit_sim=18.0,
                joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
                stiffness=30.0,
                damping=2.0,
                armature=2.0 * armature_4310,
            ),
            "arms": DelayedImplicitActuatorCfg(
                max_delay=8,
                min_delay=2,
                joint_names_expr=[
                    ".*_Shoulder_Pitch",
                    ".*_Shoulder_Roll",
                    ".*_Elbow_Pitch",
                    ".*_Elbow_Yaw",
                ],
                effort_limit_sim=14.0,
                velocity_limit_sim=18.0,
                stiffness=4.0,
                damping=1.0,
                armature=0.001,
            ),
            "head": DelayedImplicitActuatorCfg(
                max_delay=8,
                min_delay=2,
                joint_names_expr=[".*Head.*"],
                effort_limit_sim=6.0,
                velocity_limit_sim=20.0,
                stiffness=4.0,
                damping=1.0,
                armature=0.001,
            ),
        },
    )


def disable_randomization(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    """Disable random perturbations that make deploy comparisons noisy."""

    if not hasattr(env_cfg, "events") or not hasattr(env_cfg, "commands"):
        return

    env_cfg.events.physics_material = None
    env_cfg.events.add_joint_default_pos = None
    env_cfg.events.base_com = None
    env_cfg.commands.motion.pose_range = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    env_cfg.commands.motion.velocity_range = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)


def apply_k1_profile(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, profile: str) -> None:
    if profile != "legacy":
        return

    env_cfg.scene.robot = make_legacy_k1_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.actions.joint_pos.scale = LEGACY_K1_ACTION_SCALE


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Run the diagnostic rollout."""

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    apply_k1_profile(env_cfg, args_cli.k1_profile)
    if args_cli.disable_randomization:
        disable_randomization(env_cfg)

    policy_path = retrieve_file_path(args_cli.checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading TorchScript policy from: {policy_path}")
    policy = torch.jit.load(policy_path, map_location=env.unwrapped.device)
    policy.to(env.unwrapped.device).eval()
    if hasattr(policy, "reset"):
        policy.reset()

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = obs

    done_count = 0
    first_done = None
    root_z_min = float("inf")
    root_z_last = None

    for step in range(args_cli.max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        root_z = float(env.unwrapped.scene["robot"].data.root_pos_w[0, 2].detach().cpu())
        root_z_min = min(root_z_min, root_z)
        root_z_last = root_z

        if torch.as_tensor(dones).any().item():
            done_count += int(torch.as_tensor(dones).sum().item())
            if first_done is None:
                first_done = step

    result = {
        "checkpoint": policy_path,
        "task": args_cli.task,
        "k1_profile": args_cli.k1_profile,
        "disable_randomization": args_cli.disable_randomization,
        "num_envs": args_cli.num_envs,
        "max_steps": args_cli.max_steps,
        "done_count": done_count,
        "first_done": first_done,
        "root_z_min": root_z_min,
        "root_z_last": root_z_last,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
