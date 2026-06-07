"""Launch the K1 port of the Unitree velocity task without training."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher


DEVICE = "cuda:0"
HEADLESS = False
NUM_ENVS = 1
MAX_STEPS = 240
PRINT_INTERVAL = 40


parser = argparse.ArgumentParser(description="Smoke-test the K1 port of the Unitree velocity environment.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.device = DEVICE
args_cli.headless = HEADLESS

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

from booster_train.tasks.manager_based.locomotion.robots.k1.unitree_velocity.env_cfg import K1UnitreeVelocityEnvCfg


def _make_smoke_cfg() -> K1UnitreeVelocityEnvCfg:
    env_cfg = K1UnitreeVelocityEnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.sim.device = DEVICE

    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.lin_vel_cmd_levels = None

    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

    env_cfg.events.physics_material = None
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.reset_base = None
    env_cfg.events.reset_robot_joints = None
    env_cfg.events.push_robot = None

    env_cfg.terminations.time_out = None
    env_cfg.terminations.base_height = None
    env_cfg.terminations.bad_orientation = None

    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (2.5, -3.0, 1.2)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.6)
    return env_cfg


def main():
    if not os.environ.get("DISPLAY"):
        print("[WARN] DISPLAY is not set. Run with `export DISPLAY=:1` for GUI.")

    env = ManagerBasedRLEnv(cfg=_make_smoke_cfg())
    env.reset()

    robot = env.scene["robot"]
    print("[INFO] body_names:", robot.body_names)
    print("[INFO] joint_names:", robot.joint_names)
    print("[INFO] action_shape:", tuple(env.action_manager.action.shape))

    count = 0
    with torch.inference_mode():
        while simulation_app.is_running() and count < MAX_STEPS:
            actions = torch.zeros_like(env.action_manager.action)
            obs, rew, terminated, truncated, _ = env.step(actions)
            if count % PRINT_INTERVAL == 0:
                done_count = int((terminated | truncated).sum().item())
                print(
                    f"[INFO] step={count:04d} "
                    f"reward0={rew[0].item():.4f} "
                    f"done_count={done_count} "
                    f"policy_shape={tuple(obs['policy'].shape)}"
                )
            count += 1

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
