"""Run the Booster K1 Fight environment without starting RL training.

Launch with DISPLAY set externally when using the GUI, for example:
    DISPLAY=:1 python scripts/run_k1_fight_env.py
"""

from isaaclab.app import AppLauncher


DEVICE = "cuda:0"
HEADLESS = False

NUM_ENVS = 1
MAX_STEPS = 1000
RESET_INTERVAL = 0
PRINT_INTERVAL = 50
SEED = 42

CAMERA_ENV_INDEX = 0
CAMERA_EYE = (2.0, 2.0, 0.5)
CAMERA_LOOKAT = (0.0, 0.0, 0.0)


app_launcher = AppLauncher({"device": DEVICE, "headless": HEADLESS})
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv

from booster_train.tasks.manager_based.beyond_mimic.robots.k1.fight_001.env_cfg import FlatWoStateEstimationEnvCfg


def _configure_viewer(env_cfg: FlatWoStateEstimationEnvCfg) -> None:
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = CAMERA_ENV_INDEX
    env_cfg.viewer.eye = CAMERA_EYE
    env_cfg.viewer.lookat = CAMERA_LOOKAT


def _disable_terminations(env_cfg: FlatWoStateEstimationEnvCfg) -> None:
    env_cfg.terminations.time_out = None
    env_cfg.terminations.anchor_pos = None
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.ee_body_pos = None


def _disable_randomization(env_cfg: FlatWoStateEstimationEnvCfg) -> None:
    env_cfg.commands.motion.play = True
    env_cfg.commands.motion.debug_vis = False
    env_cfg.commands.motion.pose_range = {}
    env_cfg.commands.motion.velocity_range = {}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)

    env_cfg.events.physics_material = None
    env_cfg.events.add_joint_default_pos = None
    env_cfg.events.base_com = None
    env_cfg.events.push_robot = None


def main() -> None:
    env_cfg = FlatWoStateEstimationEnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.sim.device = DEVICE
    env_cfg.seed = SEED
    _configure_viewer(env_cfg)
    _disable_terminations(env_cfg)
    _disable_randomization(env_cfg)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    print(
        "[INFO]: Running Booster K1 Fight environment "
        f"num_envs={NUM_ENVS}, max_steps={MAX_STEPS}, device={DEVICE}, terrain=flat, "
        "action_mode=zero, randomization=off, terminations=off, camera_mode=robot"
    )

    count = 0
    try:
        with torch.inference_mode():
            while simulation_app.is_running():
                if MAX_STEPS > 0 and count >= MAX_STEPS:
                    break

                if RESET_INTERVAL > 0 and count > 0 and count % RESET_INTERVAL == 0:
                    env.reset()
                    print("-" * 80)
                    print(f"[INFO]: Resetting environment at step {count}.")

                actions = torch.zeros_like(env.action_manager.action)
                obs, rew, terminated, truncated, _ = env.step(actions)

                if PRINT_INTERVAL > 0 and count % PRINT_INTERVAL == 0:
                    policy_obs = obs["policy"]
                    done_count = int(torch.count_nonzero(terminated | truncated).item())
                    print(
                        f"[step {count:06d}] "
                        f"reward_env0={rew[0].item(): .5f} "
                        f"policy_obs_shape={tuple(policy_obs.shape)} "
                        f"done_count={done_count}"
                    )

                count += 1
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
