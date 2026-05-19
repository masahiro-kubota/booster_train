# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version
import json
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import booster_train.tasks  # noqa: F401


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _action_cfg(env_cfg):
    actions_cfg = getattr(env_cfg, "actions", None)
    return getattr(actions_cfg, "joint_pos", None)


def _velocity_command_ranges(env_cfg):
    commands_cfg = getattr(env_cfg, "commands", None)
    command_cfg = getattr(commands_cfg, "base_velocity", None)
    ranges_cfg = getattr(command_cfg, "ranges", None)
    if ranges_cfg is None:
        return None

    ranges = {}
    for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"):
        value = getattr(ranges_cfg, name, None)
        if value is not None:
            ranges[name] = value
    return ranges


def _policy_observation_history_length(env_cfg):
    history_length = getattr(env_cfg, "observation_history_length", None)
    if history_length is not None:
        return history_length

    observations_cfg = getattr(env_cfg, "observations", None)
    policy_cfg = getattr(observations_cfg, "policy", None)
    if policy_cfg is None:
        return None

    term_history_lengths = []
    for name in dir(policy_cfg):
        if name.startswith("_") or name in ("concatenate_terms", "concatenate_dim", "enable_corruption"):
            continue
        term_cfg = getattr(policy_cfg, name)
        if hasattr(term_cfg, "history_length"):
            term_history_lengths.append(getattr(term_cfg, "history_length", 0))

    if len(term_history_lengths) == 1:
        return term_history_lengths[0]
    if term_history_lengths and len(set(term_history_lengths)) == 1:
        return term_history_lengths[0]
    return _jsonable(term_history_lengths) if term_history_lengths else None


def _policy_manifest(env_cfg, agent_cfg, args_cli, resume_path: str, jit_filename: str, onnx_filename: str):
    robot_cfg = getattr(env_cfg.scene, "robot", None)
    joint_pos_action_cfg = _action_cfg(env_cfg)
    action_joint_names = getattr(env_cfg, "action_joint_names", None) or getattr(
        joint_pos_action_cfg, "joint_names", None
    )
    actuators = getattr(robot_cfg, "actuators", {}) if robot_cfg is not None else {}
    actuator_manifest = {}
    for name, actuator_cfg in actuators.items():
        actuator_manifest[name] = {
            "class_type": getattr(
                getattr(actuator_cfg, "class_type", None),
                "__name__",
                repr(getattr(actuator_cfg, "class_type", None)),
            ),
            "joint_names_expr": _jsonable(getattr(actuator_cfg, "joint_names_expr", None)),
            "effort_limit_sim": _jsonable(getattr(actuator_cfg, "effort_limit_sim", None)),
            "velocity_limit_sim": _jsonable(getattr(actuator_cfg, "velocity_limit_sim", None)),
            "stiffness": _jsonable(getattr(actuator_cfg, "stiffness", None)),
            "damping": _jsonable(getattr(actuator_cfg, "damping", None)),
            "armature": _jsonable(getattr(actuator_cfg, "armature", None)),
        }

    return {
        "task": args_cli.task,
        "experiment_name": agent_cfg.experiment_name,
        "source_checkpoint": resume_path,
        "exported_jit": jit_filename,
        "exported_onnx": onnx_filename,
        "motion_file": _jsonable(getattr(getattr(env_cfg.commands, "motion", None), "motion_file", None)),
        "action_scale": _jsonable(getattr(joint_pos_action_cfg, "scale", None)),
        "action_joint_names": _jsonable(action_joint_names),
        "action_preserve_order": _jsonable(getattr(joint_pos_action_cfg, "preserve_order", None)),
        "actor_output_dim": len(action_joint_names) if action_joint_names is not None else None,
        "observation_history_length": _jsonable(_policy_observation_history_length(env_cfg)),
        "single_frame_observation_dim": _jsonable(getattr(env_cfg, "single_frame_observation_dim", None)),
        "exported_observation_dim": _jsonable(getattr(env_cfg, "exported_observation_dim", None)),
        "velocity_command_ranges": _jsonable(_velocity_command_ranges(env_cfg)),
        "robot_asset_path": _jsonable(getattr(getattr(robot_cfg, "spawn", None), "asset_path", None)),
        "actuators": actuator_manifest,
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    elif hasattr(ppo_runner, "obs_normalizer"):     # compatibility for older versions
        normalizer = ppo_runner.obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    run_name = os.path.basename(log_dir)
    jit_filename = f"{agent_cfg.experiment_name}_{run_name}.pt"
    onnx_filename = f"{agent_cfg.experiment_name}_{run_name}.onnx"
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename=jit_filename)
    export_policy_as_onnx(
        policy_nn, normalizer=normalizer, path=export_model_dir,
        filename=onnx_filename
    )
    manifest_path = os.path.join(export_model_dir, f"{agent_cfg.experiment_name}_{run_name}.manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            _policy_manifest(env_cfg, agent_cfg, args_cli, resume_path, jit_filename, onnx_filename),
            f,
            indent=2,
        )
    print(f"[INFO]: Exported policy manifest to: {manifest_path}")

    if args_cli.headless and not args_cli.video:
        print("[INFO] Headless mode and no video recording. Exiting after model export.")
        env.close()
        return

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = obs
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
