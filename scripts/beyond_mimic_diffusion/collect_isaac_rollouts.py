#!/usr/bin/env python3
from __future__ import annotations

"""Collect Isaac Lab teacher rollouts for BeyondMimic diffusion training."""

import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
BOOSTER_TRAIN_DIR = SCRIPT_DIR.parents[1]
sys.path.append(str(BOOSTER_TRAIN_DIR / "scripts" / "rsl_rl"))
sys.path.append(str(BOOSTER_TRAIN_DIR / "source" / "booster_train"))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(
    description="Collect teacher rollouts from an Isaac Lab task for BeyondMimic diffusion.",
)
parser.add_argument("--task", default="Booster-K1-Locomotion-v0-Play")
parser.add_argument("--output-dir", default=None)
parser.add_argument("--teacher-id", default="k1_locomotion")
parser.add_argument(
    "--teacher-type",
    choices=("auto", "rsl_rl", "jit"),
    default="auto",
    help="Use an RSL-RL checkpoint, an exported TorchScript policy, or auto-detect.",
)
parser.add_argument("--num_envs", "--num-envs", dest="num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1000, help="Environment steps to run when --record-steps is not set.")
parser.add_argument(
    "--record-steps",
    type=int,
    default=None,
    help="Number of saved samples per env. If set, env steps = record_steps * record_stride.",
)
parser.add_argument(
    "--record-stride",
    type=int,
    default=2,
    help="Save one sample every N env steps. Default 2 converts the existing 50 Hz K1 tasks to 25 Hz data.",
)
parser.add_argument("--command-name", default="base_velocity")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=1)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    parser.print_help()
    sys.exit(0)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.checkpoint is None:
    parser.error("--checkpoint must point to an RSL-RL checkpoint or an exported TorchScript policy.")
if args_cli.output_dir is None:
    parser.error("--output-dir is required.")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
import booster_train.tasks  # noqa: F401,E402
from booster_train.tasks.manager_based.beyond_mimic.diffusion.schema import (  # noqa: E402
    ACTION_KEY,
    BODY_POS_KEY,
    BODY_VELOCITY_KEY,
    COMMAND_KEY,
    JOINT_STATE_KEY,
    LAST_ACTION_KEY,
    ROOT_POSE_KEY,
    ROOT_VELOCITY_KEY,
    STATE_KEY,
    TEACHER_ID_KEY,
    DatasetManifest,
    paper_state_layout,
)
from booster_train.tasks.manager_based.beyond_mimic.diffusion.state import build_character_yaw_state  # noqa: E402
from booster_train.tasks.manager_based.locomotion.robots.k1.walk.env_cfg import (  # noqa: E402
    K1_WALK_POLICY_JOINT_NAMES,
)


def _joint_ids(asset, joint_names: list[str]) -> list[int]:
    missing = [name for name in joint_names if name not in asset.joint_names]
    if missing:
        raise ValueError(f"Robot asset is missing canonical K1 joints: {missing}")
    return [asset.joint_names.index(name) for name in joint_names]


def _action_joint_names(env_cfg, env) -> list[str]:
    names = getattr(env_cfg, "action_joint_names", None)
    if names is not None:
        return list(names)

    action_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    names = getattr(action_cfg, "joint_names", None)
    if names is not None and all("*" not in name and "." not in name for name in names):
        return list(names)

    return list(env.unwrapped.scene["robot"].joint_names)


def _canonical_action_indices(source_joint_names: list[str], canonical_joint_names: list[str]) -> list[int]:
    missing = [name for name in canonical_joint_names if name not in source_joint_names]
    if missing:
        raise ValueError(
            "Teacher action space cannot be projected to the canonical K1 joystick action space. "
            f"Missing joints: {missing}. Source action joints: {source_joint_names}"
        )
    return [source_joint_names.index(name) for name in canonical_joint_names]


def _root_velocity_w(asset) -> tuple[torch.Tensor, torch.Tensor]:
    root_lin_vel_w = getattr(asset.data, "root_lin_vel_w", None)
    root_ang_vel_w = getattr(asset.data, "root_ang_vel_w", None)
    if root_lin_vel_w is not None and root_ang_vel_w is not None:
        return root_lin_vel_w, root_ang_vel_w
    root_quat_w = asset.data.root_quat_w
    return (
        math_utils.quat_apply(root_quat_w, asset.data.root_lin_vel_b),
        math_utils.quat_apply(root_quat_w, asset.data.root_ang_vel_b),
    )


def _body_state_w(asset) -> tuple[torch.Tensor, torch.Tensor]:
    body_pos_w = getattr(asset.data, "body_pos_w")
    body_lin_vel_w = getattr(asset.data, "body_lin_vel_w", None)
    if body_lin_vel_w is None:
        body_lin_vel_w = torch.zeros_like(body_pos_w)
    return body_pos_w, body_lin_vel_w


def _build_state(asset, joint_ids: list[int], last_action: torch.Tensor) -> torch.Tensor:
    joint_pos = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    joint_vel = asset.data.joint_vel[:, joint_ids]
    root_lin_vel_w, root_ang_vel_w = _root_velocity_w(asset)
    body_pos_w, body_lin_vel_w = _body_state_w(asset)
    return build_character_yaw_state(
        root_pos_w=asset.data.root_pos_w,
        root_quat_w=asset.data.root_quat_w,
        root_lin_vel_w=root_lin_vel_w,
        root_ang_vel_w=root_ang_vel_w,
        body_pos_w=body_pos_w,
        body_lin_vel_w=body_lin_vel_w,
        joint_pos_rel=joint_pos,
        joint_vel=joint_vel,
        last_action=last_action,
    )


def _command(env, command_name: str, num_envs: int, device: torch.device) -> torch.Tensor:
    if not hasattr(env.unwrapped, "command_manager"):
        return torch.zeros(num_envs, 3, dtype=torch.float32, device=device)

    try:
        command = env.unwrapped.command_manager.get_command(command_name)
    except Exception:
        return torch.zeros(num_envs, 3, dtype=torch.float32, device=device)

    command = torch.as_tensor(command, dtype=torch.float32, device=device)
    if command.dim() == 1:
        command = command.view(1, -1).expand(num_envs, -1)
    if command.shape[-1] < 3:
        padded = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        padded[:, : command.shape[-1]] = command
        return padded
    return command[:, :3]


def _load_jit_policy(checkpoint_path: str, device: torch.device | str) -> torch.jit.ScriptModule:
    policy = torch.jit.load(checkpoint_path, map_location=device)
    policy.to(device).eval()
    if hasattr(policy, "reset"):
        policy.reset()
    return policy


def _load_teacher_policy(args: argparse.Namespace, env, agent_cfg, checkpoint_path: str):
    jit_error: Exception | None = None
    if args.teacher_type in ("auto", "jit"):
        try:
            policy = _load_jit_policy(checkpoint_path, env.unwrapped.device)
            print(f"[INFO]: Loading TorchScript teacher policy from: {checkpoint_path}")
            return policy, "jit"
        except Exception as exc:
            if args.teacher_type == "jit":
                raise
            jit_error = exc

    if args.teacher_type in ("auto", "rsl_rl"):
        print(f"[INFO]: Loading RSL-RL teacher checkpoint from: {checkpoint_path}")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        try:
            runner.load(checkpoint_path)
        except Exception as exc:
            if jit_error is not None:
                raise RuntimeError(
                    "Failed to auto-detect teacher checkpoint type. "
                    f"TorchScript load error: {jit_error}; RSL-RL load error: {exc}"
                ) from exc
            raise
        return runner.get_inference_policy(device=env.unwrapped.device), "rsl_rl"

    raise ValueError(f"Unsupported teacher type: {args.teacher_type}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    policy, resolved_teacher_type = _load_teacher_policy(args_cli, env, agent_cfg, checkpoint_path)

    env.reset()
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = obs

    asset = env.unwrapped.scene["robot"]
    canonical_action_joint_names = list(K1_WALK_POLICY_JOINT_NAMES)
    source_action_joint_names = _action_joint_names(env_cfg, env)
    action_projection = _canonical_action_indices(source_action_joint_names, canonical_action_joint_names)
    joint_ids = _joint_ids(asset, canonical_action_joint_names)
    num_envs = int(env.num_envs)
    action_dim = len(canonical_action_joint_names)
    last_action = torch.zeros(num_envs, action_dim, dtype=torch.float32, device=env.unwrapped.device)
    record_stride = max(1, int(args_cli.record_stride))
    if args_cli.record_steps is None:
        total_env_steps = int(args_cli.steps)
        target_record_steps = int(math.ceil(total_env_steps / record_stride))
    else:
        target_record_steps = int(args_cli.record_steps)
        total_env_steps = target_record_steps * record_stride

    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    root_pose: list[torch.Tensor] = []
    root_velocity: list[torch.Tensor] = []
    body_pos_w: list[torch.Tensor] = []
    body_velocity_w: list[torch.Tensor] = []
    joint_state: list[torch.Tensor] = []
    last_actions: list[torch.Tensor] = []
    commands: list[torch.Tensor] = []

    for step in range(total_env_steps):
        with torch.inference_mode():
            raw_action = policy(obs)
        if raw_action.dim() == 1:
            raw_action = raw_action.view(1, -1)
        action = raw_action[:, action_projection]
        state = _build_state(asset, joint_ids, last_action)
        if step % record_stride == 0 and len(states) < target_record_steps:
            root_lin_vel_w, root_ang_vel_w = _root_velocity_w(asset)
            current_body_pos_w, current_body_velocity_w = _body_state_w(asset)
            states.append(state.detach().cpu())
            actions.append(action.detach().cpu())
            root_pose.append(torch.cat((asset.data.root_pos_w, asset.data.root_quat_w), dim=-1).detach().cpu())
            root_velocity.append(torch.cat((root_lin_vel_w, root_ang_vel_w), dim=-1).detach().cpu())
            body_pos_w.append(current_body_pos_w.detach().cpu())
            body_velocity_w.append(current_body_velocity_w.detach().cpu())
            joint_state.append(
                torch.cat((asset.data.joint_pos[:, joint_ids], asset.data.joint_vel[:, joint_ids]), dim=-1).detach().cpu()
            )
            last_actions.append(last_action.detach().cpu())
            commands.append(_command(env, args_cli.command_name, num_envs, env.unwrapped.device).detach().cpu())

        obs, _, dones, _ = env.step(raw_action)
        last_action = action.detach()
        if torch.any(dones > 0):
            last_action = last_action.clone()
            last_action[dones > 0] = 0.0

    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        STATE_KEY: torch.stack(states, dim=0).numpy(),
        ACTION_KEY: torch.stack(actions, dim=0).numpy(),
        ROOT_POSE_KEY: torch.stack(root_pose, dim=0).numpy(),
        ROOT_VELOCITY_KEY: torch.stack(root_velocity, dim=0).numpy(),
        BODY_POS_KEY: torch.stack(body_pos_w, dim=0).numpy(),
        BODY_VELOCITY_KEY: torch.stack(body_velocity_w, dim=0).numpy(),
        JOINT_STATE_KEY: torch.stack(joint_state, dim=0).numpy(),
        LAST_ACTION_KEY: torch.stack(last_actions, dim=0).numpy(),
        COMMAND_KEY: torch.stack(commands, dim=0).numpy(),
    }
    recorded_steps = int(arrays[STATE_KEY].shape[0])

    for env_id in range(num_envs):
        shard = {key: value[:, env_id] for key, value in arrays.items()}
        shard[TEACHER_ID_KEY] = np.full((recorded_steps,), args_cli.teacher_id, dtype="U64")
        np.savez_compressed(output_dir / f"{args_cli.teacher_id}_{env_id:04d}.npz", **shard)

    manifest = DatasetManifest(
        version=2,
        state_dim=int(arrays[STATE_KEY].shape[-1]),
        action_dim=action_dim,
        action_joint_names=canonical_action_joint_names,
        state_layout=paper_state_layout(int(arrays[BODY_POS_KEY].shape[-2]), action_dim),
        policy_dt=float(env.unwrapped.step_dt) * record_stride,
        source=f"isaaclab:{args_cli.task}:{resolved_teacher_type}",
        body_names=list(asset.body_names),
        num_bodies=int(arrays[BODY_POS_KEY].shape[-2]),
    )
    manifest.to_file(output_dir / "manifest.json")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_envs": num_envs,
                "env_steps": total_env_steps,
                "recorded_steps": recorded_steps,
                "record_stride": record_stride,
                "env_step_dt": float(env.unwrapped.step_dt),
                "dataset_policy_dt": float(env.unwrapped.step_dt) * record_stride,
                "teacher_id": args_cli.teacher_id,
                "teacher_type": resolved_teacher_type,
                "checkpoint": checkpoint_path,
                "action_joint_names": canonical_action_joint_names,
                "source_action_joint_names": source_action_joint_names,
            },
            f,
            indent=2,
        )
        f.write("\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
