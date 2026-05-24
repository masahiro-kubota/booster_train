#!/usr/bin/env python3
from __future__ import annotations

"""Collect paper-style diffusion rollouts by executing a trained VAE policy."""

import argparse
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
    description="Collect VAE-policy rollouts with OU action noise for BeyondMimic diffusion training.",
)
parser.add_argument("--task", default="Booster-K1-Locomotion-v0-Play")
parser.add_argument("--vae-decoder", required=True, help="TorchScript vae_decoder.pt from train_vae.py/export.py.")
parser.add_argument("--normalizer", required=True, help="Normalizer JSON paired with the VAE.")
parser.add_argument("--output-dir", required=True)
parser.add_argument("--teacher-id", default="vae_ou")
parser.add_argument("--num_envs", "--num-envs", dest="num_envs", type=int, default=16)
parser.add_argument("--record-duration-s", type=float, default=2.5)
parser.add_argument("--min-episode-s", type=float, default=5.0)
parser.add_argument(
    "--record-stride",
    type=int,
    default=2,
    help="Save one sample every N env steps. Default 2 converts the existing 50 Hz K1 tasks to 25 Hz data.",
)
parser.add_argument(
    "--target-accepted-rollouts",
    type=int,
    default=None,
    help="Repeat collection batches until this many non-failed rollouts are saved.",
)
parser.add_argument("--max-batches", type=int, default=100000)
parser.add_argument("--latent-dim", type=int, default=32)
parser.add_argument("--latent-mode", choices=("zero", "normal"), default="normal")
parser.add_argument("--command-name", default="base_velocity")
parser.add_argument("--ou-theta", type=float, default=0.8)
parser.add_argument("--ou-mu", type=float, default=0.0)
parser.add_argument("--ou-sigma", type=float, default=0.1)
parser.add_argument("--ou-dt", type=float, default=1.0)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=1)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    parser.print_help()
    sys.exit(0)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

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
from booster_train.tasks.manager_based.locomotion.robots.k1.walk.env_cfg import K1_WALK_POLICY_JOINT_NAMES  # noqa: E402


def _load_normalizer(path: str | Path, device: torch.device | str) -> dict[str, torch.Tensor]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in data.items()}


def _joint_ids(asset, joint_names: list[str]) -> list[int]:
    missing = [name for name in joint_names if name not in asset.joint_names]
    if missing:
        raise ValueError(f"Robot asset is missing canonical K1 joints: {missing}")
    return [asset.joint_names.index(name) for name in joint_names]


def _action_joint_names(env_cfg, env) -> list[str]:
    action_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    names = getattr(action_cfg, "joint_names", None)
    if names is not None and all("*" not in name and "." not in name for name in names):
        return list(names)
    return list(env.unwrapped.scene["robot"].joint_names)


def _canonical_to_env_action(action: torch.Tensor, source_joint_names: list[str], canonical_joint_names: list[str]) -> torch.Tensor:
    missing = [name for name in canonical_joint_names if name not in source_joint_names]
    if missing:
        raise ValueError(f"VAE rollout task action space is missing canonical K1 joints: {missing}")
    if source_joint_names == canonical_joint_names:
        return action
    raw_action = torch.zeros(action.shape[0], len(source_joint_names), dtype=action.dtype, device=action.device)
    for canonical_index, name in enumerate(canonical_joint_names):
        raw_action[:, source_joint_names.index(name)] = action[:, canonical_index]
    return raw_action


def _root_velocity_w(asset) -> tuple[torch.Tensor, torch.Tensor]:
    root_lin_vel_w = getattr(asset.data, "root_lin_vel_w", None)
    root_ang_vel_w = getattr(asset.data, "root_ang_vel_w", None)
    if root_lin_vel_w is not None and root_ang_vel_w is not None:
        return root_lin_vel_w, root_ang_vel_w
    return (
        math_utils.quat_apply(asset.data.root_quat_w, asset.data.root_lin_vel_b),
        math_utils.quat_apply(asset.data.root_quat_w, asset.data.root_ang_vel_b),
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
    padded = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
    padded[:, : min(3, command.shape[-1])] = command[:, :3]
    return padded


def _ou_update(noise: torch.Tensor, theta: float, mu: float, sigma: float, dt: float) -> torch.Tensor:
    return noise + theta * (mu - noise) * dt + sigma * (dt**0.5) * torch.randn_like(noise)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    env.reset()

    device = env.unwrapped.device
    asset = env.unwrapped.scene["robot"]
    canonical_joint_names = list(K1_WALK_POLICY_JOINT_NAMES)
    source_action_joint_names = _action_joint_names(env_cfg, env)
    joint_ids = _joint_ids(asset, canonical_joint_names)
    action_dim = len(canonical_joint_names)
    num_envs = int(env.num_envs)
    record_stride = max(1, int(args_cli.record_stride))
    env_step_dt = float(env.unwrapped.step_dt)
    dataset_policy_dt = env_step_dt * record_stride
    record_steps = max(1, int(math.ceil(args_cli.record_duration_s / dataset_policy_dt)))
    record_env_steps = record_steps * record_stride
    min_env_steps = max(record_env_steps, int(math.ceil(args_cli.min_episode_s / env_step_dt)))

    decoder = torch.jit.load(str(args_cli.vae_decoder), map_location=device).eval()
    normalizer = _load_normalizer(args_cli.normalizer, device)

    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_accepted = int(args_cli.target_accepted_rollouts or num_envs)
    max_batches = 1 if args_cli.target_accepted_rollouts is None else int(args_cli.max_batches)
    accepted_count = 0
    rejected_count = 0
    kept_env_ids_by_batch: list[list[int]] = []
    last_arrays: dict[str, np.ndarray] | None = None

    for batch_index in range(max_batches):
        env.reset()
        last_action = torch.zeros(num_envs, action_dim, dtype=torch.float32, device=device)
        ou_noise = torch.zeros_like(last_action)
        failed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        states: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        root_pose: list[torch.Tensor] = []
        root_velocity: list[torch.Tensor] = []
        body_pos_w: list[torch.Tensor] = []
        body_velocity_w: list[torch.Tensor] = []
        joint_state: list[torch.Tensor] = []
        last_actions: list[torch.Tensor] = []
        commands: list[torch.Tensor] = []

        for step in range(min_env_steps):
            state = _build_state(asset, joint_ids, last_action)
            state_norm = (state - normalizer["state_mean"]) / normalizer["state_std"]
            if args_cli.latent_mode == "normal":
                latent = torch.randn(num_envs, args_cli.latent_dim, dtype=torch.float32, device=device)
            else:
                latent = torch.zeros(num_envs, args_cli.latent_dim, dtype=torch.float32, device=device)
            with torch.inference_mode():
                action_norm = decoder(state_norm, latent)
            action = action_norm * normalizer["action_std"] + normalizer["action_mean"]
            ou_noise = _ou_update(ou_noise, args_cli.ou_theta, args_cli.ou_mu, args_cli.ou_sigma, args_cli.ou_dt)
            noisy_action = action + ou_noise
            raw_action = _canonical_to_env_action(noisy_action, source_action_joint_names, canonical_joint_names)

            if step % record_stride == 0 and len(states) < record_steps:
                root_lin_vel_w, root_ang_vel_w = _root_velocity_w(asset)
                current_body_pos_w, current_body_velocity_w = _body_state_w(asset)
                states.append(state.detach().cpu())
                actions.append(noisy_action.detach().cpu())
                root_pose.append(torch.cat((asset.data.root_pos_w, asset.data.root_quat_w), dim=-1).detach().cpu())
                root_velocity.append(torch.cat((root_lin_vel_w, root_ang_vel_w), dim=-1).detach().cpu())
                body_pos_w.append(current_body_pos_w.detach().cpu())
                body_velocity_w.append(current_body_velocity_w.detach().cpu())
                joint_state.append(
                    torch.cat((asset.data.joint_pos[:, joint_ids], asset.data.joint_vel[:, joint_ids]), dim=-1).detach().cpu()
                )
                last_actions.append(last_action.detach().cpu())
                commands.append(_command(env, args_cli.command_name, num_envs, device).detach().cpu())

            _, _, dones, _ = env.step(raw_action)
            failed |= dones > 0
            last_action = noisy_action.detach()
            if torch.any(dones > 0):
                last_action = last_action.clone()
                ou_noise = ou_noise.clone()
                last_action[dones > 0] = 0.0
                ou_noise[dones > 0] = 0.0

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
        last_arrays = arrays
        kept_env_ids: list[int] = []
        for env_id in range(num_envs):
            if bool(failed[env_id].detach().cpu()):
                rejected_count += 1
                continue
            if accepted_count >= target_accepted:
                break
            shard = {key: value[:, env_id] for key, value in arrays.items()}
            shard[TEACHER_ID_KEY] = np.full((record_steps,), args_cli.teacher_id, dtype="U64")
            np.savez_compressed(output_dir / f"{args_cli.teacher_id}_{accepted_count:06d}.npz", **shard)
            kept_env_ids.append(env_id)
            accepted_count += 1
        kept_env_ids_by_batch.append(kept_env_ids)
        print(
            json.dumps(
                {
                    "batch": batch_index + 1,
                    "accepted_total": accepted_count,
                    "target_accepted": target_accepted,
                    "accepted_this_batch": len(kept_env_ids),
                    "failed_this_batch": int(torch.sum(failed).detach().cpu()),
                }
            ),
            flush=True,
        )
        if accepted_count >= target_accepted:
            break

    if last_arrays is None:
        raise RuntimeError("No VAE rollout batches were collected")

    manifest = DatasetManifest(
        version=2,
        state_dim=int(last_arrays[STATE_KEY].shape[-1]),
        action_dim=action_dim,
        action_joint_names=canonical_joint_names,
        state_layout=paper_state_layout(int(last_arrays[BODY_POS_KEY].shape[-2]), action_dim),
        policy_dt=dataset_policy_dt,
        source=f"isaaclab:{args_cli.task}:vae_ou",
        body_names=list(asset.body_names),
        num_bodies=int(last_arrays[BODY_POS_KEY].shape[-2]),
    )
    manifest.to_file(output_dir / "manifest.json")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_envs": num_envs,
                "target_accepted_rollouts": target_accepted,
                "accepted_rollouts": accepted_count,
                "rejected_rollouts": rejected_count,
                "batches": len(kept_env_ids_by_batch),
                "kept_env_ids_by_batch": kept_env_ids_by_batch,
                "record_steps": record_steps,
                "record_env_steps": record_env_steps,
                "min_env_steps": min_env_steps,
                "record_stride": record_stride,
                "env_step_dt": env_step_dt,
                "dataset_policy_dt": dataset_policy_dt,
                "latent_mode": args_cli.latent_mode,
                "ou": {
                    "theta": args_cli.ou_theta,
                    "mu": args_cli.ou_mu,
                    "sigma": args_cli.ou_sigma,
                    "dt": args_cli.ou_dt,
                },
            },
            f,
            indent=2,
        )
        f.write("\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
