#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.append(
    str(
        Path(__file__).resolve().parents[2]
        / "source"
        / "booster_train"
        / "booster_train"
        / "tasks"
        / "manager_based"
        / "beyond_mimic"
    )
)

from diffusion import (  # noqa: E402
    ACTION_KEY,
    BODY_POS_KEY,
    BODY_VELOCITY_KEY,
    COMMAND_KEY,
    JOINT_STATE_KEY,
    LAST_ACTION_KEY,
    ROOT_VELOCITY_KEY,
    STATE_KEY,
    TEACHER_ID_KEY,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply K1 sagittal left/right symmetry augmentation to rollout shards.")
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suffix", default="mirror")
    return parser.parse_args()


def _swap_name(name: str) -> str:
    replacements = (
        ("Left", "__TMP_RIGHT__"),
        ("Right", "Left"),
        ("__TMP_RIGHT__", "Right"),
        ("left", "__TMP_RIGHT_LOWER__"),
        ("right", "left"),
        ("__TMP_RIGHT_LOWER__", "right"),
    )
    result = name
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def _swap_indices(names: list[str]) -> list[int]:
    return [names.index(_swap_name(name)) if _swap_name(name) in names else index for index, name in enumerate(names)]


def _joint_signs(names: list[str]) -> np.ndarray:
    signs = np.ones(len(names), dtype=np.float32)
    for index, name in enumerate(names):
        if "Roll" in name or "Yaw" in name:
            signs[index] = -1.0
    return signs


def _mirror_state(state: np.ndarray, layout: dict[str, list[int]], action_swap: list[int], action_signs: np.ndarray) -> np.ndarray:
    mirrored = state.copy()
    root_pos = slice(*layout["root_pos_rel"])
    root_rot = slice(*layout["root_rot_rel_6d"])
    root_lin = slice(*layout["root_lin_vel_rel"])
    root_ang = slice(*layout["root_ang_vel_rel"])
    body_pos = slice(*layout["body_pos_local"])
    body_vel = slice(*layout["body_velocity_local"])
    gravity = slice(*layout["projected_gravity"])
    joint_pos = slice(*layout["joint_pos_rel"])
    joint_vel = slice(*layout["joint_vel"])
    last_action = slice(*layout["last_action"])

    mirrored[..., root_pos][..., 1] *= -1.0
    rot6 = mirrored[..., root_rot].reshape(state.shape[0], 3, 2)
    col0 = rot6[..., 0]
    col1 = rot6[..., 1]
    col2 = np.cross(col0, col1)
    rot = np.stack((col0, col1, col2), axis=-1)
    mirror = np.diag([1.0, -1.0, 1.0]).astype(np.float32)
    mirrored_rot = mirror @ rot @ mirror
    mirrored[..., root_rot] = mirrored_rot[..., :2].reshape(state.shape[0], 6)
    mirrored[..., root_lin][..., 1] *= -1.0
    mirrored[..., root_ang][..., [0, 2]] *= -1.0
    body_pos_values = mirrored[..., body_pos].reshape(state.shape[0], -1, 3)
    body_pos_values[..., 1] *= -1.0
    mirrored[..., body_pos] = body_pos_values.reshape(state.shape[0], -1)
    body_vel_values = mirrored[..., body_vel].reshape(state.shape[0], -1, 3)
    body_vel_values[..., 1] *= -1.0
    mirrored[..., body_vel] = body_vel_values.reshape(state.shape[0], -1)
    mirrored[..., gravity][..., 1] *= -1.0
    mirrored[..., joint_pos] = mirrored[..., joint_pos][..., action_swap] * action_signs
    mirrored[..., joint_vel] = mirrored[..., joint_vel][..., action_swap] * action_signs
    mirrored[..., last_action] = mirrored[..., last_action][..., action_swap] * action_signs
    return mirrored


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    action_names = list(manifest["action_joint_names"])
    body_names = list(manifest.get("body_names") or [])
    action_swap = _swap_indices(action_names)
    body_swap = _swap_indices(body_names) if body_names else None
    action_signs = _joint_signs(action_names)
    layout = manifest["state_layout"]

    for shard_path in args.shards:
        path = Path(shard_path)
        with np.load(path, allow_pickle=False) as data:
            shard = {key: data[key] for key in data.files}
        shard[STATE_KEY] = _mirror_state(shard[STATE_KEY], layout, action_swap, action_signs)
        shard[ACTION_KEY] = shard[ACTION_KEY][:, action_swap] * action_signs
        shard[LAST_ACTION_KEY] = shard[LAST_ACTION_KEY][:, action_swap] * action_signs
        if JOINT_STATE_KEY in shard:
            q = shard[JOINT_STATE_KEY][:, : len(action_names)]
            qd = shard[JOINT_STATE_KEY][:, len(action_names) :]
            shard[JOINT_STATE_KEY] = np.concatenate((q[:, action_swap] * action_signs, qd[:, action_swap] * action_signs), axis=-1)
        if BODY_POS_KEY in shard:
            shard[BODY_POS_KEY] = shard[BODY_POS_KEY].copy()
            shard[BODY_POS_KEY][..., 1] *= -1.0
            if body_swap is not None:
                shard[BODY_POS_KEY] = shard[BODY_POS_KEY][:, body_swap]
        if BODY_VELOCITY_KEY in shard:
            shard[BODY_VELOCITY_KEY] = shard[BODY_VELOCITY_KEY].copy()
            shard[BODY_VELOCITY_KEY][..., 1] *= -1.0
            if body_swap is not None:
                shard[BODY_VELOCITY_KEY] = shard[BODY_VELOCITY_KEY][:, body_swap]
        if ROOT_VELOCITY_KEY in shard:
            shard[ROOT_VELOCITY_KEY] = shard[ROOT_VELOCITY_KEY].copy()
            shard[ROOT_VELOCITY_KEY][:, 1] *= -1.0
            shard[ROOT_VELOCITY_KEY][:, [3, 5]] *= -1.0
        if COMMAND_KEY in shard:
            shard[COMMAND_KEY] = shard[COMMAND_KEY].copy()
            shard[COMMAND_KEY][:, 1] *= -1.0
            shard[COMMAND_KEY][:, 2] *= -1.0
        if TEACHER_ID_KEY in shard:
            shard[TEACHER_ID_KEY] = np.asarray([f"{value}_{args.suffix}" for value in shard[TEACHER_ID_KEY]], dtype="U96")
        np.savez_compressed(output_dir / f"{path.stem}_{args.suffix}.npz", **shard)

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({**manifest, "source": f"{manifest.get('source', 'unknown')}:sagittal_symmetry"}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
