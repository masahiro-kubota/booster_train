from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import ACTION_KEY, STATE_KEY


def _paths(paths: Iterable[str | Path]) -> list[Path]:
    result = [Path(path) for path in paths]
    if not result:
        raise ValueError("At least one rollout shard is required")
    return result


def _load_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if key not in data:
            raise KeyError(f"{path} is missing required array '{key}'")
        return np.asarray(data[key], dtype=np.float32)


def _as_2d(array: np.ndarray, path: Path, key: str) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"{path}:{key} must be 2D, got shape {array.shape}")
    return array


class RolloutDataset(Dataset):
    """Flat state/action dataset backed by one or more .npz rollout shards."""

    def __init__(
        self,
        paths: Iterable[str | Path],
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
        action_mean: np.ndarray | None = None,
        action_std: np.ndarray | None = None,
    ) -> None:
        self.paths = _paths(paths)
        states = []
        actions = []
        for path in self.paths:
            state = _as_2d(_load_array(path, STATE_KEY), path, STATE_KEY)
            action = _as_2d(_load_array(path, ACTION_KEY), path, ACTION_KEY)
            if state.shape[0] != action.shape[0]:
                raise ValueError(f"{path} state/action length mismatch: {state.shape[0]} != {action.shape[0]}")
            states.append(state)
            actions.append(action)

        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.state_mean = np.zeros(self.states.shape[1], dtype=np.float32) if state_mean is None else state_mean
        self.state_std = np.ones(self.states.shape[1], dtype=np.float32) if state_std is None else state_std
        self.action_mean = np.zeros(self.actions.shape[1], dtype=np.float32) if action_mean is None else action_mean
        self.action_std = np.ones(self.actions.shape[1], dtype=np.float32) if action_std is None else action_std

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        state = (self.states[index] - self.state_mean) / self.state_std
        action = (self.actions[index] - self.action_mean) / self.action_std
        return {
            "state": torch.tensor(state.astype(np.float32).tolist(), dtype=torch.float32),
            "action": torch.tensor(action.astype(np.float32).tolist(), dtype=torch.float32),
        }


class TrajectoryWindowDataset(Dataset):
    """Windowed normalized states/actions for diffusion training."""

    def __init__(
        self,
        paths: Iterable[str | Path],
        window_length: int,
        state_mean: np.ndarray,
        state_std: np.ndarray,
    ) -> None:
        if window_length < 2:
            raise ValueError("window_length must be >= 2")
        self.paths = _paths(paths)
        self.window_length = int(window_length)
        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)
        self.action_mean: np.ndarray | None = None
        self.action_std: np.ndarray | None = None
        self.trajectories: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.index: list[tuple[int, int]] = []

        for traj_id, path in enumerate(self.paths):
            state = _as_2d(_load_array(path, STATE_KEY), path, STATE_KEY)
            action = _as_2d(_load_array(path, ACTION_KEY), path, ACTION_KEY)
            if state.shape[0] != action.shape[0]:
                raise ValueError(f"{path} state/action length mismatch: {state.shape[0]} != {action.shape[0]}")
            if state.shape[0] < self.window_length:
                continue
            state = (state - self.state_mean) / self.state_std
            self.trajectories.append(state.astype(np.float32))
            self.actions.append(action.astype(np.float32))
            for start in range(0, state.shape[0] - self.window_length + 1):
                self.index.append((traj_id, start))

        if not self.index:
            raise ValueError("No diffusion windows could be built from the provided shards")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        traj_id, start = self.index[index]
        window = self.trajectories[traj_id][start:start + self.window_length]
        action_window = self.actions[traj_id][start:start + self.window_length]
        return {
            "state": torch.tensor(window.astype(np.float32).tolist(), dtype=torch.float32),
            "action": torch.tensor(action_window.astype(np.float32).tolist(), dtype=torch.float32),
        }


def compute_normalizer(paths: Iterable[str | Path], eps: float = 1.0e-6) -> dict[str, list[float]]:
    paths = _paths(paths)
    states = []
    actions = []
    for path in paths:
        states.append(_as_2d(_load_array(path, STATE_KEY), path, STATE_KEY))
        actions.append(_as_2d(_load_array(path, ACTION_KEY), path, ACTION_KEY))

    state = np.concatenate(states, axis=0).astype(np.float32)
    action = np.concatenate(actions, axis=0).astype(np.float32)
    return {
        "state_mean": state.mean(axis=0).tolist(),
        "state_std": np.maximum(state.std(axis=0), eps).tolist(),
        "action_mean": action.mean(axis=0).tolist(),
        "action_std": np.maximum(action.std(axis=0), eps).tolist(),
    }


def save_normalizer(normalizer: dict[str, list[float]], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalizer, f, indent=2, sort_keys=True)
        f.write("\n")


def load_normalizer(path: str | Path) -> dict[str, np.ndarray]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {key: np.asarray(value, dtype=np.float32) for key, value in data.items()}
