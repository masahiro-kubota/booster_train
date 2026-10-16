"""Reference of the Pikachu idle-and-shuffle policy (no simulator imports, so the deploy side runs the same code).

The neck and arms always follow the approved Idle motion, walking or not. The legs and the torso follow the Idle motion while
the velocity command is zero and the shuffle-walk gait library while it is not, blended over a short time at each change.
The switches depend only on the velocity command and time:
- standing to walking: as soon as the command is non-zero, the gait starts at phase 0 (left lift-off);
- walking to standing: once the command is zero, at the next double support of the zero-command gait.
Over BLEND_S after a switch the legs and torso move from the pose they left (the double-support gait pose is held; the Idle
motion keeps running) to the new one with a smoothstep weight. The Idle motion is its 0.5 s hold at frame 0 followed by its
205-frame ping-pong cycle (frames 25 to 229), stepped one frame per 20 ms policy step and never restarted. Torso poses are in
the gait library's path frame: the origin under the Idle frame 0 torso, x along its heading.
"""
from pathlib import Path

import numpy as np
import torch

try:
    from ..pikachu_shuffle_walk.gait import FIELDS, GaitLibrary
except ImportError:            # loaded by file path on the deploy side
    import importlib.util
    _path = Path(__file__).resolve().parents[1] / "pikachu_shuffle_walk" / "gait.py"
    _spec = importlib.util.spec_from_file_location("shuffle_gait", _path)
    _gait = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_gait)
    FIELDS, GaitLibrary = _gait.FIELDS, _gait.GaitLibrary

BLEND_S = 0.4
MOVING = 1e-4                  # |command| above this counts as walking


def _yaw(q):
    return np.arctan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2))


def _qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0); w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), -1)


def _rotate_inverse(q, v):
    xyz = -q[..., 1:]; t = 2 * np.cross(xyz, v)
    return v + q[..., :1] * t + np.cross(xyz, t)


class IdleMotion:
    """The Idle motion's frames 0 to 229 as references in the path frame (joint order of joint_names)."""
    START, CYCLE = 25, 205

    def __init__(self, path: str | Path, joint_names: list[str], device):
        with np.load(path, allow_pickle=False) as z:
            if float(np.ravel(z["fps"])[0]) != 50.0:
                raise ValueError("expected the 50 Hz Idle motion")
            names = z["joint_names"].tolist(); order = [names.index(n) for n in joint_names]
            self.upper_ids = [i for i, n in enumerate(joint_names) if not any(s in n for s in ("Hip", "Knee", "Ankle"))]
            trunk = z["body_names"].tolist().index("Trunk")
            n = self.START + self.CYCLE
            jp, jv = z["joint_pos"][:n, order], z["joint_vel"][:n, order]
            p, q = z["body_pos_w"][:n, trunk].astype(np.float64), z["body_quat_w"][:n, trunk].astype(np.float64)
            v, w = z["body_lin_vel_w"][:n, trunk], z["body_ang_vel_w"][:n, trunk]
            later = z["joint_pos"][n:n + self.CYCLE, order]
        if not (np.allclose(jp[:self.START + 1], jp[0], atol=1e-5) and np.allclose(later, jp[self.START:], atol=1e-5)):
            raise ValueError("the Idle motion is not a hold at frame 0 followed by a repeated 205-frame cycle")
        half = -0.5 * _yaw(q[0]); q0 = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
        rel = _rotate_inverse(np.broadcast_to(q0 * [1, -1, -1, -1], q.shape), p - p[0]); rel[:, 2] = p[:, 2]
        parts = {"joint_pos": jp, "joint_vel": jv, "torso_pos": rel, "torso_quat": _qmul(np.broadcast_to(q0, q.shape), q),
                 "torso_lin_vel_b": _rotate_inverse(q, v), "torso_ang_vel_b": _rotate_inverse(q, w), "contact": np.ones((n, 2))}
        self.data = {k: torch.tensor(np.asarray(x, np.float32), device=device) for k, x in parts.items()}

    def frame(self, k: torch.Tensor) -> torch.Tensor:
        """Motion frame for k policy steps after the Idle motion (re)started."""
        return torch.where(k <= self.START, k, self.START + (k - self.START) % self.CYCLE)

    def phase(self, k: torch.Tensor) -> torch.Tensor:
        """Position in the ping-pong cycle, 0 during the hold."""
        f = self.frame(k)
        return torch.where(f <= self.START, torch.zeros_like(f, dtype=torch.float32), (f - self.START).float() / self.CYCLE)

    def sample(self, k: torch.Tensor) -> dict:
        f = self.frame(k)
        return {key: x[f] for key, x in self.data.items()}


def _blend(gait: dict, idle: dict, w: torch.Tensor) -> dict:
    w1 = w[:, None]; out = {}
    for key in FIELDS:
        if key == "torso_quat":
            qi = idle[key] * torch.where((gait[key] * idle[key]).sum(-1, keepdim=True) < 0, -1.0, 1.0)
            q = w1 * gait[key] + (1 - w1) * qi; out[key] = q / q.norm(dim=-1, keepdim=True)
        elif key == "contact":
            out[key] = torch.where(w1 > 0.5, gait[key], idle[key])
        else:
            out[key] = w1 * gait[key] + (1 - w1) * idle[key]
    return out


class IdleWalkReference:
    """Per-environment state of the reference: walking or standing, gait phase, Idle step and blend weight of the legs and
    torso."""

    def __init__(self, library: GaitLibrary, motion: IdleMotion, num: int, device):
        self.library, self.motion, self.device = library, motion, device
        self.walking = torch.zeros(num, dtype=torch.bool, device=device)
        self.phase = torch.zeros(num, device=device)
        self.idle_k = torch.zeros(num, dtype=torch.long, device=device)
        self.alpha = torch.zeros(num, device=device)                 # 0 Idle, 1 gait (linear in time; smoothstep when used)

    def reset(self, ids, walking: torch.Tensor, phase: torch.Tensor, idle_k: torch.Tensor):
        self.walking[ids] = walking; self.phase[ids] = phase; self.idle_k[ids] = idle_k; self.alpha[ids] = walking.float()

    def double_support(self) -> torch.Tensor:
        zero = torch.zeros((self.phase.numel(), 3), device=self.device)
        return (self.library.sample(zero, self.phase)["contact"] > 0.5).all(-1)

    def step(self, command: torch.Tensor, dt: float):
        """Advance one policy step with the (smoothed) velocity command; switches as described in the module docstring."""
        dt = float(dt); moving = command.abs().amax(-1) > MOVING
        start = ~self.walking & moving
        stop = self.walking & ~moving & self.double_support()
        self.walking = (self.walking | start) & ~stop
        self.phase = torch.where(start, torch.zeros_like(self.phase), self.phase)
        self.phase = torch.where(self.walking, (self.phase + dt / self.library.cycle_s) % 1.0, self.phase)
        self.idle_k = self.idle_k + 1
        rate = torch.full_like(self.alpha, dt / BLEND_S)
        self.alpha = (self.alpha + torch.where(self.walking, rate, -rate)).clamp(0.0, 1.0)

    def sample(self, command: torch.Tensor) -> dict:
        gait_cmd = torch.where(self.walking[:, None], command, torch.zeros_like(command))
        a = self.alpha; idle = self.motion.sample(self.idle_k)
        out = _blend(self.library.sample(gait_cmd, self.phase), idle, a * a * (3 - 2 * a))
        for key in ("joint_pos", "joint_vel"):
            out[key][:, self.motion.upper_ids] = idle[key][:, self.motion.upper_ids]
        return out

    def clock(self) -> torch.Tensor:
        """sin, cos of the gait phase and of the Idle cycle position, and the blend weight (policy observation)."""
        g, i = 2 * torch.pi * self.phase, 2 * torch.pi * self.motion.phase(self.idle_k)
        return torch.stack((torch.sin(g), torch.cos(g), torch.sin(i), torch.cos(i), self.alpha), -1)
