"""Look-up of the shuffle-walk gait library (no simulator imports).

The library (booster project-memos/walk-leg-clearance-20261016/scripts/gait_library.py) holds one 0.5 s cycle per (vx, vy, wz)
grid point, 25 samples at 50 Hz, solved periodically in the path frame (the frame moving with the commanded velocity). Phase 0 is
the left lift-off, 0.5 the right one. The look-up is trilinear in the command (clamped to the grid) and linear in the phase, as
Disney's BDX interpolates densely sampled gait references during RL.
"""
from pathlib import Path

import numpy as np
import torch

FIELDS = ("joint_pos", "joint_vel", "torso_pos", "torso_quat", "torso_lin_vel_b", "torso_ang_vel_b", "contact")


class GaitLibrary:
    def __init__(self, path: str | Path, joint_names: list[str], device):
        z = np.load(path)
        names = [str(n) for n in z["joint_names"]]
        order = [names.index(n) for n in joint_names]                    # library order -> robot order
        self.axes = [torch.tensor(z[k], dtype=torch.float32, device=device) for k in ("vx", "vy", "wz")]
        self.cycle_s = float(z["cycle_s"])
        parts = {k: z[k].astype(np.float32) for k in FIELDS}
        parts["joint_pos"], parts["joint_vel"] = parts["joint_pos"][..., order], parts["joint_vel"][..., order]
        self.widths = {k: parts[k].shape[-1] for k in FIELDS}
        self.shape = parts["joint_pos"].shape[:4]                         # (nvx, nvy, nwz, samples)
        flat = np.concatenate([parts[k].reshape(*self.shape, -1) for k in FIELDS], axis=-1)
        self.data = torch.tensor(flat.reshape(-1, flat.shape[-1]), device=device)
        self.ranges = [(float(a[0]), float(a[-1])) for a in self.axes]

    def _axis(self, a, x):
        n = a.numel(); u = ((x - a[0]) / (a[-1] - a[0]) * (n - 1)).clamp(0, n - 1 - 1e-6)
        i = u.floor().long(); return i, u - i

    def sample(self, command: torch.Tensor, phase: torch.Tensor) -> dict:
        """command (N, 3) = (vx, vy, wz) in the path frame, phase (N,) in [0, 1)."""
        (i, fi), (j, fj), (k, fk) = (self._axis(a, command[:, c]) for c, a in enumerate(self.axes))
        ns = self.shape[3]; s = (phase % 1.0) * ns; p0 = s.floor().long() % ns; fp = s - s.floor(); p1 = (p0 + 1) % ns
        out = 0.0
        for di, wi in ((0, 1 - fi), (1, fi)):
            for dj, wj in ((0, 1 - fj), (1, fj)):
                for dk, wk in ((0, 1 - fk), (1, fk)):
                    base = (((i + di) * self.shape[1] + (j + dj)) * self.shape[2] + (k + dk)) * ns
                    w = (wi * wj * wk)[:, None]
                    out = out + w * ((1 - fp)[:, None] * self.data[base + p0] + fp[:, None] * self.data[base + p1])
        result, at = {}, 0
        for key in FIELDS:
            result[key] = out[:, at:at + self.widths[key]]; at += self.widths[key]
        result["torso_quat"] = result["torso_quat"] / result["torso_quat"].norm(dim=-1, keepdim=True)
        return result
