"""Periodic sampling of the *already ping-ponged* approved Idle (no simulator imports)."""
from pathlib import Path

import numpy as np
import torch

UPPER_BODIES = ["Head_2", "Left_Arm_2", "Left_Arm_3", "left_hand_link",
                "Right_Arm_2", "Right_Arm_3", "right_hand_link"]
LEG_BODIES = ["Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Shank", "Left_Ankle_Cross", "left_foot_link",
              "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Shank", "Right_Ankle_Cross", "right_foot_link"]


def inverse_rotate(q, v):
    """wxyz quaternion, arbitrary leading dimensions."""
    xyz = -q[..., 1:]
    t = 2 * np.cross(xyz, v)
    return v + q[..., :1] * t + np.cross(xyz, t)


def multiply(a, b):
    return np.concatenate((a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True),
                           a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])), axis=-1)


class IdleReference:
    """C1 periodic Hermite interpolation; node poses remain exactly the source poses.

    Harmonic, sign-preserving slopes avoid joint-limit overshoot and give zero velocity
    at reversals. Source NPZ velocities are intentionally not reused across its seams.
    The initial hold occurs once, not once per cycle. No robot state is written here.
    """

    def __init__(self, path: str | Path, joint_names=None, device="cpu", dtype=torch.float32,
                 cycle_start=25, cycle_samples=205, initial_hold_s=0.5):
        with np.load(path, allow_pickle=False) as d:
            self.fps = float(np.asarray(d["fps"]).reshape(-1)[0])
            source_names = d["joint_names"].tolist()
            self.joint_names = list(joint_names or source_names)
            ids = [source_names.index(n) for n in self.joint_names]
            self.upper_ids = [i for i, n in enumerate(self.joint_names)
                              if not any(s in n for s in ("Hip", "Knee", "Ankle"))]
            self.body_names = d["body_names"].tolist()
            root_id = self.body_names.index("Trunk")
            bid = [self.body_names.index(n) for n in UPPER_BODIES]
            sl = slice(cycle_start, cycle_start + cycle_samples)
            q = np.asarray(d["joint_pos"][sl][:, ids], dtype=np.float64)
            self.initial_joint_pos = np.array(d["joint_pos"][0, ids], copy=True)
            self.initial_root_pos = np.array(d["body_pos_w"][0, root_id], copy=True)
            self.initial_root_quat = np.array(d["body_quat_w"][0, root_id], copy=True)
            root_p = d["body_pos_w"][sl, root_id].astype(np.float64)
            root_q = d["body_quat_w"][sl, root_id].astype(np.float64)
            body_p = inverse_rotate(root_q[:, None, :], d["body_pos_w"][sl][:, bid] - root_p[:, None, :])
            inv = root_q.copy()
            inv[:, 1:] *= -1
            body_q = multiply(inv[:, None, :], d["body_quat_w"][sl][:, bid])
            for j in range(len(bid)):
                for i in range(1, len(body_q)):
                    if np.dot(body_q[i-1, j], body_q[i, j]) < 0:
                        body_q[i, j] *= -1
            gravity = inverse_rotate(root_q, np.broadcast_to([0., 0., -1.], root_p.shape))
            if len(q) != cycle_samples or self.fps != 50.0:
                raise ValueError("Expected the verified 205-sample, 50 Hz Scale0842 ping-pong cycle")
            other = d["joint_pos"][cycle_start + cycle_samples:cycle_start + 2*cycle_samples][:, ids]
            if other.shape != q.shape or not np.allclose(q, other, atol=2e-5, rtol=0):
                raise ValueError("Source does not contain the expected repeated ping-pong cycle")
            if not np.allclose(q[0], self.initial_joint_pos, atol=2e-5, rtol=0):
                raise ValueError("Initial hold and loop start disagree")
        self.n_joints = len(ids)
        self.n_bodies = len(bid)
        self.initial_hold_s = initial_hold_s
        self.period_s = cycle_samples / self.fps
        values = np.concatenate((q, root_p[:, 2:3], gravity, body_p.reshape(len(q), -1), body_q.reshape(len(q), -1)), axis=-1)
        before = (values - np.roll(values, 1, axis=0)) * self.fps
        after = (np.roll(values, -1, axis=0) - values) * self.fps
        slopes = np.zeros_like(values)
        same = before * after > 0
        slopes[same] = 2 * before[same] * after[same] / (before[same] + after[same])
        self.values = torch.tensor(values, device=device, dtype=dtype)
        self.slopes = torch.tensor(slopes, device=device, dtype=dtype)

    def sample(self, elapsed: torch.Tensor):
        x = torch.remainder(torch.clamp(elapsed - self.initial_hold_s, min=0) * self.fps, len(self.values))
        i = x.floor().long()
        u = (x - i).unsqueeze(-1)
        j = (i + 1) % len(self.values)
        a, b = self.values[i], self.values[j]
        va, vb = self.slopes[i], self.slopes[j]
        u2, u3 = u*u, u*u*u
        value = (2*u3-3*u2+1)*a + (u3-2*u2+u)*va/self.fps + (-2*u3+3*u2)*b + (u3-u2)*vb/self.fps
        velocity = ((6*u2-6*u)*a + (3*u2-4*u+1)*va/self.fps + (-6*u2+6*u)*b + (3*u2-2*u)*vb/self.fps)*self.fps
        velocity = torch.where((elapsed >= self.initial_hold_s).unsqueeze(-1), velocity, 0.)
        n = self.n_joints
        gravity = torch.nn.functional.normalize(value[..., n+1:n+4], dim=-1)
        bp = value[..., n+4:n+4+3*self.n_bodies].reshape(*elapsed.shape, self.n_bodies, 3)
        bq = value[..., n+4+3*self.n_bodies:].reshape(*elapsed.shape, self.n_bodies, 4)
        bq = torch.nn.functional.normalize(bq, dim=-1)
        return {"joint_pos": value[..., :n], "joint_vel": velocity[..., :n],
                "height": value[..., n], "gravity": gravity, "body_pos": bp, "body_quat": bq,
                "phase": x / len(self.values)}
