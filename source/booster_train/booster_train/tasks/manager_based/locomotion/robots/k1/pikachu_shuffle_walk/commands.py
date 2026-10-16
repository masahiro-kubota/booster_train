"""Velocity command and the gait reference (phase, path frame, library look-up) of the shuffle walk."""
import math
from dataclasses import MISSING

import numpy as np
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, wrap_to_pi, euler_xyz_from_quat

from ..pikachu_idle_walk.commands import SmoothVelocityCommand, SmoothVelocityCommandCfg
from .gait import GaitLibrary


class GaitVelocityCommand(SmoothVelocityCommand):
    """SmoothVelocityCommand that starts part of the episodes already walking at the sampled command."""

    def reset(self, env_ids=None):
        result = super().reset(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        moving = ids[torch.rand(ids.numel(), device=self.device) < self.cfg.moving_start_fraction]
        self.current[moving] = self.target[moving]; self.start[moving] = self.target[moving]; self.delay[moving] = 0.
        return result


@configclass
class GaitVelocityCommandCfg(SmoothVelocityCommandCfg):
    class_type: type = GaitVelocityCommand
    moving_start_fraction: float = 0.6


class ShuffleGaitCommand(CommandTerm):
    """Gait phase and path frame driven by the velocity command; samples the library reference every step.

    The path frame integrates the commanded velocity. If the robot falls behind by more than path_slack_m (or path_slack_yaw),
    the path frame is pulled toward it, so the reference stays near the robot without a position estimate on the real robot.
    At reset the robot is written to the reference at a random phase, or, for standing starts with probability
    idle_start_fraction, to the approved Idle frame 0 (reset_pose.npz) so that the first steps out of Idle are learned.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.library = GaitLibrary(cfg.library_file, self.robot.joint_names, self.device)
        n = self.num_envs
        self.phase = torch.zeros(n, device=self.device)
        self.path_pos = torch.zeros((n, 2), device=self.device)
        self.path_yaw = torch.zeros(n, device=self.device)
        with np.load(cfg.idle_pose_file, allow_pickle=False) as z:
            names = z["joint_names"].tolist()
            self.idle_joint_pos = torch.tensor([float(z["joint_pos"][names.index(j)]) for j in self.robot.joint_names], device=self.device)
            self.idle_root_z = float(z["root_pos"][2]); self.idle_root_quat = torch.tensor(z["root_quat"], dtype=torch.float32, device=self.device)
        self.ref = self.library.sample(torch.zeros((n, 3), device=self.device), self.phase)
        self.ref_root_pos_w = torch.zeros((n, 3), device=self.device)
        self.ref_root_quat_w = torch.zeros((n, 4), device=self.device); self.ref_root_quat_w[:, 0] = 1.
        self.leg_ids = self.robot.find_joints(r"^(Left|Right)_(Hip|Knee|Ankle)_.*")[0]
        self.metrics["leg_joint_error_rad"] = torch.zeros(n, device=self.device)
        self.metrics["torso_xy_error_m"] = torch.zeros(n, device=self.device)

    @property
    def command(self):
        return torch.stack((torch.sin(2 * math.pi * self.phase), torch.cos(2 * math.pi * self.phase)), dim=-1)

    def velocity(self):
        return self._env.command_manager.get_command(self.cfg.velocity_command)

    def _path_quat(self, yaw):
        zero = torch.zeros_like(yaw)
        return quat_from_euler_xyz(zero, zero, yaw)

    def _world(self):
        q = self._path_quat(self.path_yaw)
        rel = self.ref["torso_pos"]
        xy = self.path_pos + quat_apply(q, torch.cat((rel[:, :2], torch.zeros_like(rel[:, :1])), -1))[:, :2]
        self.ref_root_pos_w = torch.cat((xy, (self._env.scene.env_origins[:, 2] + rel[:, 2])[:, None]), -1)
        self.ref_root_quat_w = quat_mul(q, self.ref["torso_quat"])

    def _resample_command(self, env_ids):
        # Called at episode reset only (resampling_time_range is effectively infinite).
        ids = torch.as_tensor(env_ids, device=self.device)
        cmd = self.velocity()[ids]
        standing = cmd.abs().amax(-1) < 1e-6
        idle = standing & (torch.rand(ids.numel(), device=self.device) < self.cfg.idle_start_fraction)
        self.phase[ids] = torch.where(idle, torch.zeros_like(self.phase[ids]), torch.rand(ids.numel(), device=self.device))
        self.path_yaw[ids] = (torch.rand(ids.numel(), device=self.device) * 2 - 1) * math.pi
        self.path_pos[ids] = self._env.scene.env_origins[ids, :2]
        ref = self.library.sample(cmd, self.phase[ids])
        q = self._path_quat(self.path_yaw[ids])
        rel = ref["torso_pos"]
        pos = torch.cat((self.path_pos[ids] + quat_apply(q, torch.cat((rel[:, :2], torch.zeros_like(rel[:, :1])), -1))[:, :2],
                         (self._env.scene.env_origins[ids, 2] + rel[:, 2])[:, None]), -1)
        quat = quat_mul(q, ref["torso_quat"])
        lin, ang = quat_apply(quat, ref["torso_lin_vel_b"]), quat_apply(quat, ref["torso_ang_vel_b"])
        joint_pos, joint_vel = ref["joint_pos"].clone(), ref["joint_vel"].clone()
        if idle.any():
            k = idle.nonzero().squeeze(-1)
            pos[k, :2] = self.path_pos[ids][k]; pos[k, 2] = self._env.scene.env_origins[ids[k], 2] + self.idle_root_z
            quat[k] = quat_mul(q[k], self.idle_root_quat.expand(k.numel(), 4))
            lin[k] = 0.; ang[k] = 0.; joint_pos[k] = self.idle_joint_pos; joint_vel[k] = 0.
        limits = self.robot.data.soft_joint_pos_limits[ids]
        joint_pos = torch.max(torch.min(joint_pos, limits[..., 1]), limits[..., 0])
        self.robot.write_root_state_to_sim(torch.cat((pos, quat, lin, ang), -1), env_ids=ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
        self.ref = self.library.sample(self.velocity(), self.phase); self._world()

    def _update_command(self):
        cmd, dt = self.velocity(), self._env.step_dt
        self.phase = (self.phase + dt / self.library.cycle_s) % 1.0
        self.path_yaw = wrap_to_pi(self.path_yaw + cmd[:, 2] * dt)
        q = self._path_quat(self.path_yaw)
        self.path_pos = self.path_pos + quat_apply(q, torch.cat((cmd[:, :2], torch.zeros_like(cmd[:, :1])), -1))[:, :2] * dt
        self.ref = self.library.sample(cmd, self.phase); self._world()
        # keep the path frame within reach of the robot
        err = self.robot.data.root_pos_w[:, :2] - self.ref_root_pos_w[:, :2]; dist = err.norm(dim=-1, keepdim=True)
        self.path_pos = self.path_pos + err * (1 - self.cfg.path_slack_m / dist.clamp(min=1e-6)).clamp(min=0.)
        _, _, yaw_r = euler_xyz_from_quat(self.robot.data.root_quat_w); _, _, yaw_ref = euler_xyz_from_quat(self.ref_root_quat_w)
        dyaw = wrap_to_pi(yaw_r - yaw_ref)
        self.path_yaw = wrap_to_pi(self.path_yaw + dyaw - dyaw.clamp(-self.cfg.path_slack_yaw, self.cfg.path_slack_yaw))
        self._world()

    def _update_metrics(self):
        self.metrics["leg_joint_error_rad"][:] = (self.robot.data.joint_pos - self.ref["joint_pos"])[:, self.leg_ids].abs().mean(-1)
        self.metrics["torso_xy_error_m"][:] = (self.robot.data.root_pos_w[:, :2] - self.ref_root_pos_w[:, :2]).norm(dim=-1)


@configclass
class ShuffleGaitCommandCfg(CommandTermCfg):
    class_type: type = ShuffleGaitCommand
    asset_name: str = "robot"
    library_file: str = MISSING
    idle_pose_file: str = MISSING
    velocity_command: str = "velocity"
    idle_start_fraction: float = 0.5
    path_slack_m: float = 0.15
    path_slack_yaw: float = 0.3
    resampling_time_range: tuple = (1.0e9, 1.0e9)
