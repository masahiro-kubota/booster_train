"""Independent Idle clock and smoothly changing locomotion commands."""
from dataclasses import MISSING

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from .reference import IdleReference


class IdleLoopCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.reference = IdleReference(cfg.motion_file, self.robot.joint_names, self.device,
                                       initial_hold_s=cfg.initial_hold_s)
        self.elapsed = torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
        self.upper_ids = self.reference.upper_ids
        self.sampled = self.reference.sample(self.elapsed.float())
        self.metrics["phase"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["height_error_m"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self):
        s = self.sampled
        return torch.cat((s["joint_pos"][:, self.upper_ids], s["joint_vel"][:, self.upper_ids],
                          s["height"][:, None], s["gravity"],
                          torch.sin(2*torch.pi*s["phase"])[:, None], torch.cos(2*torch.pi*s["phase"])[:, None]), dim=-1)

    def _resample_command(self, env_ids):
        # Called only at episode reset: reference looping never enters this method.
        self.elapsed[env_ids] = 0.
        self.sampled = self.reference.sample(self.elapsed.float())

    def _update_command(self):
        self.elapsed += self._env.step_dt
        self.sampled = self.reference.sample(self.elapsed.float())

    def _update_metrics(self):
        self.metrics["phase"][:] = self.sampled["phase"]
        height = self.robot.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        self.metrics["height_error_m"][:] = (height - self.sampled["height"]).abs()


@configclass
class IdleLoopCommandCfg(CommandTermCfg):
    class_type: type = IdleLoopCommand
    asset_name: str = "robot"
    motion_file: str = MISSING
    initial_hold_s: float = 0.5
    resampling_time_range: tuple = (1.0e9, 1.0e9)


class SmoothVelocityCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.current = torch.zeros((self.num_envs, 3), device=self.device)
        self.start = self.current.clone()
        self.target = self.current.clone()
        self.ramp_time = torch.zeros(self.num_envs, device=self.device)
        self.delay = self.ramp_time.clone()
        self.metrics["velocity_xy_error_mps"] = self.ramp_time.clone()
        self.metrics["yaw_rate_error_radps"] = self.ramp_time.clone()

    @property
    def command(self):
        return self.current

    def set_target(self, env_ids, target):
        self.start[env_ids] = self.current[env_ids]
        self.target[env_ids] = target
        self.ramp_time[env_ids] = 0.
        self.delay[env_ids] = 0.

    def reset(self, env_ids=None):
        # ManagerBasedRLEnv clears episode_length_buf AFTER manager resets, so
        # that buffer cannot identify reset inside _resample_command.
        env_ids = slice(None) if env_ids is None else env_ids
        self.current[env_ids] = 0.
        result = super().reset(env_ids)
        self.start[env_ids] = 0.
        self.delay[env_ids] = self.cfg.initial_stand_s
        return result

    def _resample_command(self, env_ids):
        count = self.current[env_ids].shape[0]
        target = torch.empty((count, 3), device=self.device)
        for i, bounds in enumerate((self.cfg.forward_range, self.cfg.lateral_range, self.cfg.yaw_range)):
            target[:, i].uniform_(*bounds)
        target[torch.rand(count, device=self.device) < self.cfg.standing_fraction] = 0.
        self.set_target(env_ids, target)

    def _update_command(self):
        self.ramp_time += self._env.step_dt
        u = ((self.ramp_time-self.delay)/self.cfg.ramp_s).clamp(0., 1.)[:, None]
        blend = u*u*u*(10.-15.*u+6.*u*u)
        self.current[:] = self.start + blend*(self.target-self.start)

    def _update_metrics(self):
        vel = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), self.robot.data.root_lin_vel_w)
        self.metrics["velocity_xy_error_mps"][:] = (vel[:, :2]-self.current[:, :2]).norm(dim=-1)
        self.metrics["yaw_rate_error_radps"][:] = (self.robot.data.root_ang_vel_w[:, 2]-self.current[:, 2]).abs()


@configclass
class SmoothVelocityCommandCfg(CommandTermCfg):
    class_type: type = SmoothVelocityCommand
    asset_name: str = "robot"
    resampling_time_range: tuple = (3.0, 6.0)
    forward_range: tuple = (-0.10, 0.10)
    lateral_range: tuple = (-0.05, 0.05)
    yaw_range: tuple = (-0.25, 0.25)
    standing_fraction: float = 0.2
    initial_stand_s: float = 2.0
    ramp_s: float = 1.0
