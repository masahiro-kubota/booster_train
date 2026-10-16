"""The idle-and-shuffle reference as a command term: path frame, reference (reference.py) and the reset states."""
from dataclasses import MISSING

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz, quat_mul, wrap_to_pi

from ..pikachu_shuffle_walk.gait import GaitLibrary
from .reference import MOVING, IdleMotion, IdleWalkReference

ARM_JOINTS = ["ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
              "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw"]
ARM_LIMIT_MARGIN_DEG = 3.0


class IdleShuffleCommand(CommandTerm):
    """Reference of the single policy (the Idle motion standing, the gait library walking) placed in the path frame.

    The path frame integrates the commanded velocity and is pulled toward the robot beyond path_slack_m (path_slack_yaw), as
    in the shuffle walk. At reset an environment with a moving command starts walking at a random gait phase; the others
    start standing at a random Idle frame. The robot is written to the reference with joint noise (reset_joint_noise).
    Arm references are kept arm_limit_margin_deg inside the joint limits, so that holding them does not press the arms
    against their stops (the Idle pose has the shoulder rolls and elbows on their limits).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.library = GaitLibrary(cfg.library_file, self.robot.joint_names, self.device)
        self.motion = IdleMotion(cfg.motion_file, self.robot.joint_names, self.device)
        n = self.num_envs
        self.reference = IdleWalkReference(self.library, self.motion, n, self.device)
        self.path_pos = torch.zeros((n, 2), device=self.device)
        self.path_yaw = torch.zeros(n, device=self.device)
        self.arm_ids = self.robot.find_joints(cfg.arm_joints, preserve_order=True)[0]
        self.leg_ids = self.robot.find_joints(r"^(Left|Right)_(Hip|Knee|Ankle)_.*")[0]
        self._sample(torch.zeros((n, 3), device=self.device))
        self.ref_root_pos_w = torch.zeros((n, 3), device=self.device)
        self.ref_root_quat_w = torch.zeros((n, 4), device=self.device); self.ref_root_quat_w[:, 0] = 1.
        self.metrics["leg_joint_error_rad"] = torch.zeros(n, device=self.device)
        self.metrics["torso_xy_error_m"] = torch.zeros(n, device=self.device)
        self.metrics["gait_weight"] = torch.zeros(n, device=self.device)

    @property
    def command(self):
        return self.reference.clock()

    def velocity(self):
        return self._env.command_manager.get_command(self.cfg.velocity_command)

    def _sample(self, cmd):
        self.ref = self.reference.sample(cmd)
        limits = self.robot.data.soft_joint_pos_limits[:, self.arm_ids]
        margin = torch.deg2rad(torch.tensor(self.cfg.arm_limit_margin_deg, device=self.device))
        arms = self.ref["joint_pos"][:, self.arm_ids]
        self.ref["joint_pos"][:, self.arm_ids] = torch.max(torch.min(arms, limits[..., 1] - margin), limits[..., 0] + margin)

    def _path_quat(self, yaw):
        zero = torch.zeros_like(yaw)
        return quat_from_euler_xyz(zero, zero, yaw)

    def _world(self):
        q = self._path_quat(self.path_yaw); rel = self.ref["torso_pos"]
        xy = self.path_pos + quat_apply(q, torch.cat((rel[:, :2], torch.zeros_like(rel[:, :1])), -1))[:, :2]
        self.ref_root_pos_w = torch.cat((xy, (self._env.scene.env_origins[:, 2] + rel[:, 2])[:, None]), -1)
        self.ref_root_quat_w = quat_mul(q, self.ref["torso_quat"])

    def _resample_command(self, env_ids):
        # Called at episode reset only (resampling_time_range is effectively infinite).
        ids = torch.as_tensor(env_ids, device=self.device); m = ids.numel()
        moving = self.velocity()[ids].abs().amax(-1) > MOVING
        self.reference.reset(ids, moving, torch.rand(m, device=self.device),
                             torch.randint(0, self.motion.START + self.motion.CYCLE, (m,), device=self.device))
        self.path_yaw[ids] = (torch.rand(m, device=self.device) * 2 - 1) * torch.pi
        self.path_pos[ids] = self._env.scene.env_origins[ids, :2]
        self._sample(self.velocity()); self._world()
        quat, pos = self.ref_root_quat_w[ids], self.ref_root_pos_w[ids]
        lin, ang = quat_apply(quat, self.ref["torso_lin_vel_b"][ids]), quat_apply(quat, self.ref["torso_ang_vel_b"][ids])
        noise = (torch.rand((m, len(self.robot.joint_names)), device=self.device) * 2 - 1) * self.cfg.reset_joint_noise
        limits = self.robot.data.soft_joint_pos_limits[ids]
        joint_pos = torch.max(torch.min(self.ref["joint_pos"][ids] + noise, limits[..., 1]), limits[..., 0])
        self.robot.write_root_state_to_sim(torch.cat((pos, quat, lin, ang), -1), env_ids=ids)
        self.robot.write_joint_state_to_sim(joint_pos, self.ref["joint_vel"][ids].clone(), env_ids=ids)

    def _update_command(self):
        cmd, dt = self.velocity(), self._env.step_dt
        self.reference.step(cmd, dt)
        self.path_yaw = wrap_to_pi(self.path_yaw + cmd[:, 2] * dt)
        q = self._path_quat(self.path_yaw)
        self.path_pos = self.path_pos + quat_apply(q, torch.cat((cmd[:, :2], torch.zeros_like(cmd[:, :1])), -1))[:, :2] * dt
        self._sample(cmd); self._world()
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
        self.metrics["gait_weight"][:] = self.reference.alpha


@configclass
class IdleShuffleCommandCfg(CommandTermCfg):
    class_type: type = IdleShuffleCommand
    asset_name: str = "robot"
    library_file: str = MISSING
    motion_file: str = MISSING
    velocity_command: str = "velocity"
    arm_joints: list = ARM_JOINTS
    arm_limit_margin_deg: float = ARM_LIMIT_MARGIN_DEG
    reset_joint_noise: float = 0.05
    path_slack_m: float = 0.15
    path_slack_yaw: float = 0.3
    resampling_time_range: tuple = (1.0e9, 1.0e9)
