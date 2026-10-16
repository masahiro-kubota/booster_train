"""Rewards, observations and terminations of the shuffle walk: joint-space imitation of the gait library (as BDX), the head
orientation as in the Idle task, and the hip-flexion limit that keeps the K1 thighs off the hard waist ring of the torso exterior."""
import math

import torch
from isaaclab.utils.math import (euler_xyz_from_quat, quat_apply_inverse, quat_error_magnitude, quat_from_euler_xyz, quat_inv,
                                 quat_mul, wrap_to_pi, yaw_quat)

FEET = ("left_foot_link", "right_foot_link")
LEGS = r"^(Left|Right)_(Hip|Knee|Ankle)_.*"

# Hip flexion (-Hip_Pitch) at which the thigh comes within 10 mm of the waist ring's groove lip, for the torso exterior S3-12
# (project-memos/walk-leg-clearance-20261016, knee-swing-map): about 60 deg at the Idle hip roll, falling 0.72 deg per degree
# the leg opens (abduction). Hip_Roll at Idle frame 0 is +29.1 deg (left) and -34.7 deg (right); opening is +roll on the left.
HIP_FLEX_LIMIT, HIP_FLEX_SLOPE = math.radians(60.0), 0.72
HIP_ROLL_IDLE = {"Left": math.radians(29.1), "Right": math.radians(-34.7)}


def gait(env):
    return env.command_manager.get_term("gait")


def _ids(env, names):
    return [env.scene["robot"].joint_names.index(n) for n in names]


def joint_tracking(env, joint_names: list[str], std: float):
    robot = env.scene["robot"]; ids = robot.find_joints(joint_names, preserve_order=True)[0]
    error = robot.data.joint_pos[:, ids] - gait(env).ref["joint_pos"][:, ids]
    return torch.exp(-error.square().mean(-1) / std ** 2)


def torso_orientation(env, std: float):
    return torch.exp(-quat_error_magnitude(env.scene["robot"].data.root_quat_w, gait(env).ref_root_quat_w).square() / std ** 2)


def torso_height(env, std: float):
    error = env.scene["robot"].data.root_pos_w[:, 2] - gait(env).ref_root_pos_w[:, 2]
    return torch.exp(-error.square() / std ** 2)


def torso_position_xy(env, std: float):
    error = env.scene["robot"].data.root_pos_w[:, :2] - gait(env).ref_root_pos_w[:, :2]
    return torch.exp(-error.square().sum(-1) / std ** 2)


def torso_lin_vel_xy(env, std: float):
    error = env.scene["robot"].data.root_lin_vel_b[:, :2] - gait(env).ref["torso_lin_vel_b"][:, :2]
    return torch.exp(-error.square().sum(-1) / std ** 2)


def torso_yaw_rate(env, std: float):
    error = env.scene["robot"].data.root_ang_vel_b[:, 2] - gait(env).ref["torso_ang_vel_b"][:, 2]
    return torch.exp(-error.square() / std ** 2)


def neck_frame(env, joint_pos):
    """Head_2 orientation relative to the torso for the given joint positions, Rz(AAHead_yaw) Ry(Head_pitch) (checked against the
    Trunk and Head_2 orientations of the Idle motion)."""
    yaw, pitch = _ids(env, ["AAHead_yaw", "Head_pitch"]); zero = torch.zeros_like(joint_pos[:, 0])
    return quat_mul(quat_from_euler_xyz(zero, zero, joint_pos[:, yaw]), quat_from_euler_xyz(zero, joint_pos[:, pitch], zero))


def head_orientation(env, std: float):
    """Head_2 orientation against the reference torso orientation turned by the reference neck joints. As the Idle task's
    motion_head_ori, the reference is turned by the yaw between the robot torso and the reference torso, so torso pitch and roll
    errors count but a yaw lag behind the path frame does not."""
    robot, g = env.scene["robot"], gait(env)
    delta = yaw_quat(quat_mul(robot.data.root_quat_w, quat_inv(g.ref_root_quat_w)))
    target = quat_mul(quat_mul(delta, g.ref_root_quat_w), neck_frame(env, g.ref["joint_pos"]))
    head = robot.body_names.index("Head_2")
    return torch.exp(-quat_error_magnitude(robot.data.body_quat_w[:, head], target).square() / std ** 2)


def foot_contact(env, threshold: float = 10.0):
    sensor = env.scene["contact_forces"]; ids = [sensor.body_names.index(n) for n in FEET]
    return sensor.data.net_forces_w_history[:, :, ids].norm(dim=-1).amax(dim=1) > threshold


def contact_match(env):
    return (foot_contact(env) == (gait(env).ref["contact"] > 0.5)).float().mean(-1)


def feet_slide(env):
    robot = env.scene["robot"]; ids = [robot.body_names.index(n) for n in FEET]
    return (robot.data.body_link_lin_vel_w[:, ids, :2].square().sum(-1) * foot_contact(env)).sum(-1)


def hip_flex_excess(env, margin_deg: float = 2.0):
    """Squared hip flexion beyond the waist-ring limit (rad^2), the limit lowered as the leg opens."""
    q = env.scene["robot"].data.joint_pos; total = 0.
    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        pitch, roll = _ids(env, [f"{side}_Hip_Pitch", f"{side}_Hip_Roll"])
        opening = sign * (q[:, roll] - HIP_ROLL_IDLE[side])
        limit = HIP_FLEX_LIMIT - HIP_FLEX_SLOPE * opening - math.radians(margin_deg)
        total = total + (-q[:, pitch] - limit).clamp(min=0.).square()
    return total


def path_heading(env):
    """sin and cos of the torso yaw relative to the reference torso yaw (yaw only; no position estimate needed)."""
    _, _, yaw = euler_xyz_from_quat(env.scene["robot"].data.root_quat_w); _, _, ref = euler_xyz_from_quat(gait(env).ref_root_quat_w)
    d = wrap_to_pi(yaw - ref)
    return torch.stack((torch.sin(d), torch.cos(d)), -1)


def torso_pos_in_reference(env):
    """Critic only: torso position error to the reference, in the reference torso yaw frame."""
    g = gait(env); err = env.scene["robot"].data.root_pos_w - g.ref_root_pos_w
    return quat_apply_inverse(g.ref_root_quat_w, err)


def reference_contact(env):
    return gait(env).ref["contact"]


def fallen(env, height_m: float = 0.12, tilt_rad: float = 0.6):
    robot, g = env.scene["robot"], gait(env)
    low = (g.ref_root_pos_w[:, 2] - robot.data.root_pos_w[:, 2]) > height_m
    gravity = robot.data.projected_gravity_b
    return low | (torch.acos((-gravity[:, 2]).clamp(-1., 1.)) > tilt_rad + 0.1)
