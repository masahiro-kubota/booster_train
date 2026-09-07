"""Idle-conditioned upper body; velocity-conditioned legs with hard shell clearance."""
import math
import torch
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude, quat_inv, quat_mul, yaw_quat

from .collision import ShellClearance
from .guard import SubstepClearanceGuard
from .reference import UPPER_BODIES


def idle(env):
    return env.command_manager.get_term("idle")


def upper_joint_tracking(env, std=0.15):
    c = idle(env)
    error = env.scene["robot"].data.joint_pos[:, c.upper_ids] - c.sampled["joint_pos"][:, c.upper_ids]
    return torch.exp(-error.square().mean(-1)/std**2)


def upper_body_tracking(env, std=0.03, orientation=False):
    c, robot = idle(env), env.scene["robot"]
    ids = [robot.body_names.index(n) for n in UPPER_BODIES]
    root = robot.body_names.index("Trunk")
    q = robot.data.body_link_quat_w[:, root:root+1].expand(-1,len(ids),-1)
    if orientation:
        actual = quat_mul(quat_inv(q), robot.data.body_link_quat_w[:, ids])
        error = quat_error_magnitude(actual, c.sampled["body_quat"]).square().mean(-1)
    else:
        actual = quat_apply_inverse(q, robot.data.body_link_pos_w[:, ids]-robot.data.body_link_pos_w[:, root:root+1])
        error = (actual-c.sampled["body_pos"]).square().sum(-1).mean(-1)
    return torch.exp(-error/std**2)


def height_error(env):
    return env.scene["robot"].data.root_pos_w[:, 2]-env.scene.env_origins[:, 2]-idle(env).sampled["height"]


def height_tracking(env, std=0.025):
    return torch.exp(-height_error(env).square()/std**2)


def height_observation(env):
    return height_error(env).unsqueeze(-1)


def clearance_observation(env):
    return shell_clearance(env).unsqueeze(-1)


def tilt_error(env):
    actual = env.scene["robot"].data.projected_gravity_b
    target = idle(env).sampled["gravity"]
    return torch.acos((actual*target).sum(-1).clamp(-1.,1.))


def tilt_tracking(env, std=0.12):
    return torch.exp(-tilt_error(env).square()/std**2)


def velocity_tracking(env, std=0.05, yaw=False):
    robot = env.scene["robot"]
    target = env.command_manager.get_command("velocity")
    if yaw:
        error = (robot.data.root_ang_vel_w[:, 2]-target[:, 2]).square()
    else:
        velocity = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)
        error = (velocity[:, :2]-target[:, :2]).square().sum(-1)
    return torch.exp(-error/std**2)


def knees(env):
    robot = env.scene["robot"]
    return robot.data.joint_pos[:, [robot.joint_names.index(n) for n in ("Left_Knee_Pitch", "Right_Knee_Pitch")]]


def knee_range(env):
    q = knees(env)
    violation = (math.radians(70)-q).clamp(min=0)+(q-math.radians(105)).clamp(min=0)
    return (violation/math.radians(10)).square().sum(-1)


def shell_guard(env):
    if not hasattr(env, "_pikachu_shell_guard"):
        checker = ShellClearance(env.cfg.shell_directory, env.scene["robot"].body_names, env.device)
        env._pikachu_shell_guard = SubstepClearanceGuard(checker,env.num_envs)
    return env._pikachu_shell_guard


def shell_clearance(env):
    """Include the final physics sample in the minimum latched by action substeps."""
    robot=env.scene['robot']
    return shell_guard(env).observe(robot.data.body_link_pos_w,robot.data.body_link_quat_w,env._sim_step_counter)


def shell_proximity(env, margin=0.01):
    return ((margin-shell_clearance(env))/margin).clamp(min=0).square()


def shell_unsafe(env, threshold=0.002):
    return shell_clearance(env) <= threshold


def posture_failure(env):
    return (height_error(env).abs() > 0.12) | (tilt_error(env) > 0.6)


def foot_contact(env):
    sensor = env.scene["contact_forces"]
    ids = [sensor.body_names.index(n) for n in ("left_foot_link", "right_foot_link")]
    return sensor.data.net_forces_w_history[:, :, ids].norm(dim=-1).amax(dim=1) > 10.


def feet_slide(env):
    robot = env.scene["robot"]
    ids = [robot.body_names.index(n) for n in ("left_foot_link", "right_foot_link")]
    return (robot.data.body_link_lin_vel_w[:, ids, :2].square().sum(-1)*foot_contact(env)).sum(-1)


def flight(env):
    return (~foot_contact(env).any(-1)).float()


def biped_air_time(env):
    sensor = env.scene["contact_forces"]
    ids = [sensor.body_names.index(n) for n in ("left_foot_link", "right_foot_link")]
    contact = foot_contact(env)
    time = torch.where(contact, sensor.data.current_contact_time[:, ids], sensor.data.current_air_time[:, ids])
    reward = time.amin(-1).clamp(max=0.4)*(contact.sum(-1)==1)
    command = env.command_manager.get_command("velocity")
    moving = (command[:, :2].norm(dim=-1) > 0.01) | (command[:, 2].abs() > 0.03)
    return reward*moving


def reset_idle(env, env_ids):
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    # Original Idle crouch, verified against the current 400 mm shell.
    q = robot.data.default_joint_pos[env_ids].clone()
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    root[:, 7:] = 0.
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
    if hasattr(env,'_pikachu_shell_guard'):
        env._pikachu_shell_guard.reset(env_ids)
