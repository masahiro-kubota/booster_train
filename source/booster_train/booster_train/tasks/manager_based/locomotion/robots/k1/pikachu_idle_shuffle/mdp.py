"""Rewards, observations and terminations of the idle-and-shuffle policy: the shuffle walk's terms on the blended reference
(the command term is named "gait" as in the shuffle walk) and the reference joint positions and velocities as observations."""
from ..pikachu_shuffle_walk.mdp import (FEET, contact_match, fallen, feet_slide, foot_contact, gait, head_orientation,  # noqa: F401
                                        hip_flex_excess, joint_tracking, neck_frame, path_heading, reference_contact,
                                        torso_height, torso_lin_vel_xy, torso_orientation, torso_pos_in_reference, torso_position_xy,
                                        torso_yaw_rate)


def reference_joint_pos(env):
    """Reference joint positions relative to the default (the action offset), as the joint position observation."""
    return gait(env).ref["joint_pos"] - env.scene["robot"].data.default_joint_pos


def reference_joint_vel(env):
    return gait(env).ref["joint_vel"]
