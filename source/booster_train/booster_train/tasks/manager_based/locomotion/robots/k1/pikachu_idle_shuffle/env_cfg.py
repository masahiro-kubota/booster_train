"""Pikachu idle and shuffle walk in one policy: the approved Idle motion while the velocity command is zero, the shuffle-walk
gait library while it is not, and the blends between them (reference.py), so the robot never steps in place when it stands.

Built on the shuffle walk (pikachu_shuffle_walk): the same rewards on the blended reference, the v1 head (HeadV1Sim settings),
the Idle frame 0 pose as the action offset and the hip-flexion limit for the torso exterior. Changes:
- the policy also sees the reference joint positions and velocities and a clock (gait phase, Idle cycle position, blend
  weight); proprioception keeps 5 steps of history, the reference terms only the current step;
- episodes start standing at a random Idle frame or walking at a random gait phase, with joint noise, and the velocity
  command switches between standing and walking several times per episode;
- the arm references stay 3 deg inside the joint limits and arm torque is penalized, so the arms are not held against their
  stops (the shuffle walk pressed the right shoulder roll at its 14 N m limit).
"""
import math
from pathlib import Path

import isaaclab.envs.mdp as base
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Noise
from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.robots.booster import BOOSTER_K1_CFG, K1_ACTION_SCALE
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842.tracking_env_cfg import TrackingEnvCfg

from ..pikachu_shuffle_walk.commands import GaitVelocityCommandCfg
from ..pikachu_shuffle_walk.env_cfg import LEG_JOINTS, LIBRARY, PUSH, UPPER_JOINTS, idle_default_joint_pos, use_v1_head
from . import mdp
from .commands import ARM_JOINTS, ARM_LIMIT_MARGIN_DEG, IdleShuffleCommandCfg

MOTION = f"{BOOSTER_ASSETS_DIR}/motions/K1/k1_pikachu_idle_scale0842_pingpong_tile3_blender50_grounded_hold0p5.npz"
ARM_MARGIN = math.radians(ARM_LIMIT_MARGIN_DEG)
# Torso exterior S3-12 (shell, cover plates, side and lower foam, clamp V19, bolts) on Trunk and the arm foam on the hand links:
# design volumes and assumed densities, not weighed (project-memos/walk-leg-clearance-20261016/scripts/exterior_mass_properties.py).
# Trunk 6.50 -> 8.28 kg (exterior 1.78 kg, 1.53..1.96 over the density range); the Trunk mass is varied by EXTERIOR_MASS_ADD_KG.
EXTERIOR_MASS_FILE = str(Path(__file__).parent / "exterior_s3-12_mass_properties.json")
EXTERIOR_MASS_ADD_KG = (-0.3, 0.3)


@configclass
class CommandsCfg:
    # The velocity term must come first: the reference term reads it at reset and every step.
    velocity = GaitVelocityCommandCfg(forward_range=(-0.06, 0.06), lateral_range=(-0.04, 0.04), yaw_range=(-0.2, 0.2),
                                      standing_fraction=0.35, moving_start_fraction=0.5, initial_stand_s=1.0, ramp_s=1.0,
                                      resampling_time_range=(3.0, 6.0))
    gait = IdleShuffleCommandCfg(library_file=LIBRARY, motion_file=MOTION)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        velocity_command = ObsTerm(func=base.generated_commands, params={"command_name": "velocity"})
        reference_clock = ObsTerm(func=base.generated_commands, params={"command_name": "gait"})
        heading = ObsTerm(func=mdp.path_heading)
        reference_joint_pos = ObsTerm(func=mdp.reference_joint_pos)
        reference_joint_vel = ObsTerm(func=mdp.reference_joint_vel)
        base_ang_vel = ObsTerm(func=base.base_ang_vel, noise=Noise(n_min=-0.2, n_max=0.2), history_length=5)
        gravity = ObsTerm(func=base.projected_gravity, noise=Noise(n_min=-0.05, n_max=0.05), history_length=5)
        joint_pos = ObsTerm(func=base.joint_pos_rel, noise=Noise(n_min=-0.01, n_max=0.01), history_length=5)
        joint_vel = ObsTerm(func=base.joint_vel_rel, noise=Noise(n_min=-0.5, n_max=0.5), history_length=5)
        actions = ObsTerm(func=base.last_action, history_length=5)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=base.base_lin_vel)
        torso_pos_error = ObsTerm(func=mdp.torso_pos_in_reference)
        reference_contact = ObsTerm(func=mdp.reference_contact)

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    policy = PolicyCfg()
    critic = CriticCfg()


@configclass
class RewardsCfg:
    leg_joints = RewTerm(func=mdp.joint_tracking, weight=15.0, params={"joint_names": LEG_JOINTS, "std": 0.15})
    upper_joints = RewTerm(func=mdp.joint_tracking, weight=5.0, params={"joint_names": UPPER_JOINTS, "std": 0.15})
    head_orientation = RewTerm(func=mdp.head_orientation, weight=2.5, params={"std": 0.2})
    torso_orientation = RewTerm(func=mdp.torso_orientation, weight=10.0, params={"std": 0.2})
    torso_height = RewTerm(func=mdp.torso_height, weight=5.0, params={"std": 0.03})
    torso_position = RewTerm(func=mdp.torso_position_xy, weight=2.0, params={"std": 0.1})
    torso_velocity = RewTerm(func=mdp.torso_lin_vel_xy, weight=5.0, params={"std": 0.05})
    yaw_rate = RewTerm(func=mdp.torso_yaw_rate, weight=2.0, params={"std": 0.2})
    contact = RewTerm(func=mdp.contact_match, weight=2.0)
    hip_flex = RewTerm(func=mdp.hip_flex_excess, weight=-5000.0)
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-5.0)
    # Holding an arm on its stop takes up to the 14 N m limit (shuffle walk); holding it against gravity about 1 to 4 N m.
    arm_torque = RewTerm(func=base.joint_torques_l2, weight=-0.01, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)})
    action_rate = RewTerm(func=base.action_rate_l2, weight=-1.0)
    joint_limit = RewTerm(func=base.joint_pos_limits, weight=-10.0)
    joint_acceleration = RewTerm(func=base.joint_acc_l2, weight=-2.0e-7)
    undesired_contacts = RewTerm(func=base.undesired_contacts, weight=-10.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=r"^(?!left_foot_link$)(?!right_foot_link$).+$"), "threshold": 1.0})
    termination = RewTerm(func=base.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base.time_out, time_out=True)
    fallen = DoneTerm(func=mdp.fallen)


@configclass
class IdleShuffleEnvCfg(TrackingEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.scene.robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state = self.scene.robot.init_state.replace(joint_pos=idle_default_joint_pos(
            self.scene.robot.spawn.asset_path, margins={n: ARM_MARGIN for n in ARM_JOINTS}))
        # The Idle pose and the gait reach the actual joint limits (e.g. the right ankle roll); penalize only beyond them.
        self.scene.robot.soft_joint_pos_limit_factor = 1.0
        self.scene.contact_forces.debug_vis = False
        self.actions.joint_pos.scale = K1_ACTION_SCALE
        use_v1_head(self)
        self.events.push_robot.params["velocity_range"] = PUSH
        self.viewer.eye = (1.7, -1.7, 1.15)
        self.viewer.lookat = (0.0, 0.0, 0.45)


@configclass
class IdleShufflePlayEnvCfg(IdleShuffleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 30.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.velocity.moving_start_fraction = 0.0


def use_exterior(cfg):
    """The torso exterior and arm foam masses (EXTERIOR_MASS_FILE, which also keeps the v1 head) and the Trunk mass variation."""
    cfg.scene.robot.spawn.mass_file = EXTERIOR_MASS_FILE
    cfg.events.trunk_mass = EventTerm(func=base.randomize_rigid_body_mass, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"), "mass_distribution_params": EXTERIOR_MASS_ADD_KG,
        "operation": "add", "recompute_inertia": True})


@configclass
class IdleShuffleExteriorEnvCfg(IdleShuffleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_exterior(self)


@configclass
class IdleShuffleExteriorPlayEnvCfg(IdleShufflePlayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_exterior(self)
