"""Pikachu shuffle walk: one policy conditioned on the velocity command and the gait phase, imitating the gait library.

As Disney's BDX and Olaf walking policies: the reference (joint positions, torso pose and velocity, foot contacts) is looked
up from densely sampled gait cycles and used only in the rewards; the actor sees the command, the phase, the torso yaw relative
to the reference and proprioception. The head is the v1 head on its current mount with the HeadV1Sim settings (neck kp 30,
kd 3, firmware target clip, Head_2 mass 2.5 to 5.0 kg and COM randomized around v1).
"""
from dataclasses import fields
from pathlib import Path

import isaaclab.envs.mdp as base
from isaaclab.managers import EventTermCfg as EventTerm, ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Noise
from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.robots.booster import BOOSTER_K1_CFG, K1_ACTION_SCALE
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842.tracking_env_cfg import TrackingEnvCfg
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_r27.link_mass import (
    LinkMassUrdfFileCfg, spawn_with_link_masses)
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v1sim_headori import env_cfg as v1sim
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v2sim import env_cfg as v2sim

from . import mdp
from .commands import GaitVelocityCommandCfg, ShuffleGaitCommandCfg

HERE = Path(__file__).resolve().parent
LIBRARY = str(HERE / "gait_library_S3-12.npz")
IDLE_POSE = f"{BOOSTER_ASSETS_DIR}/robots/K1/pikachu_scale0842/reset_pose.npz"
LEG_JOINTS = [f"{s}_{j}" for s in ("Left", "Right") for j in ("Hip_Pitch", "Hip_Roll", "Hip_Yaw", "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll")]
UPPER_JOINTS = ["AAHead_yaw", "Head_pitch", "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
                "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw"]
PUSH = {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.2, 0.2), "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.4, 0.4)}


@configclass
class CommandsCfg:
    # The velocity term must come first: the gait term reads it at reset and every step.
    velocity = GaitVelocityCommandCfg(forward_range=(-0.06, 0.06), lateral_range=(-0.04, 0.04), yaw_range=(-0.2, 0.2),
                                      standing_fraction=0.15, initial_stand_s=1.0, ramp_s=1.0, resampling_time_range=(3.0, 6.0))
    gait = ShuffleGaitCommandCfg(library_file=LIBRARY, idle_pose_file=IDLE_POSE)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        velocity_command = ObsTerm(func=base.generated_commands, params={"command_name": "velocity"})
        gait_phase = ObsTerm(func=base.generated_commands, params={"command_name": "gait"})
        heading = ObsTerm(func=mdp.path_heading)
        base_ang_vel = ObsTerm(func=base.base_ang_vel, noise=Noise(n_min=-0.2, n_max=0.2))
        gravity = ObsTerm(func=base.projected_gravity, noise=Noise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=base.joint_pos_rel, noise=Noise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=base.joint_vel_rel, noise=Noise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=base.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

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
    torso_orientation = RewTerm(func=mdp.torso_orientation, weight=10.0, params={"std": 0.2})
    torso_height = RewTerm(func=mdp.torso_height, weight=5.0, params={"std": 0.03})
    torso_position = RewTerm(func=mdp.torso_position_xy, weight=2.0, params={"std": 0.1})
    torso_velocity = RewTerm(func=mdp.torso_lin_vel_xy, weight=5.0, params={"std": 0.05})
    yaw_rate = RewTerm(func=mdp.torso_yaw_rate, weight=2.0, params={"std": 0.2})
    contact = RewTerm(func=mdp.contact_match, weight=2.0)
    hip_flex = RewTerm(func=mdp.hip_flex_excess, weight=-5000.0)
    feet_slide = RewTerm(func=mdp.feet_slide, weight=-5.0)
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


def use_v1_head(cfg):
    """HeadV1Sim head: v1 (r27) inertial, neck gains, action scale and target clip, Head_2 mass and COM randomization."""
    source = cfg.scene.robot.spawn
    values = {f.name: getattr(source, f.name) for f in fields(source) if f.init}
    values["func"] = spawn_with_link_masses
    cfg.scene.robot.spawn = LinkMassUrdfFileCfg(**values, mass_file=v1sim.MASS_FILE)
    head = cfg.scene.robot.actuators["head"]
    head.stiffness, head.damping = v2sim.NECK_KP, v2sim.NECK_KD
    scale = dict(K1_ACTION_SCALE); scale[".*Head.*"] = 0.25 * head.effort_limit_sim / v2sim.NECK_KP
    cfg.actions.joint_pos.scale = scale
    cfg.actions.joint_pos.clip = dict(v2sim.NECK_TARGET_CLIP)
    cfg.events.head_mass = EventTerm(func=base.randomize_rigid_body_mass, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names="Head_2"), "mass_distribution_params": v1sim.HEAD_MASS_KG,
        "operation": "abs", "recompute_inertia": True})
    cfg.events.head_com = EventTerm(func=base.randomize_rigid_body_com, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names="Head_2"), "com_range": dict(v1sim.HEAD_COM_RANGE_M)})


@configclass
class ShuffleWalkEnvCfg(TrackingEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.scene.robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # The Idle pose and the gait reach the actual joint limits (e.g. the right ankle roll); penalize only beyond them.
        self.scene.robot.soft_joint_pos_limit_factor = 1.0
        self.scene.contact_forces.debug_vis = False
        self.actions.joint_pos.scale = K1_ACTION_SCALE
        use_v1_head(self)
        self.events.push_robot.params["velocity_range"] = PUSH
        self.viewer.eye = (1.7, -1.7, 1.15)
        self.viewer.lookat = (0.0, 0.0, 0.45)


@configclass
class ShuffleWalkPlayEnvCfg(ShuffleWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 30.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.velocity.moving_start_fraction = 0.0
