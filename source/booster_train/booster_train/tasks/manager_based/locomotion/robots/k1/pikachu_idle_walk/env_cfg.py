"""Scale0842 Idle-style locomotion, stage 1: flat ground and geometry clearance."""
from dataclasses import fields
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import isaaclab.envs.mdp as base
from isaaclab.managers import EventTermCfg as EventTerm, ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Noise
from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.robots.booster import BOOSTER_K1_CFG, K1_ACTION_SCALE
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842.tracking_env_cfg import TrackingEnvCfg

from . import mdp
from .asset import TorsoUrdfFileCfg, spawn_with_torso
from .commands import IdleLoopCommandCfg, SmoothVelocityCommandCfg
from .actions import GuardedJointPositionAction

DIRECTORY = f"{BOOSTER_ASSETS_DIR}/robots/K1/pikachu_scale0842"
MOTION = f"{BOOSTER_ASSETS_DIR}/motions/K1/k1_pikachu_idle_scale0842_pingpong_tile3_blender50_grounded_hold0p5.npz"


@configclass
class CommandsCfg:
    idle = IdleLoopCommandCfg(motion_file=MOTION)
    velocity = SmoothVelocityCommandCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        velocity_command = ObsTerm(func=base.generated_commands, params={"command_name": "velocity"})
        idle_command = ObsTerm(func=base.generated_commands, params={"command_name": "idle"})
        base_ang_vel = ObsTerm(func=base.base_ang_vel, noise=Noise(n_min=-0.2,n_max=0.2))
        gravity = ObsTerm(func=base.projected_gravity, noise=Noise(n_min=-0.05,n_max=0.05))
        joint_pos = ObsTerm(func=base.joint_pos_rel, noise=Noise(n_min=-0.01,n_max=0.01))
        joint_vel = ObsTerm(func=base.joint_vel_rel, noise=Noise(n_min=-0.5,n_max=0.5))
        actions = ObsTerm(func=base.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=base.base_lin_vel)
        height_error = ObsTerm(func=mdp.height_observation)
        clearance = ObsTerm(func=mdp.clearance_observation)

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False
            self.history_length = 1

    policy = PolicyCfg()
    critic = CriticCfg()


@configclass
class RewardsCfg:
    velocity_xy = RewTerm(func=mdp.velocity_tracking,weight=30.,params={"std":0.05})
    velocity_yaw = RewTerm(func=mdp.velocity_tracking,weight=10.,params={"std":0.15,"yaw":True})
    idle_height = RewTerm(func=mdp.height_tracking,weight=20.)
    idle_tilt = RewTerm(func=mdp.tilt_tracking,weight=30.)
    upper_joint = RewTerm(func=mdp.upper_joint_tracking,weight=5.)
    upper_position = RewTerm(func=mdp.upper_body_tracking,weight=15.)
    upper_orientation = RewTerm(func=mdp.upper_body_tracking,weight=10.,params={"std":0.15,"orientation":True})
    crouch = RewTerm(func=mdp.knee_range,weight=-5.)
    shell_margin = RewTerm(func=mdp.shell_proximity,weight=-100.)
    action_rate = RewTerm(func=base.action_rate_l2,weight=-2.)
    joint_limit = RewTerm(func=base.joint_pos_limits,weight=-10.)
    joint_acceleration = RewTerm(func=base.joint_acc_l2,weight=-2.e-7)
    feet_slide = RewTerm(func=mdp.feet_slide,weight=-1.)
    flight = RewTerm(func=mdp.flight,weight=-5.)
    air_time = RewTerm(func=mdp.biped_air_time,weight=2.)
    undesired_contacts = RewTerm(func=base.undesired_contacts,weight=-10.,params={
        "sensor_cfg": SceneEntityCfg("contact_forces",body_names=r"^(?!left_foot_link$)(?!right_foot_link$).+$"),"threshold":1.})


@configclass
class EventsCfg:
    reset_pose = EventTerm(func=mdp.reset_idle,mode="reset")
    physics_material = EventTerm(func=base.randomize_rigid_body_material,mode="startup",params={
        "asset_cfg":SceneEntityCfg("robot",body_names=".*"),"static_friction_range":(0.3,0.6),
        "dynamic_friction_range":(0.3,0.6),"restitution_range":(0.,0.5),"num_buckets":64})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base.time_out,time_out=True)
    shell_clearance = DoneTerm(func=mdp.shell_unsafe)
    posture = DoneTerm(func=mdp.posture_failure)


@configclass
class IdleWalkEnvCfg(TrackingEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    shell_directory: str = DIRECTORY

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.
        self.scene.num_envs = 1024
        self.scene.env_spacing = 2.5
        self.scene.contact_forces.debug_vis = False
        self.scene.robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        source = self.scene.robot.spawn
        values = {f.name:getattr(source,f.name) for f in fields(source) if f.init}
        values["func"] = spawn_with_torso
        self.scene.robot.spawn = TorsoUrdfFileCfg(**values, shell_file=f"{DIRECTORY}/torso_shell_trunk.npz")
        with np.load(Path(DIRECTORY)/"reset_pose.npz",allow_pickle=False) as d:
            self.scene.robot.init_state.pos = tuple(float(x) for x in d["root_pos"])
            self.scene.robot.init_state.rot = tuple(float(x) for x in d["root_quat"])
            self.scene.robot.init_state.joint_pos = dict(zip(d["joint_names"].tolist(),d["joint_pos"].tolist()))
        # URDF degrees -> PhysX radians can round an exact limit outward.
        # A 10 microradian inset prevents rejecting the approved limit poses.
        for joint in ET.parse(source.asset_path).getroot().findall("joint"):
            name, limit = joint.get("name"), joint.find("limit")
            if limit is not None and name in self.scene.robot.init_state.joint_pos:
                self.scene.robot.init_state.joint_pos[name] = float(np.clip(
                    self.scene.robot.init_state.joint_pos[name],float(limit.get("lower"))+1.e-5,float(limit.get("upper"))-1.e-5))
        # Approved Idle touches the original soft envelope: use the actual joint
        # limits, not a narrower envelope that penalizes the requested pose.
        self.scene.robot.soft_joint_pos_limit_factor = 1.0
        self.actions.joint_pos.scale = K1_ACTION_SCALE
        self.actions.joint_pos.class_type = GuardedJointPositionAction
        self.viewer.eye = (1.7,-1.7,1.15)
        self.viewer.lookat = (0.,0.,0.45)


@configclass
class IdleWalkPlayEnvCfg(IdleWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 65.
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.commands.velocity.forward_range = (0.05,0.05)
        self.commands.velocity.lateral_range = (0.,0.)
        self.commands.velocity.yaw_range = (0.,0.)
        self.commands.velocity.standing_fraction = 0.
