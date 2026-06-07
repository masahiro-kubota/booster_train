# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from booster_train.assets.robots.booster import BOOSTER_K1_CFG, K1_ACTION_SCALE
import booster_train.tasks.manager_based.locomotion.velocity.mdp as mdp


K1_FLAT_OBS_DOF_VEL_SCALE = 0.1
K1_FLAT_OBSERVATION_HISTORY_LENGTH = 10
K1_FLAT_SINGLE_FRAME_OBSERVATION_DIM = 69
K1_FLAT_EXPORTED_OBSERVATION_DIM = 690

K1_FLAT_POLICY_JOINT_NAMES = [
    "ALeft_Shoulder_Pitch",
    "ARight_Shoulder_Pitch",
    "Left_Hip_Pitch",
    "Right_Hip_Pitch",
    "Left_Shoulder_Roll",
    "Right_Shoulder_Roll",
    "Left_Hip_Roll",
    "Right_Hip_Roll",
    "Left_Elbow_Pitch",
    "Right_Elbow_Pitch",
    "Left_Hip_Yaw",
    "Right_Hip_Yaw",
    "Left_Elbow_Yaw",
    "Right_Elbow_Yaw",
    "Left_Knee_Pitch",
    "Right_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Right_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Ankle_Roll",
]


def _make_k1_velocity_action_scale(action_joint_names: list[str]) -> dict[str, float]:
    action_scale: dict[str, float] = {}
    matched_joint_names: set[str] = set()
    for joint_name_expr, scale in K1_ACTION_SCALE.items():
        matched_names = [joint_name for joint_name in action_joint_names if re.fullmatch(joint_name_expr, joint_name)]
        if not matched_names:
            continue
        action_scale[joint_name_expr] = scale
        for joint_name in matched_names:
            if joint_name in matched_joint_names:
                raise ValueError(f"Multiple K1 action scale patterns match joint '{joint_name}'.")
            matched_joint_names.add(joint_name)

    missing_joint_names = sorted(set(action_joint_names) - matched_joint_names)
    if missing_joint_names:
        raise ValueError(f"Missing K1 action scale for joints: {missing_joint_names}")
    return action_scale


K1_FLAT_ACTION_SCALE = _make_k1_velocity_action_scale(K1_FLAT_POLICY_JOINT_NAMES)

K1_REAL_JOINT_NAMES = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

K1_DEFAULT_JOINT_POS = [
    0.0,
    0.0,
    0.2,
    -1.25,
    0.0,
    -0.5,
    0.2,
    1.25,
    0.0,
    0.5,
    -0.15,
    0.0,
    0.0,
    0.3,
    -0.15,
    0.0,
    -0.15,
    0.0,
    0.0,
    0.3,
    -0.15,
    0.0,
]

K1_DEFAULT_JOINT_POS_BY_NAME = dict(zip(K1_REAL_JOINT_NAMES, K1_DEFAULT_JOINT_POS))

K1_FOOT_BODY_NAMES = ["left_foot_link", "right_foot_link"]
K1_LEG_JOINT_NAMES = [
    ".*_Hip_.*",
    ".*_Knee_Pitch",
]
K1_ANKLE_JOINT_NAMES = [
    ".*_Ankle_.*",
]
K1_HIP_DEVIATION_JOINT_NAMES = [
    ".*_Hip_Yaw",
    ".*_Hip_Roll",
]
K1_ARM_JOINT_NAMES = [
    "ALeft_Shoulder_Pitch",
    "ARight_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Right_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Right_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "Right_Elbow_Yaw",
]
K1_ARM_ACTION_INDICES = [K1_FLAT_POLICY_JOINT_NAMES.index(name) for name in K1_ARM_JOINT_NAMES]


@configclass
class K1FlatSceneCfg(InteractiveSceneCfg):
    """Configuration for the K1 flat velocity scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAAC_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=K1_FLAT_POLICY_JOINT_NAMES,
        scale=K1_FLAT_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        deploy_locomotion_obs = ObsTerm(
            func=mdp.k1_deploy_locomotion_observation,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=K1_FLAT_POLICY_JOINT_NAMES,
                    preserve_order=True,
                ),
                "obs_dof_vel_scale": K1_FLAT_OBS_DOF_VEL_SCALE,
            },
            clip=(-100.0, 100.0),
            history_length=K1_FLAT_OBSERVATION_HISTORY_LENGTH,
            flatten_history_dim=True,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        privileged_locomotion_obs = ObsTerm(
            func=mdp.k1_privileged_locomotion_observation,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=K1_FLAT_POLICY_JOINT_NAMES,
                    preserve_order=True,
                ),
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=K1_FOOT_BODY_NAMES,
                    preserve_order=True,
                ),
                "obs_dof_vel_scale": K1_FLAT_OBS_DOF_VEL_SCALE,
                "contact_threshold": 0.5,
            },
            clip=(-100.0, 100.0),
            history_length=K1_FLAT_OBSERVATION_HISTORY_LENGTH,
            flatten_history_dim=True,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for K1 deploy-compatible flat velocity locomotion."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(
        func=mdp.k1_joint_energy,
        weight=-1.0e-3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=K1_FLAT_POLICY_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    joint_torques_l2 = RewTerm(
        func=mdp.k1_joint_torques_l2,
        weight=-2.0e-6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=K1_FLAT_POLICY_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=K1_FLAT_POLICY_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    arm_joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=K1_ARM_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    arm_action_l2 = RewTerm(
        func=mdp.action_l2_subset,
        weight=-0.03,
        params={"action_indices": K1_ARM_ACTION_INDICES},
    )
    arm_action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_subset,
        weight=-0.02,
        params={"action_indices": K1_ARM_ACTION_INDICES},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=K1_FOOT_BODY_NAMES,
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=K1_FOOT_BODY_NAMES,
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=K1_FOOT_BODY_NAMES,
                preserve_order=True,
            ),
        },
    )
    feet_force = RewTerm(
        func=mdp.k1_body_force,
        weight=-3.0e-3,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=K1_FOOT_BODY_NAMES,
                preserve_order=True,
            ),
            "threshold": 500.0,
            "max_reward": 400.0,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.k1_feet_too_near_humanoid,
        weight=-2.0,
        params={
            "threshold": 0.2,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=K1_FOOT_BODY_NAMES,
                preserve_order=True,
            ),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.k1_feet_stumble,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=K1_FOOT_BODY_NAMES,
                preserve_order=True,
            )
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=K1_FLAT_POLICY_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!left_foot_link$)(?!right_foot_link$).+$"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class RewardARewardsCfg:
    """Isaac Lab G1/H1-flat-like baseline rewards for K1 reward ablation."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINT_NAMES)},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_LEG_JOINT_NAMES)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=K1_FOOT_BODY_NAMES,
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=K1_FOOT_BODY_NAMES,
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=K1_FOOT_BODY_NAMES,
                preserve_order=True,
            ),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_ANKLE_JOINT_NAMES)},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=K1_HIP_DEVIATION_JOINT_NAMES)},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=K1_ARM_JOINT_NAMES,
                preserve_order=True,
            )
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class K1FlatEnvCfg(ManagerBasedRLEnvCfg):
    """Flat-terrain K1 velocity-tracking environment."""

    scene: K1FlatSceneCfg = K1FlatSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    action_scale = K1_FLAT_ACTION_SCALE
    action_joint_names = K1_FLAT_POLICY_JOINT_NAMES
    observation_history_length = K1_FLAT_OBSERVATION_HISTORY_LENGTH
    single_frame_observation_dim = K1_FLAT_SINGLE_FRAME_OBSERVATION_DIM
    exported_observation_dim = K1_FLAT_EXPORTED_OBSERVATION_DIM

    def __post_init__(self) -> None:
        """Post initialization."""
        self.scene.robot = BOOSTER_K1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.57),
                joint_pos=K1_DEFAULT_JOINT_POS_BY_NAME,
                joint_vel={".*": 0.0},
            ),
        )

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.origin_type = "world"
        self.viewer.eye = (3.0, -4.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class K1FlatEnvCfg_PLAY(K1FlatEnvCfg):
    """Reduced K1 flat velocity environment for play/export."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.reset_base = None
        self.events.reset_robot_joints = None
        self.events.push_robot = None


@configclass
class K1FlatRewardAEnvCfg(K1FlatEnvCfg):
    """K1 flat velocity task with Isaac Lab G1/H1-flat-like baseline rewards."""

    rewards: RewardARewardsCfg = RewardARewardsCfg()


class K1FlatRewardAEnvCfg_PLAY(K1FlatRewardAEnvCfg):
    """Reduced Reward-A K1 flat velocity environment for play/export."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.reset_base = None
        self.events.reset_robot_joints = None
        self.events.push_robot = None
