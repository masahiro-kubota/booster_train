"""Fixed-height locomotion with hard-shell clearance and no Idle imitation."""

import isaaclab.envs.mdp as base
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Noise

from ..pikachu_idle_walk import mdp as idle_walk_mdp
from ..pikachu_idle_walk.commands import SmoothVelocityCommandCfg
from ..pikachu_idle_walk.env_cfg import IdleWalkEnvCfg
from . import mdp


@configclass
class CommandsCfg:
    """Start with forward-only commands before adding lateral motion and yaw."""

    velocity = SmoothVelocityCommandCfg(
        forward_range=(0.0, 0.10),
        lateral_range=(0.0, 0.0),
        yaw_range=(0.0, 0.0),
        standing_fraction=0.2,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        velocity_command = ObsTerm(func=base.generated_commands, params={"command_name": "velocity"})
        base_ang_vel = ObsTerm(func=base.base_ang_vel, noise=Noise(n_min=-0.2, n_max=0.2))
        gravity = ObsTerm(func=base.projected_gravity, noise=Noise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=base.joint_pos_rel, noise=Noise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=base.joint_vel_rel, noise=Noise(n_min=-0.5, n_max=0.5))
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
        clearance = ObsTerm(func=idle_walk_mdp.clearance_observation)

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False
            self.history_length = 1

    policy = PolicyCfg()
    critic = CriticCfg()


@configclass
class RewardsCfg:
    """Locomotion rewards plus only height and shell-clearance posture goals."""

    velocity_xy = RewTerm(func=idle_walk_mdp.velocity_tracking, weight=30.0, params={"std": 0.05})
    velocity_yaw = RewTerm(func=idle_walk_mdp.velocity_tracking, weight=10.0, params={"std": 0.15, "yaw": True})
    trunk_height = RewTerm(func=mdp.height_tracking, weight=20.0)
    shell_margin = RewTerm(func=idle_walk_mdp.shell_proximity, weight=-100.0)
    action_rate = RewTerm(func=base.action_rate_l2, weight=-2.0)
    joint_limit = RewTerm(func=base.joint_pos_limits, weight=-10.0)
    joint_acceleration = RewTerm(func=base.joint_acc_l2, weight=-2.0e-7)
    feet_slide = RewTerm(func=idle_walk_mdp.feet_slide, weight=-1.0)
    flight = RewTerm(func=idle_walk_mdp.flight, weight=-5.0)
    air_time = RewTerm(func=idle_walk_mdp.biped_air_time, weight=2.0)
    undesired_contacts = RewTerm(
        func=base.undesired_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=r"^(?!left_foot_link$)(?!right_foot_link$).+$"),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base.time_out, time_out=True)
    shell_clearance = DoneTerm(func=idle_walk_mdp.shell_unsafe)
    low_height = DoneTerm(func=mdp.low_height)


@configclass
class HeightWalkEnvCfg(IdleWalkEnvCfg):
    """Training task that leaves joint posture and Trunk tilt to the policy."""

    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # RTX 4090 / 32 GiB host benchmarked with video recording enabled.
        # 6,144 environments exhausted host-memory headroom and failed during
        # PhysX startup, while 4,096 completed with about 9 GiB available.
        self.scene.num_envs = 4096


@configclass
class HeightWalkPlayEnvCfg(HeightWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 65.0
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.commands.velocity.forward_range = (0.05, 0.05)
        self.commands.velocity.standing_fraction = 0.0
