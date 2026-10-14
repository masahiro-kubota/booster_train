from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import booster_train.tasks.manager_based.beyond_mimic.mdp as mdp
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842 import env_cfg as base
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v2sim import env_cfg as v2sim

HEAD_MASS_KG = 2.0                     # v2 target; inertia scaled from the nominal file, COM at the nominal
HEAD_ORI_WEIGHT, HEAD_ORI_STD = 10.0, 0.2
EMPHASIS_TERMS = ("motion_foot_ori", "motion_foot_pos", "motion_hand_ori", "motion_hand_pos",
                  "motion_trunk_ori", "motion_trunk_pos", "motion_trunk_ang_vel")


def use_fixed_v2_head(cfg):
    """The v2 sim head, neck gains and target clip, with Head_2 fixed at HEAD_MASS_KG and its COM not moved."""
    v2sim.use_v2sim_head(cfg)
    cfg.events.head_mass.params["mass_distribution_params"] = (HEAD_MASS_KG, HEAD_MASS_KG)
    cfg.events.head_com = None


def add_head_orientation(cfg):
    cfg.rewards.motion_head_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=HEAD_ORI_WEIGHT,
        params={"command_name": "motion", "std": HEAD_ORI_STD, "body_names": ["Head_2"]},
    )


def use_generic_rewards(cfg):
    """mj_dance_002's rewards: the generic BeyondMimic terms, light action rate and contact penalties."""
    for name in EMPHASIS_TERMS:
        setattr(cfg.rewards, name, None)
    cfg.rewards.action_rate_l2.weight = -0.1
    cfg.rewards.undesired_contacts.weight = -0.1
    cfg.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces",
        body_names=[r"^(?!left_hand_link$)(?!right_hand_link$)(?!left_foot_link$)(?!right_foot_link$).+$"],
    )


@configclass
class HeadOriRoughWoStateEstimationEnvCfg(base.RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_fixed_v2_head(self)
        add_head_orientation(self)


@configclass
class HeadOriPlayFlatWoStateEstimationEnvCfg(base.PlayFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_fixed_v2_head(self)
        add_head_orientation(self)


@configclass
class GenericRoughWoStateEstimationEnvCfg(base.RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_fixed_v2_head(self)
        use_generic_rewards(self)


@configclass
class GenericPlayFlatWoStateEstimationEnvCfg(base.PlayFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_fixed_v2_head(self)
        use_generic_rewards(self)
