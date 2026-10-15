from isaaclab.utils import configclass
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842 import env_cfg as base
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_r27 import env_cfg as r27
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v2fix import env_cfg as v2fix
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v2sim import env_cfg as v2sim

MASS_FILE = r27.MASS_FILE
HEAD_MASS_KG = (2.5, 5.0)                                                   # whole Head_2, absolute; CAD 4.62
HEAD_COM_RANGE_M = {"x": (-0.02, 0.02), "y": (-0.01, 0.01), "z": (-0.02, 0.02)}  # Head_2 frame, around the v1 COM


def use_v1sim_head(cfg):
    """The v2sim neck gains, target clip and randomization, on the v1 head inertial and the original reference."""
    motion_file = cfg.commands.motion.motion_file
    v2sim.use_v2sim_head(cfg)
    cfg.scene.robot.spawn.mass_file = MASS_FILE
    cfg.commands.motion.motion_file = motion_file
    cfg.events.head_mass.params["mass_distribution_params"] = HEAD_MASS_KG
    cfg.events.head_com.params["com_range"] = dict(HEAD_COM_RANGE_M)


@configclass
class RoughWoStateEstimationEnvCfg(base.RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_v1sim_head(self)
        v2fix.add_head_orientation(self)


@configclass
class PlayFlatWoStateEstimationEnvCfg(base.PlayFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_v1sim_head(self)
        v2fix.add_head_orientation(self)
