from isaaclab.utils import configclass
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842 import env_cfg as base
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v2fix import env_cfg as v2fix
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v2sim import env_cfg as v2sim


@configclass
class RoughWoStateEstimationEnvCfg(base.RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        v2sim.use_v2sim_head(self)
        v2fix.add_head_orientation(self)


@configclass
class PlayFlatWoStateEstimationEnvCfg(base.PlayFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        v2sim.use_v2sim_head(self)
        v2fix.add_head_orientation(self)
