from isaaclab.utils import configclass
from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class HeadOriPPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 2000
    experiment_name = "k1_pikachu_idle_scale0842_head_v2fix_headori"


@configclass
class GenericPPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 2000
    experiment_name = "k1_pikachu_idle_scale0842_head_v2fix_generic"
