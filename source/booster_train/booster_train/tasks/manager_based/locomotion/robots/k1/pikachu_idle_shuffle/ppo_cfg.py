from isaaclab.utils import configclass
from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "k1_pikachu_idle_shuffle"
    max_iterations = 8000
    save_interval = 500


@configclass
class PPORunnerExteriorCfg(PPORunnerCfg):
    experiment_name = "k1_pikachu_idle_shuffle_exterior"
