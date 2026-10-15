from isaaclab.utils import configclass
from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "k1_pikachu_shuffle_walk"
    max_iterations = 6000
    save_interval = 500
