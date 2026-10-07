from isaaclab.utils import configclass
from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 6000           # the stock-head run was stopped at 6000 and model_5000 was adopted
    experiment_name = "k1_pikachu_idle_scale0842_head_r27"
