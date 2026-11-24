from isaaclab.utils import configclass
from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    max_iterations = 6000           # the v1sim_headori idle needed about 6000 for the arms to settle
    experiment_name = "k1_pikachu_t215_k4_head_v1sim"
