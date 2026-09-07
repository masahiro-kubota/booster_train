from isaaclab.utils import configclass
from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "k1_pikachu_idle_walk_scale0842"
    max_iterations = 6000
    save_interval = 100

    def __post_init__(self):
        super().__post_init__()
        # The shell leaves millimetres of space; the old 1 rad-scale initial
        # exploration would overwhelmingly terminate before learning support.
        self.policy.init_noise_std = 0.15
