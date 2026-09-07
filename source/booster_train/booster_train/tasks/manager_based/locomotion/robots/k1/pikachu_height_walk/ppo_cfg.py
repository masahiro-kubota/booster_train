"""PPO configuration for fixed-height Pikachu locomotion."""

from isaaclab.utils import configclass

from ..pikachu_idle_walk.ppo_cfg import PPORunnerCfg as IdleWalkPPORunnerCfg


@configclass
class PPORunnerCfg(IdleWalkPPORunnerCfg):
    experiment_name = "k1_pikachu_height_walk_scale0842"
