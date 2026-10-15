"""Scale0842 idle with the v1 head on its current mount, randomized around it, plus the head orientation reward.

The neck gains, target clip and the head orientation term are those of pikachu_idle_scale0842_head_v2sim_headori
(C in project-memos/v2-head-sim-20261008); the head inertial is the v1 (r27) one of pikachu_idle_scale0842_head_r27,
and the reference is the original idle, which holds the neck at 20 deg up. The firmware clips the target at 17 deg up,
so the neck is expected to settle about 3 deg below that and the head about 6 deg below the animation. The Head_2 mass
(2.5 to 5.0 kg) and COM are randomized around v1 because the printed head has not been weighed.
"""
import gymnasium as gym

gym.register(
    id="Booster-K1-Pikachu_Idle-Scale0842-HeadV1Sim-HeadOri-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RoughWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Pikachu_Idle-Scale0842-HeadV1Sim-HeadOri-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PlayFlatWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)
