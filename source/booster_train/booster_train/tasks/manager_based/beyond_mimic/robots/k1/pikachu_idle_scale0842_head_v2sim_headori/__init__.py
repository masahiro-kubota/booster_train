"""Scale0842 idle with the v2 sim head randomized as in pikachu_idle_scale0842_head_v2sim, plus a head orientation reward.

The head, neck gains, target clip and the Head_2 mass (1.5 to 5.0 kg) and COM randomization are those of
pikachu_idle_scale0842_head_v2sim; the rewards add the Head_2 orientation term of pikachu_idle_scale0842_head_v2fix
HeadOri (weight 10, std 0.2), which made the neck follow the reference with the head fixed at 2.0 kg.
project-memos/v2-head-sim-20261008 checks whether it still does over the mass and COM candidates.
"""
import gymnasium as gym

gym.register(
    id="Booster-K1-Pikachu_Idle-Scale0842-HeadV2Sim-HeadOri-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RoughWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Pikachu_Idle-Scale0842-HeadV2Sim-HeadOri-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PlayFlatWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)
