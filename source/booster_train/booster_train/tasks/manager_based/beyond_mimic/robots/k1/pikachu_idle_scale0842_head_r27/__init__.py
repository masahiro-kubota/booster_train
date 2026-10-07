"""Scale0842 idle motion imitation with the r27 Pikachu head mass on Head_2 (is the head too heavy?).

Same motion, rewards, terminations and events as pikachu_idle_scale0842; only Head_2's inertial is replaced by
booster_assets/robots/K1/pikachu_scale0842/head_r27_mass_properties.json (stock 0.70 kg + the r27 head, solid).
"""
import gymnasium as gym

gym.register(
    id="Booster-K1-Pikachu_Idle-Scale0842-HeadR27-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RoughWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Pikachu_Idle-Scale0842-HeadR27-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PlayFlatWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)
