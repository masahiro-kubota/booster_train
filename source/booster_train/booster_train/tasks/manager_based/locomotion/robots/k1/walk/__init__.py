import gymnasium as gym


gym.register(
    id="Booster-K1-Locomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Locomotion-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PlayFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Locomotion-IdealFlatForward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:IdealFlatForwardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:IdealFlatForwardPPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Locomotion-IdealFlatForward-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:IdealFlatForwardPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:IdealFlatForwardPPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Locomotion-IdealFlatCommandRandom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:IdealFlatCommandRandomEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:IdealFlatCommandRandomPPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Locomotion-IdealFlatCommandRandom-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:IdealFlatCommandRandomPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:IdealFlatCommandRandomPPORunnerCfg",
    },
)
