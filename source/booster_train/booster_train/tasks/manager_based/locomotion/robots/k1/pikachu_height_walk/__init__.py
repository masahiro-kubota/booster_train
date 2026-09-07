"""Low-height Pikachu locomotion without time-varying Idle imitation."""

import gymnasium as gym

for suffix, config in (("", "HeightWalkEnvCfg"), ("-Play", "HeightWalkPlayEnvCfg")):
    gym.register(
        id=f"Booster-K1-Pikachu-HeightWalk-Scale0842-v0{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
        },
    )
