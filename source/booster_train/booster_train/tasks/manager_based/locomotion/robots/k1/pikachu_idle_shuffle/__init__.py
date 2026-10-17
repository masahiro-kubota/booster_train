"""Pikachu idle and shuffle walk in one policy: the Idle motion while the velocity command is zero, the shuffle-walk gait
library while it is not (pikachu_shuffle_walk), blended at each change, so standing never steps in place and no policy switch
is needed on the robot."""
import gymnasium as gym

# Exterior: the same task with the torso exterior and arm foam masses (env_cfg.use_exterior).
for name, config, runner in (("", "IdleShuffle", "PPORunnerCfg"), ("-Exterior", "IdleShuffleExterior", "PPORunnerExteriorCfg")):
    for suffix, kind in (("", "EnvCfg"), ("-Play", "PlayEnvCfg")):
        gym.register(
            id=f"Booster-K1-Pikachu-IdleShuffle{name}-v0{suffix}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{config}{kind}",
                    "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:{runner}"},
        )
