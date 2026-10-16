"""Pikachu idle and shuffle walk in one policy: the Idle motion while the velocity command is zero, the shuffle-walk gait
library while it is not (pikachu_shuffle_walk), blended at each change, so standing never steps in place and no policy switch
is needed on the robot."""
import gymnasium as gym

for suffix, config in (("", "IdleShuffleEnvCfg"), ("-Play", "IdleShufflePlayEnvCfg")):
    gym.register(
        id=f"Booster-K1-Pikachu-IdleShuffle-v0{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{config}",
                "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg"},
    )
