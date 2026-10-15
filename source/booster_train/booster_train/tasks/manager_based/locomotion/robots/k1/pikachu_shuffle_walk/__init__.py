"""Pikachu shuffle walk: one walking policy conditioned on the velocity command and the gait phase.

It imitates the shuffle-walk gait library (project-memos/walk-leg-clearance-20261016): 0.5 s cycles per (vx, vy, wz) grid point,
generated with the Idle stance, the hard torso exterior (S3-12) clearance and a ZMP-feasible lateral sway. As Disney's BDX and
Olaf, the reference is used only in the rewards; at run time the policy needs the velocity command and the phase.
"""
import gymnasium as gym

for suffix, config in (("", "ShuffleWalkEnvCfg"), ("-Play", "ShuffleWalkPlayEnvCfg")):
    gym.register(
        id=f"Booster-K1-Pikachu-ShuffleWalk-v0{suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{config}",
                "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg"},
    )
