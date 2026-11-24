"""T215: the cute street-Pikachu motion (k4, from the Kling 4.0 reference video) on the v1 head.

The head and neck setup, the randomization and the head orientation reward are those of
pikachu_idle_scale0842_head_v1sim_headori (the same setup as the deployed Idle policy); only the reference motion changes.
Joint ranges follow booster_train's soft limits (URDF x 0.9, decided by the user on 2026-11-25), and the elbow pitch targets
are clipped to (-1.80, 1.9024): on the real robot the right elbow stopped at -1.86 and the firmware limit is 1.9024
(project-memos/k1-idle-shuffle-test-20261017).
"""
import gymnasium as gym

gym.register(
    id="Booster-K1-Pikachu_T215_K4-HeadV1Sim-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RoughWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Pikachu_T215_K4-HeadV1Sim-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PlayFlatWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)
