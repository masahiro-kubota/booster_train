"""Scale0842 idle motion imitation for designing the v2 Pikachu head (project-memos/v2-head-sim-20261008).

Changes from pikachu_idle_scale0842:
- the head is mounted 0.349 rad further up about the neck pitch axis and centred: Head_2 takes
  booster_assets/robots/K1/pikachu_scale0842/head_v2sim_mass_properties.json and the reference is the idle with
  Head_pitch raised by the same angle (..._grounded_hold0p5_neck0.npz), so the idle holds the neck near 0;
- neck gains kp 30 N*m/rad and kd 3 N*m*s/rad (yaw and pitch), and neck targets clipped to the K1 firmware range;
- Head_2 mass (1.5 to 5.0 kg) and COM (+-40 mm forward and up, +-10 mm sideways) randomized per environment.
Rewards and terminations are unchanged.
"""
import gymnasium as gym

gym.register(
    id="Booster-K1-Pikachu_Idle-Scale0842-HeadV2Sim-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:RoughWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Booster-K1-Pikachu_Idle-Scale0842-HeadV2Sim-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PlayFlatWoStateEstimationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg",
    },
)
