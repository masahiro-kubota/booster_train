"""Scale0842 idle with the v2 sim head fixed at 2.0 kg, to find a reward that makes the neck follow the reference.

Head as in pikachu_idle_scale0842_head_v2sim (mounted 0.349 rad further up and centred, neck0 reference, neck kp 30
and kd 3, firmware target clip), but Head_2 is fixed at 2.0 kg with the nominal COM (no randomization). Two rewards:
- HeadOri: the scale0842 rewards plus a Head_2 orientation term (weight 10, std 0.2, as the hand orientation term);
- Generic: the BeyondMimic generic terms only, as mj_dance_002 (foot, hand and trunk terms removed, action rate
  -0.1, undesired contacts -0.1 with hands and feet allowed).
project-memos/v2-head-sim-20261008 compares the two in MuJoCo.
"""
import gymnasium as gym

for variant in ("HeadOri", "Generic"):
    gym.register(
        id=f"Booster-K1-Pikachu_Idle-Scale0842-HeadV2Fix-{variant}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{variant}RoughWoStateEstimationEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:{variant}PPORunnerCfg",
        },
    )
    gym.register(
        id=f"Booster-K1-Pikachu_Idle-Scale0842-HeadV2Fix-{variant}-v0-Play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{variant}PlayFlatWoStateEstimationEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:{variant}PPORunnerCfg",
        },
    )
