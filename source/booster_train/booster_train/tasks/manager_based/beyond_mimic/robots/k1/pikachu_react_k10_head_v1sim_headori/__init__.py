"""react（街のピカチュウの反応。旧名 T215）の k10 の手本を、頭が v1 の設定（デプロイ中の idle と同じ首・頭）で学習するタスク。報酬と終了条件は idle と同じ。"""
import gymnasium as gym

gym.register(
    id="Booster-K1-Pikachu-React-K10-HeadV1Sim-HeadOri-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:RoughWoStateEstimationEnvCfg", "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg"},
)
gym.register(
    id="Booster-K1-Pikachu-React-K10-HeadV1Sim-HeadOri-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:PlayFlatWoStateEstimationEnvCfg", "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:PPORunnerCfg"},
)
