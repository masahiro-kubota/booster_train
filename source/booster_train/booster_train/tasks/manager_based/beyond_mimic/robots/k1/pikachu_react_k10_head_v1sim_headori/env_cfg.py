from isaaclab.utils import configclass
from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v1sim_headori import env_cfg as v1

MOTION_FILE = f"{BOOSTER_ASSETS_DIR}/motions/K1/k1_pikachu_react_k10.npz"
EPISODE_LENGTH_S = 14.5    # 手本は 14.32 秒（716 コマ）。idle の 13 秒だと、手本の終わりまで 1 回の episode で届かない
TAIL_LEN = 0               # idle は 3 回つなげた手本の最後 615 コマを開始位置から外していた。T215 は 1 本の手本なので外さない


def use_t215(cfg):
    cfg.commands.motion.motion_file = MOTION_FILE
    cfg.commands.motion.tail_len = TAIL_LEN
    cfg.episode_length_s = EPISODE_LENGTH_S


@configclass
class RoughWoStateEstimationEnvCfg(v1.RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_t215(self)


@configclass
class PlayFlatWoStateEstimationEnvCfg(v1.PlayFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_t215(self)
