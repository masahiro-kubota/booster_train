from isaaclab.utils import configclass
from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v1sim_headori import env_cfg as v1ori

# 12 s street-Pikachu motion at 50 Hz (716 frames incl. the hold, prep, settle and tail). Made by T215/k4 (make_motion.py --range booster)
MOTION_FILE = f"{BOOSTER_ASSETS_DIR}/motions/K1/k1_pikachu_street_cute_k4.npz"
TAIL_LEN = 100  # reset frames stop 2 s before the end so that every episode sees the ending
# Elbow pitch targets. -1.80 is the reference value and the deploy-side PREPARE_ELBOW_PITCH; the right elbow stopped at -1.86 on the real robot
# (2026-10-18), where the motor pushed against the stop and heated up. The URDF limit is +-2.27 and the soft limit +-2.043, which would not stop the policy.
ELBOW_TARGET_CLIP = {"Left_Elbow_Pitch": (-1.80, 1.9024), "Right_Elbow_Pitch": (-1.80, 1.9024)}


def use_t215(cfg):
    cfg.commands.motion.motion_file = MOTION_FILE
    cfg.commands.motion.tail_len = TAIL_LEN
    clip = dict(cfg.actions.joint_pos.clip or {})
    clip.update(ELBOW_TARGET_CLIP)
    cfg.actions.joint_pos.clip = clip


@configclass
class RoughWoStateEstimationEnvCfg(v1ori.RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_t215(self)


@configclass
class PlayFlatWoStateEstimationEnvCfg(v1ori.PlayFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_t215(self)
