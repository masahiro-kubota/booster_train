from dataclasses import fields

import isaaclab.envs.mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.assets.robots.booster import K1_ACTION_SCALE
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842 import env_cfg as base
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_r27.link_mass import (
    LinkMassUrdfFileCfg, spawn_with_link_masses)

MASS_FILE = f"{BOOSTER_ASSETS_DIR}/robots/K1/pikachu_scale0842/head_v2sim_mass_properties.json"
MOTION_FILE = f"{BOOSTER_ASSETS_DIR}/motions/K1/k1_pikachu_idle_scale0842_pingpong_tile3_blender50_grounded_hold0p5_neck0.npz"
NECK_KP, NECK_KD = 30.0, 3.0
# K1 firmware clips head commands to these limits (docs/k1/README.md 4-3, 4-5); the URDF joint limits stay.
NECK_TARGET_CLIP = {"AAHead_yaw": (-1.0297, 1.0297), "Head_pitch": (-0.2967, 0.7505)}
HEAD_MASS_KG = (1.5, 5.0)                                                   # whole Head_2, absolute
HEAD_COM_RANGE_M = {"x": (-0.04, 0.04), "y": (-0.01, 0.01), "z": (-0.04, 0.04)}  # Head_2 frame, around the nominal


def use_v2sim_head(cfg):
    """Nominal head inertial, shifted reference, neck gains and target clip, and head mass/COM randomization."""
    source = cfg.scene.robot.spawn
    values = {f.name: getattr(source, f.name) for f in fields(source) if f.init}
    values["func"] = spawn_with_link_masses
    cfg.scene.robot.spawn = LinkMassUrdfFileCfg(**values, mass_file=MASS_FILE)
    cfg.commands.motion.motion_file = MOTION_FILE

    head = cfg.scene.robot.actuators["head"]
    head.stiffness, head.damping = NECK_KP, NECK_KD
    scale = dict(K1_ACTION_SCALE)
    scale[".*Head.*"] = 0.25 * head.effort_limit_sim / NECK_KP   # same rule as K1_ACTION_SCALE and booster_deploy
    cfg.actions.joint_pos.scale = scale
    cfg.actions.joint_pos.clip = dict(NECK_TARGET_CLIP)

    cfg.events.head_mass = EventTerm(
        func=isaac_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Head_2"),
            "mass_distribution_params": HEAD_MASS_KG,
            "operation": "abs",
            "recompute_inertia": True,
        },
    )
    cfg.events.head_com = EventTerm(
        func=isaac_mdp.randomize_rigid_body_com,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="Head_2"), "com_range": dict(HEAD_COM_RANGE_M)},
    )


@configclass
class RoughWoStateEstimationEnvCfg(base.RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_v2sim_head(self)


@configclass
class PlayFlatWoStateEstimationEnvCfg(base.PlayFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_v2sim_head(self)
