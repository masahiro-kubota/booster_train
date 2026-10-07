from dataclasses import fields

from isaaclab.utils import configclass
from booster_assets import BOOSTER_ASSETS_DIR
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842 import env_cfg as base
from .link_mass import LinkMassUrdfFileCfg, spawn_with_link_masses

MASS_FILE = f"{BOOSTER_ASSETS_DIR}/robots/K1/pikachu_scale0842/head_r27_mass_properties.json"


def use_head_mass(cfg):
    """Keep every URDF spawn setting; only add the Head_2 inertial override."""
    source = cfg.scene.robot.spawn
    values = {f.name: getattr(source, f.name) for f in fields(source) if f.init}
    values["func"] = spawn_with_link_masses
    cfg.scene.robot.spawn = LinkMassUrdfFileCfg(**values, mass_file=MASS_FILE)


@configclass
class RoughWoStateEstimationEnvCfg(base.RoughWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_head_mass(self)


@configclass
class PlayFlatWoStateEstimationEnvCfg(base.PlayFlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        use_head_mass(self)
