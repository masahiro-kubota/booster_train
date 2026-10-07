"""Spawn the K1 URDF, then replace the inertials of the links listed in a mass file (links.<name>.combined)."""
import json
from pathlib import Path

from pxr import Gf, Usd, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


@sim_utils.clone
def spawn_with_link_masses(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = sim_utils.spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)
    bodies = {p.GetName(): p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    links = json.loads(Path(cfg.mass_file).read_text())["links"]
    for name, row in links.items():
        data = row["combined"]
        api = UsdPhysics.MassAPI.Apply(bodies[name])
        api.CreateMassAttr(data["mass_kg"])
        api.CreateCenterOfMassAttr(Gf.Vec3f(*data["com_m"]))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*data["principal_inertia_kg_m2"]))
        q = data["principal_axes_wxyz"]
        api.CreatePrincipalAxesAttr(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
    return prim


@configclass
class LinkMassUrdfFileCfg(sim_utils.UrdfFileCfg):
    func: object = spawn_with_link_masses
    mass_file: str = ""
