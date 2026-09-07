"""K1 with explicit costume inertias and leg-to-hard-torso collision geometry."""
from pathlib import Path
import hashlib
import json

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .reference import LEG_BODIES


@sim_utils.clone
def spawn_with_torso(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = sim_utils.spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    bodies = {p.GetName(): p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    native_colliders = [p.GetPath() for p in Usd.PrimRange(prim,Usd.TraverseInstanceProxies())
                        if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert native_colliders, 'URDF instance-proxy colliders must be included in the auxiliary filters'
    directory=Path(cfg.shell_file).parent
    source=json.loads(Path(cfg.shell_file).with_suffix('.json').read_text())
    if source.get('material_boundary_construction_z_m') != .4:
        raise ValueError('This task requires the 400 mm material partition')
    trunk = bodies["Trunk"]
    path = str(trunk.GetPath()) + "/PikachuTorsoShell"
    with np.load(cfg.shell_file, allow_pickle=False) as d:
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(d["vertices"].tolist())
        mesh.CreateFaceVertexCountsAttr([3]*len(d["faces"]))
        mesh.CreateFaceVertexIndicesAttr(d["faces"].reshape(-1).tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.9,0.68,0.08)])
    mesh.CreateDisplayOpacityAttr([0.32])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(PhysxSchema.Tokens.sdf)
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh.GetPrim())
    sdf.CreateSdfResolutionAttr(cfg.sdf_resolution)
    sdf.CreateSdfSubgridResolutionAttr(6)
    collider = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
    collider.CreateContactOffsetAttr(0.001)
    collider.CreateRestOffsetAttr(0.0)
    filtered = UsdPhysics.FilteredPairsAPI.Apply(mesh.GetPrim()).CreateFilteredPairsRel()
    # Trunk also carries ArticulationRootAPI. Filtering that path suppresses
    # contact with the ENTIRE robot, not merely the torso body's native shapes.
    # A body's own colliders cannot self-collide and need no explicit exclusion.
    filtered.SetTargets([p.GetPath() for n,p in bodies.items() if n not in LEG_BODIES and n!='Trunk'])
    # Auxiliary hulls only interact with the hard shell. Original colliders keep
    # their ground and self-contact behavior. Do not filter the whole Trunk body:
    # that would also suppress the newly attached hard shell.
    auxiliaries=[]
    with np.load(directory/'leg_convex_hulls.npz',allow_pickle=False) as d:
        for i,body_id in enumerate(d['part_body_indices']):
            name=str(d['body_names'][body_id])
            hull=UsdGeom.Mesh.Define(stage,str(bodies[name].GetPath())+f'/PikachuLegEnvelope_{i:03d}')
            hull.CreatePointsAttr(d[f'{i}_physics_vertices'].tolist())
            faces=d[f'{i}_physics_faces']
            hull.CreateFaceVertexCountsAttr([3]*len(faces))
            hull.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
            hull.CreateSubdivisionSchemeAttr('none')
            hull.CreateVisibilityAttr('invisible')
            hp=hull.GetPrim()
            UsdPhysics.CollisionAPI.Apply(hp)
            UsdPhysics.MeshCollisionAPI.Apply(hp).CreateApproximationAttr('convexHull')
            cooking=PhysxSchema.PhysxConvexHullCollisionAPI.Apply(hp)
            cooking.CreateHullVertexLimitAttr(256)
            cooking.CreateMinThicknessAttr(0.)
            contact=PhysxSchema.PhysxCollisionAPI.Apply(hp)
            contact.CreateContactOffsetAttr(.001)
            contact.CreateRestOffsetAttr(0.)
            auxiliaries.append(hp)
    for hp in auxiliaries:
        UsdPhysics.FilteredPairsAPI.Apply(hp).CreateFilteredPairsRel().SetTargets(
            native_colliders+[p.GetPath() for p in auxiliaries if p!=hp]+[Sdf.Path('/World/ground')])
    masses=json.loads((directory/'costume_mass_properties.json').read_text())
    if masses['shell_sha256'] != hashlib.sha256(Path(cfg.shell_file).read_bytes()).hexdigest():
        raise ValueError('Costume inertia is stale relative to the torso mesh; rebuild the mass model')
    for name,row in masses['models'][cfg.mass_scenario]['links'].items():
        data=row['combined'];api=UsdPhysics.MassAPI.Apply(bodies[name])
        api.CreateMassAttr(data['mass_kg'])
        api.CreateCenterOfMassAttr(Gf.Vec3f(*data['com_m']))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*data['principal_inertia_kg_m2']))
        q=data['principal_axes_wxyz']
        api.CreatePrincipalAxesAttr(Gf.Quatf(q[0],Gf.Vec3f(*q[1:])))
    mesh.GetPrim().CreateAttribute('pikachu:massModel',Sdf.ValueTypeNames.String).Set('provisional_'+cfg.mass_scenario)
    return prim


@configclass
class TorsoUrdfFileCfg(sim_utils.UrdfFileCfg):
    func: object = spawn_with_torso
    shell_file: str = ""
    sdf_resolution: int = 1024
    mass_scenario: str = 'nominal'
