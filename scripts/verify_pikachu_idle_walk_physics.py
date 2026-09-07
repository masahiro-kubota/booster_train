"""Actual PhysX checks for 400 mm partition, per-substep guard and loaded inertias."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
parser.add_argument('--mass-scenario',choices=['low','nominal','high'],default='nominal')
parser.add_argument('--physics-dt',type=float,default=.005)
parser.add_argument('--isolate-shell-pairs',action='store_true')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();launcher=AppLauncher(args);app=launcher.app

import json
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from pxr import UsdPhysics,PhysxSchema,PhysicsSchemaTools,Usd,UsdUtils
import omni.physx
import booster_train.tasks
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.sensors import ContactSensorCfg
from booster_train.tasks.manager_based.locomotion.robots.k1.pikachu_idle_walk import mdp
from booster_train.tasks.manager_based.locomotion.robots.k1.pikachu_idle_walk.reference import LEG_BODIES


def main():
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    task='Booster-K1-Pikachu-IdleWalk-Scale0842-v0-Play'
    cfg=parse_env_cfg(task,device=args.device,num_envs=1);cfg.seed=842
    cfg.scene.robot.spawn.mass_scenario=args.mass_scenario
    cfg.sim.dt=args.physics_dt;cfg.decimation=round(.02/args.physics_dt);cfg.sim.render_interval=cfg.decimation
    cfg.scene.shell_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/Trunk',update_period=0.,history_length=1,
        filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/'+n for n in LEG_BODIES])
    if args.isolate_shell_pairs:
        original_spawn=cfg.scene.robot.spawn.func
        def isolated_spawn(*a,**kw):
            prim=original_spawn(*a,**kw)
            for p in list(Usd.PrimRange(prim)):
                if p.IsInstance():p.SetInstanceable(False)
            for p in Usd.PrimRange(prim):
                if p.HasAPI(UsdPhysics.CollisionAPI) and not p.GetName().startswith(('PikachuLegEnvelope','PikachuTorsoShell')):
                    UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
            return prim
        cfg.scene.robot.spawn.func=isolated_spawn
    env=gym.make(task,cfg=cfg,render_mode='rgb_array' if args.enable_cameras else None).unwrapped
    pairs=[]
    def contact_callback(headers,data):
        for h in headers:
            a=str(PhysicsSchemaTools.intToSdfPath(h.collider0));b=str(PhysicsSchemaTools.intToSdfPath(h.collider1))
            if 'PikachuTorsoShell' in a+b:pairs.append([a,b])
    subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact_callback)
    try:
        obs,_=env.reset();robot=env.scene['robot'];directory=Path(cfg.shell_directory)
        enabled_native=[str(p.GetPath()) for p in Usd.PrimRange(env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot'),Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.CollisionAPI) and not p.GetName().startswith(('PikachuLegEnvelope','PikachuTorsoShell')) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        assert bool(enabled_native) != args.isolate_shell_pairs
        mass=json.loads((directory/'costume_mass_properties.json').read_text())
        expected=mass['models'][args.mass_scenario]['links'];loaded={}
        inertias=robot.root_physx_view.get_inertias().cpu().numpy()[0].reshape(-1,3,3)
        for name,row in expected.items():
            i=robot.body_names.index(name);wanted=row['combined']
            m=float(robot.data.default_mass[0,i]);c=robot.data.body_com_pos_b[0,i].cpu().numpy()
            q=robot.data.body_com_quat_b[0,i].cpu().numpy()
            # ArticulationView returns COM-centered tensors expressed in link
            # axes (including off-diagonal entries), not the diagonal mass axes.
            tensor=inertias[i]
            rot=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            np.testing.assert_allclose(rot@np.diag(wanted['principal_inertia_kg_m2'])@rot.T,
                                       wanted['inertia_kg_m2'],rtol=1e-4,atol=2e-7)
            np.testing.assert_allclose(m,wanted['mass_kg'],rtol=2e-6)
            np.testing.assert_allclose(c,wanted['com_m'],atol=2e-7)
            np.testing.assert_allclose(tensor,wanted['inertia_kg_m2'],rtol=1e-4,atol=2e-7)
            loaded[name]={'mass_kg':m,'com_m':c.tolist(),'inertia_kg_m2':tensor.tolist()}
        np.testing.assert_allclose(robot.data.default_mass.sum().item(),mass['models'][args.mass_scenario]['robot_total_mass_kg'],atol=1e-5)
        shell=env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/Trunk/PikachuTorsoShell')
        assert UsdPhysics.MeshCollisionAPI(shell).GetApproximationAttr().Get()=='sdf'
        hulls=[]
        source_hulls=np.load(directory/'leg_convex_hulls.npz')
        for pi,bi in enumerate(source_hulls['part_body_indices']):
            name=str(source_hulls['body_names'][bi])
            p=env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/'+name+f'/PikachuLegEnvelope_{pi:03d}')
            assert p.IsValid()
            assert PhysxSchema.PhysxConvexHullCollisionAPI(p).GetHullVertexLimitAttr().Get()==256
            filters=UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets()
            assert all(str(v) not in ('/World/envs/env_0/Robot/Trunk',str(shell.GetPath())) for v in filters)
            assert any(str(v)=='/World/ground' for v in filters)
            hulls.append(str(p.GetPath()))
        initial=mdp.shell_clearance(env).item();assert initial>.0297,initial
        assert obs['policy'].shape==(1,1010)
        if args.enable_cameras:
            for _ in range(32):
                pixels=env.render()
                if pixels.std()>2:break
            assert pixels.std()>2
            Image.fromarray(pixels[:,:,:3]).save(out/'initial-state.png')
        before=mdp.shell_guard(env).samples;term=0
        for k in range(12):
            obs,reward,terminated,truncated,extras=env.step(torch.zeros((1,22),device=env.device))
            assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
            term+=int(terminated.sum())
        observed=mdp.shell_guard(env).samples-before
        assert observed>=12*cfg.decimation,(observed,cfg.decimation)
        env.reset()
        # Inspect the collision meshes actually cooked by PhysX.
        cooked={}
        source_hulls=np.load(directory/'leg_convex_hulls.npz')
        for pi,bi in enumerate(source_hulls['part_body_indices']):
            name=str(source_hulls['body_names'][bi])
            path='/World/envs/env_0/Robot/'+name+f'/PikachuLegEnvelope_{pi:03d}'
            def cooked_callback(result,convexes):
                rows=[]
                for convex in convexes:
                    verts=np.array([[v.x,v.y,v.z] for v in convex.vertices])
                    planes=np.array([[p.plane.x,p.plane.y,p.plane.z,p.plane.w] for p in convex.polygons])
                    points=source_hulls[f'{pi}_vertices']
                    outside=(points@planes[:,:3].T+planes[:,3]).max()
                    rows.append({'vertices':len(verts),'polygons':len(planes),'source_hull_outside_m':float(outside)})
                cooked[path]={'result':str(result),'hulls':rows}
                assert len(rows)==1 and rows[0]['source_hull_outside_m']<.0001,(path,rows)
            omni.physx.get_physx_cooking_interface().request_convex_collision_representation(
                UsdUtils.StageCache.Get().GetId(env.sim.stage).ToLongInt(),PhysicsSchemaTools.sdfPathToInt(path),False,cooked_callback)
        assert len(cooked)==len(source_hulls['part_body_indices'])
        assert all(len(v['hulls'])==1 and v['hulls'][0]['source_hull_outside_m']<.0001 for v in cooked.values()),cooked
        (out/'cooked-hulls.json').write_text(json.dumps(cooked,indent=2)+'\n')
        fixture=np.load(directory/'contact_test_pose.npz')
        baseq=robot.data.default_joint_pos.clone()
        contact_cases=[];physical_pairs=[];forces=[];distances=[];fixture_distance=None
        for factor in (1.,1.05,1.1):
            env.reset();pairs.clear()
            target=torch.tensor(fixture['joint_pos'],device=env.device)[None]
            q=baseq+factor*(target-baseq)
            limits=robot.data.joint_pos_limits
            q=q.clamp(limits[:,:,0]+1e-5,limits[:,:,1]-1e-5)
            robot.write_joint_state_to_sim(q,torch.zeros_like(q));robot.set_joint_position_target(q)
            env.sim.forward();env.scene.update(env.physics_dt)
            guard=mdp.shell_guard(env);guard.reset(torch.tensor([0],device=env.device))
            fixture_distance=mdp.shell_clearance(env).item()
            forces=[];distances=[];contact_names=set()
            for k in range(12):
                env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(env.physics_dt)
                matrix=env.scene['shell_contacts'].data.force_matrix_w.norm(dim=-1)[0,0]
                forces.append(float(matrix.max()))
                contact_names.update(LEG_BODIES[i] for i in torch.where(matrix>0)[0].tolist())
                distances.append(float(guard.checker.compute(robot.data.body_link_pos_w,robot.data.body_link_quat_w).item()))
            physical_pairs=[p for p in pairs if any('PikachuLegEnvelope' in x for x in p)]
            contact_cases.append({'factor':factor,'fixture_distance_m':fixture_distance,'peak_force_N':max(forces),'pairs':pairs,'distances':distances})
            if max(forces)>0:break
        (out/'contact-diagnostics.json').write_text(json.dumps(contact_cases,indent=2)+'\n')
        print('CONTACT_DIAGNOSTIC',json.dumps(contact_cases),flush=True)
        assert fixture_distance<=.002,fixture_distance
        assert max(forces)>0,forces
        # Rigid contact tensor gives body-pair forces. In the isolated run the
        # only enabled shapes are the hard shell and leg envelopes. GPU callback
        # reports are not required: they may not be dispatched in headless steps.
        contact_body_names=sorted(contact_names)
        assert contact_body_names
        env.reset()
        result={'status':'PASS','mass_scenario':args.mass_scenario,'physics_dt':env.physics_dt,'control_dt':env.step_dt,
                'partition_construction_z_m':.4,'source_shell_sha256':mass['shell_sha256'],
                'initial_clearance_m_capped':initial,'loaded_link_inertias':loaded,
                'total_mass_kg':float(robot.data.default_mass.sum()),'auxiliary_leg_hulls':hulls,
                'control_steps':12,'physics_samples_checked':observed,'untrained_posture_terminations':term,
                'contact_fixture_clearance_m':fixture_distance,'contact_force_peak_N':max(forces),
                'contact_callback_pairs':sorted({tuple(p) for p in physical_pairs}),
                'contact_tensor_body_pairs':[['Trunk',n] for n in contact_body_names],
                'native_colliders_disabled_for_pair_isolation':args.isolate_shell_pairs,
                'enabled_native_collider_count':len(enabled_native),
                'contact_forces_N':forces,'post_contact_separation_m':distances,
                'limitations':['12 control steps are startup/detection checks, not trained gait evaluation.',
                    'Costume inertias are provisional estimates. The guard bounds interpolated motion between physics samples; real hardware is not certified.']}
        (out/'runtime.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        subscription=None
        env.close()


try:
    main()
except BaseException:
    import traceback,sys
    traceback.print_exc();sys.stderr.flush()
    raise
finally:app.close()
