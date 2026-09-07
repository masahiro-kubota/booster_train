"""Check the actual task, command clock, reset, shell and actor inputs; no learned-gait claim."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument("--output",required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
launcher=AppLauncher(args)
app=launcher.app

import json
import xml.etree.ElementTree as ET
import gymnasium as gym
import numpy as np
from PIL import Image
import torch
from pxr import PhysxSchema, UsdPhysics
import booster_train.tasks
from isaaclab_tasks.utils import parse_env_cfg
from booster_train.tasks.manager_based.locomotion.robots.k1.pikachu_idle_walk import mdp


def main():
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    task="Booster-K1-Pikachu-IdleWalk-Scale0842-v0-Play"
    cfg=parse_env_cfg(task,device=args.device,num_envs=1)
    cfg.seed=42
    cfg.commands.velocity.resampling_time_range=(1e9,1e9)
    env=gym.make(task,cfg=cfg,render_mode="rgb_array" if args.enable_cameras else None).unwrapped
    try:
        obs,_=env.reset()
        robot=env.scene["robot"]
        initial_clearance=mdp.shell_clearance(env).item()
        assert initial_clearance >=.0058,initial_clearance
        assert not mdp.shell_unsafe(env).any()
        assert env.action_manager.total_action_dim==22
        assert obs["policy"].shape==(1,1010),obs["policy"].shape
        assert all(torch.isfinite(v).all() for v in obs.values())
        shell=env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot/Trunk/PikachuTorsoShell")
        assert shell.IsValid()
        approximation=UsdPhysics.MeshCollisionAPI(shell).GetApproximationAttr().Get()
        assert approximation=="sdf",approximation
        resolution=PhysxSchema.PhysxSDFMeshCollisionAPI(shell).GetSdfResolutionAttr().Get()
        filters=[str(p) for p in UsdPhysics.FilteredPairsAPI(shell).GetFilteredPairsRel().GetTargets()]
        assert any(p.endswith("Head_2") for p in filters)
        assert not any(p.endswith("Left_Shank") for p in filters)
        urdf=ET.parse(cfg.scene.robot.spawn.asset_path).getroot()
        source_trunk_mass=float(urdf.find("link[@name='Trunk']/inertial/mass").get("value"))
        trunk_mass=robot.data.default_mass[0,robot.body_names.index("Trunk")].item()
        manifest=json.loads((Path(cfg.shell_directory)/'costume_mass_properties.json').read_text())
        expected_mass=manifest['models']['nominal']['links']['Trunk']['combined']['mass_kg']
        assert abs(trunk_mass-expected_mass)<1e-5,(trunk_mass,expected_mass)
        if args.enable_cameras:
            # The first render creates the annotator and may return a black image.
            for _ in range(32):
                pixels=env.render()
                if pixels.std()>2:
                    break
            assert pixels.std()>2,"Renderer did not produce a nonblank frame"
            Image.fromarray(pixels[:,:,:3]).save(output/"initial-state.png")
        command=env.command_manager.get_term("idle")
        velocity=env.command_manager.get_term("velocity")
        writes=[]
        saved={}
        for name in ("write_root_pose_to_sim","write_root_velocity_to_sim","write_joint_state_to_sim"):
            saved[name]=getattr(robot,name)
            setattr(robot,name,lambda *a,**k: writes.append(True))
        states=[]
        try:
            for step in range(3250):
                if step in (150,1000,2000,2800):
                    target={150:[.05,0,0],1000:[.1,0,0],2000:[0,.04,.15],2800:[0,0,0]}[step]
                    previous=command.elapsed.clone()
                    velocity.set_target(slice(None),torch.tensor(target,device=env.device))
                    assert torch.equal(previous,command.elapsed)
                env.command_manager.compute(env.step_dt)
                if step%25==0:
                    states.append([step*.02,*velocity.command[0].tolist(),command.sampled["phase"].item()])
            assert not writes
            assert abs(command.elapsed.item()-65.)<1e-6
            assert velocity.command.abs().max()<1e-7
            # Reset after a nonzero command and nonzero episode_length, as the real manager does.
            velocity.set_target(slice(None),torch.tensor([.1,0,0],device=env.device))
            for _ in range(60):velocity.compute(.02)
            env.episode_length_buf[:]=123
            velocity.reset(torch.tensor([0],device=env.device))
            assert velocity.command.abs().max()==0
            for _ in range(90):velocity.compute(.02)
            assert velocity.command.abs().max()==0
        finally:
            for name,func in saved.items():setattr(robot,name,func)
        env.reset()
        # The original Idle is now clear. Use a separate leg-only contact fixture.
        ref=command.reference
        fixture=np.load(Path(cfg.shell_directory)/'contact_test_pose.npz')
        q=torch.tensor(fixture['joint_pos'],device=env.device)[None]
        robot.write_joint_state_to_sim(q,torch.zeros_like(q))
        env.sim.forward()
        env.scene.update(env.step_dt)
        mdp.shell_guard(env).reset(torch.tensor([0],device=env.device))
        unsafe_clearance=mdp.shell_clearance(env).item()
        assert mdp.shell_unsafe(env).all(),unsafe_clearance
        env.reset()
        results={"task":task,"initial_clearance_m":initial_clearance,"contact_fixture_clearance_m":unsafe_clearance,
                 "knee_deg":torch.rad2deg(mdp.knees(env))[0].tolist(),"actor_shape":list(obs["policy"].shape),
                 "critic_shape":list(obs["critic"].shape),"action_dim":22,"sdf_resolution":resolution,
                 "shell_approximation":approximation,"filtered_shell_pairs":filters,
                 "trunk_mass_kg":trunk_mass,"urdf_trunk_mass_kg":source_trunk_mass,
                 "command_only_duration_s":65,"command_clock_state_writes":len(writes),
                 "continuous_command_changes_checked":True,"reset_command_zero_checked":True,
                 "reference_period_s":command.reference.period_s,"physics_dt":env.physics_dt,"control_dt":env.step_dt,
                 "mass_model":"400 mm shell; provisional per-link costume mass/COM/inertia, nominal scenario",
                 "limitations":"65 s test advances command clocks only, not physics; no learned walking performance claim"}
        (output/"runtime.json").write_text(json.dumps(results,indent=2)+"\n")
        np.savetxt(output/"command-clock.csv",np.array(states),delimiter=",",header="time_s,vx_mps,vy_mps,yaw_radps,idle_phase",comments="")
        print(json.dumps(results,indent=2))
    finally:
        env.close()


try:
    main()
finally:
    app.close()
