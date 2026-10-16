"""Check in PhysX that Booster-K1-Pikachu-ShuffleWalk-v0 samples its gait reference and starts on it.

1. The velocity command stays within the gait library ranges and the gait phase advances by step_dt / cycle.
2. After reset, joint positions equal the reference (except the Idle starts, which equal reset_pose.npz).
3. Replaying the reference joint positions open loop through the PD targets from the reset: one step after the reset the torso
   height is within 10 mm of the reference (root height agrees with the ground) and, where the reference has one foot in swing,
   that foot is the higher one (contact columns follow the feet). Over 0.3 s the stance contact and the sag are only printed,
   next to the Idle starts: open loop, the crouched pose bounces after the reset and sags about 45 mm under the leg PD gains,
   more than the 12 mm lift, so the swing feet stay on the ground until a policy compensates.
4. The head orientation reward's neck frame reproduces Head_2 from the torso and the actual neck joints (frame convention).
5. Head settings (neck gains, clip, Head_2 mass range) and the reward terms.
6. The action offset is the Idle frame 0 pose: with zero actions for 0.3 s from the reset, the arm and neck joints stay within
   15 deg rms of the reference (with the K1 default as the offset, a zero action takes them 74 deg rms away, too far for the
   joint reward).

    env -u PYTHONPATH <isaaclab python> scripts/verify_shuffle_walk.py --headless --num_envs 64
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='Booster-K1-Pikachu-ShuffleWalk-v0')
parser.add_argument('--num_envs', type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.math import quat_error_magnitude, quat_mul
import booster_train.tasks  # noqa: F401  (registers the tasks)
from booster_train.tasks.manager_based.locomotion.robots.k1.pikachu_shuffle_walk import mdp as walk
from booster_train.tasks.manager_based.locomotion.robots.k1.pikachu_shuffle_walk.env_cfg import UPPER_JOINTS
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v1sim_headori import env_cfg as v1
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v2sim import env_cfg as v2
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.events.push_robot = None
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        env.reset(); robot = env.scene['robot']; gait = env.command_manager.get_term('gait'); vel = env.command_manager.get_command('velocity')
        lib = gait.library
        for c, (lo, hi) in enumerate(lib.ranges):
            assert vel[:, c].min() >= lo - 1e-6 and vel[:, c].max() <= hi + 1e-6, (c, vel[:, c].min(), vel[:, c].max())
        print(f'[verify] velocity command within the library ranges {lib.ranges}  OK')
        idle = (robot.data.joint_pos - gait.idle_joint_pos).abs().amax(-1) < 1e-4
        on_ref = (robot.data.joint_pos - gait.ref['joint_pos']).abs().amax(-1)
        print(f'[verify] reset: {int(idle.sum())} Idle starts, reference starts max joint error {on_ref[~idle].max():.2e} rad')
        assert on_ref[~idle].max() < 1e-3
        term = env.action_manager.get_term('joint_pos'); scale = term._scale; offset = term._offset
        # add_joint_default_pos shifts the offset by up to 10 mrad (calibration error); the limits keep it 1 mrad inside
        np.testing.assert_allclose(offset.cpu().numpy(), gait.idle_joint_pos.expand_as(offset).cpu().numpy(), atol=0.012)
        print(f'[verify] action offset = Idle frame 0 pose (max {(offset - gait.idle_joint_pos).abs().max()*1000:.1f} mrad)  OK')
        phase0 = gait.phase.clone(); hz, stance_ok, steps = [], [], int(round(0.3 / env.step_dt))
        h2 = robot.body_names.index('Head_2'); frame_err, head_err = [], []
        feet = [robot.body_names.index(n) for n in walk.FEET]
        for k in range(steps):
            action = (gait.ref['joint_pos'] - offset) / scale
            env.step(action)
            if k == 0:
                rc = gait.ref['contact'][~idle] > 0.5; single = rc.sum(-1) == 1
                fz = robot.data.body_link_pos_w[~idle][:, feet, 2]
                lift = (fz[single] * (~rc[single]).float()).sum(-1) - (fz[single] * rc[single].float()).sum(-1)
                hz0 = (robot.data.root_pos_w[:, 2] - gait.ref_root_pos_w[:, 2])[~idle].median()
            own = quat_mul(robot.data.root_quat_w, walk.neck_frame(env, robot.data.joint_pos))
            frame_err.append(quat_error_magnitude(robot.data.body_quat_w[:, h2], own))
            head_err.append(torch.sqrt(-torch.log(walk.head_orientation(env, std=1.0).clamp(min=1e-12))))
            hz.append(robot.data.root_pos_w[:, 2] - gait.ref_root_pos_w[:, 2])
            rc_all = gait.ref['contact'] > 0.5
            stance_ok.append((walk.foot_contact(env) & rc_all).float().sum(-1) / rc_all.float().sum(-1).clamp(min=1))
        dphase = ((gait.phase - phase0) % 1.0)
        expect = (steps * env.step_dt / lib.cycle_s) % 1.0
        alive = ~env.termination_manager.terminated
        assert torch.allclose(dphase[alive], torch.full_like(dphase[alive], expect), atol=1e-4), (dphase[:4], expect)
        print(f'[verify] phase advanced {expect:.3f} cycle in {steps} steps  OK')
        print(f'[verify] one step after reset (reference starts): torso height error median {hz0*1000:+.1f} mm; swing foot above the '
              f'stance foot by {lift.mean()*1000:+.1f} mm on average ({int(single.sum())} single-support starts)')
        assert hz0.abs() < 0.01 and lift.mean() > 0.002
        sag = torch.stack(hz)[-1]; stance = torch.stack(stance_ok)
        print(f'[verify] open-loop replay 0.3 s (no policy): stance feet in contact {stance[:, ~idle].mean():.2f} reference starts, '
              f'{stance[:, idle].mean():.2f} Idle starts; torso sag at the end {-sag[~idle].median()*1000:.1f} mm and '
              f'{-sag[idle].median()*1000:.1f} mm')
        frame_err = torch.stack(frame_err).amax().item(); head_err = torch.stack(head_err)[:, ~idle]
        print(f'[verify] neck frame vs Head_2 max {np.degrees(frame_err):.3f} deg; open-loop head orientation error median '
              f'{np.degrees(head_err.median().item()):.1f} deg (reference neck 20 deg up, clipped at 17 deg)')
        assert frame_err < np.radians(0.5)
        env.reset(); upper = robot.find_joints(UPPER_JOINTS, preserve_order=True)[0]
        for _ in range(steps):
            env.step(torch.zeros_like(action))
        up = (robot.data.joint_pos - gait.ref['joint_pos'])[:, upper].square().mean(-1).sqrt()
        print(f'[verify] zero action 0.3 s: arm and neck joints off the reference by {np.degrees(up.median().item()):.1f} deg rms '
              f'(median), {np.degrees(up.max().item()):.1f} deg max')
        assert up.median() < np.radians(15)
        head = robot.actuators['head']
        np.testing.assert_allclose(head.stiffness.cpu().numpy(), v2.NECK_KP); np.testing.assert_allclose(head.damping.cpu().numpy(), v2.NECK_KD)
        mass = robot.root_physx_view.get_masses()[:, h2].numpy()
        lo, hi = v1.HEAD_MASS_KG
        assert mass.min() >= lo - 1e-6 and mass.max() <= hi + 1e-6, (mass.min(), mass.max())
        print(f'[verify] neck kp {v2.NECK_KP} kd {v2.NECK_KD}, Head_2 {mass.min():.2f}..{mass.max():.2f} kg  OK')
        rm = env.reward_manager
        print('[verify] rewards', {n: rm.get_term_cfg(n).weight for n in rm.active_terms})
        print('[verify] observation dims', {k: v.shape for k, v in env.observation_manager.compute().items()})
    finally:
        env.close()


if __name__ == '__main__':
    main(); app.close()
