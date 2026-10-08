"""Check in PhysX that the v2 head simulation task set up what project-memos/v2-head-sim-20261008 plans.

Nominal Head_2 inertial from the mass file, per-environment Head_2 mass and COM within the randomization ranges
(inertia scaled with mass), neck gains, neck action scale and target clip, and the reference neck pitch.

    env -u PYTHONPATH <isaaclab python> scripts/verify_head_v2sim.py --headless
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='Booster-K1-Pikachu_Idle-Scale0842-HeadV2Sim-v0-Play')
parser.add_argument('--num_envs', type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); app = AppLauncher(args).app

import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import booster_train.tasks  # noqa: F401  (registers the tasks)
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v2sim import env_cfg as v2
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        env.reset(); robot = env.scene['robot']; view = robot.root_physx_view
        h2 = robot.body_names.index('Head_2')
        wanted = json.loads(Path(v2.MASS_FILE).read_text())['links']['Head_2']['combined']

        # Nominal inertial = mass file (what the spawn wrote before the startup events).
        np.testing.assert_allclose(float(robot.data.default_mass[0, h2]), wanted['mass_kg'], rtol=2e-6)
        print(f"[verify] nominal Head_2 {wanted['mass_kg']:.4f} kg, com {np.round(np.array(wanted['com_m']) * 1000, 2).tolist()} mm  OK")

        # Randomized mass, inertia scaled with it, and COM offsets.
        mass = view.get_masses()[:, h2].numpy(); lo, hi = v2.HEAD_MASS_KG
        assert mass.min() >= lo - 1e-5 and mass.max() <= hi + 1e-5, (mass.min(), mass.max())
        assert mass.min() < lo + 0.3 and mass.max() > hi - 0.3, 'mass not spread over the range'
        inert = view.get_inertias()[:, h2].numpy(); inert0 = robot.data.default_inertia[:, h2].cpu().numpy()
        np.testing.assert_allclose(inert, inert0 * (mass / wanted['mass_kg'])[:, None], rtol=1e-4, atol=1e-8)
        com = view.get_coms()[:, h2, :3].numpy(); off = com - np.array(wanted['com_m'])
        for k, axis in enumerate('xyz'):
            a, b = v2.HEAD_COM_RANGE_M[axis]
            assert off[:, k].min() >= a - 1e-6 and off[:, k].max() <= b + 1e-6, (axis, off[:, k].min(), off[:, k].max())
            assert off[:, k].min() < a * 0.8 and off[:, k].max() > b * 0.8, f'COM {axis} not spread over the range'
        print(f'[verify] Head_2 mass {mass.min():.2f}..{mass.max():.2f} kg over {len(mass)} envs, inertia scaled with mass  OK')
        print(f'[verify] Head_2 COM offset x {off[:, 0].min() * 1000:+.1f}..{off[:, 0].max() * 1000:+.1f}, '
              f'y {off[:, 1].min() * 1000:+.1f}..{off[:, 1].max() * 1000:+.1f}, z {off[:, 2].min() * 1000:+.1f}..{off[:, 2].max() * 1000:+.1f} mm  OK')

        # Neck gains.
        head = robot.actuators['head']
        assert head.joint_names == ['AAHead_yaw', 'Head_pitch'], head.joint_names
        np.testing.assert_allclose(head.stiffness.cpu().numpy(), v2.NECK_KP); np.testing.assert_allclose(head.damping.cpu().numpy(), v2.NECK_KD)
        print(f'[verify] neck kp {v2.NECK_KP}, kd {v2.NECK_KD}  OK')

        # Neck action scale and target clip.
        term = env.action_manager.get_term('joint_pos'); names = term._joint_names
        ids = [names.index(n) for n in ('AAHead_yaw', 'Head_pitch')]
        scale = term._scale[0, ids].cpu().numpy() if torch.is_tensor(term._scale) else np.full(2, term._scale)
        np.testing.assert_allclose(scale, 0.25 * 6.0 / v2.NECK_KP, rtol=1e-6)
        big = torch.zeros(env.num_envs, len(names), device=env.device); big[:, ids] = torch.tensor([-100.0, -100.0], device=env.device)
        term.process_actions(big); low = term.processed_actions[0, ids].cpu().numpy()
        big[:, ids] = 100.0; term.process_actions(big); high = term.processed_actions[0, ids].cpu().numpy()
        np.testing.assert_allclose(low, [v2.NECK_TARGET_CLIP[n][0] for n in ('AAHead_yaw', 'Head_pitch')], atol=1e-6)
        np.testing.assert_allclose(high, [v2.NECK_TARGET_CLIP[n][1] for n in ('AAHead_yaw', 'Head_pitch')], atol=1e-6)
        print(f'[verify] neck action scale {scale[0]:.4f} rad, targets clipped to yaw {low[0]:+.4f}..{high[0]:+.4f}, '
              f'pitch {low[1]:+.4f}..{high[1]:+.4f} rad  OK')

        # Reference neck pitch.
        motion = env.command_manager.get_term('motion').motion
        ref = motion.joint_pos[:, robot.joint_names.index('Head_pitch')].cpu().numpy()
        assert np.abs(ref).max() < 1e-5, ref
        print(f'[verify] reference Head_pitch {ref.min():+.5f}..{ref.max():+.5f} rad over {len(ref)} frames  OK')
        print(f'[verify] robot nominal total mass {robot.data.default_mass[0].sum().item():.4f} kg')
    finally:
        env.close()


if __name__ == '__main__':
    main(); app.close()
