"""Check in PhysX that a HeadV2Fix task (pikachu_idle_scale0842_head_v2fix) or HeadV2Sim-HeadOri
(pikachu_idle_scale0842_head_v2sim_headori) set up its head and rewards.

HeadV2Fix: Head_2 fixed at 2.0 kg in every environment with the inertia scaled from the mass file and the COM at the
file's value. HeadV2Sim-HeadOri: Head_2 mass spread over the randomization range with the inertia scaled per
environment and the COM within the range around the file's value. Both: the neck gains, action scale and target clip,
and the reward terms of the variant (HeadOri or Generic).

    env -u PYTHONPATH <isaaclab python> scripts/verify_head_v2fix.py --headless --task Booster-K1-Pikachu_Idle-Scale0842-HeadV2Fix-HeadOri-v0
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True)
parser.add_argument('--num_envs', type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); app = AppLauncher(args).app

import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import booster_train.tasks  # noqa: F401  (registers the tasks)
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v2fix import env_cfg as fix
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v2sim import env_cfg as v2
from isaaclab_tasks.utils import parse_env_cfg

GENERIC = {'motion_global_anchor_pos': 0.5, 'motion_global_anchor_ori': 0.5, 'motion_body_pos': 1.0, 'motion_body_ori': 1.0,
           'motion_body_lin_vel': 1.0, 'motion_body_ang_vel': 1.0, 'action_rate_l2': -0.1, 'joint_limit': -10.0, 'undesired_contacts': -0.1}
EMPHASIS = {'motion_foot_ori': 15.0, 'motion_foot_pos': 30.0, 'motion_hand_ori': 10.0, 'motion_hand_pos': 15.0,
            'motion_trunk_ori': 30.0, 'motion_trunk_pos': 20.0, 'motion_trunk_ang_vel': 5.0}


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        env.reset(); robot = env.scene['robot']; view = robot.root_physx_view; h2 = robot.body_names.index('Head_2')
        wanted = json.loads(Path(v2.MASS_FILE).read_text())['links']['Head_2']['combined']
        mass = view.get_masses()[:, h2].numpy()
        inert = view.get_inertias()[:, h2].numpy(); inert0 = robot.data.default_inertia[:, h2].cpu().numpy()
        com = view.get_coms()[:, h2, :3].numpy()
        if 'HeadV2Fix' in args.task:
            np.testing.assert_allclose(mass, fix.HEAD_MASS_KG, rtol=1e-6)
            np.testing.assert_allclose(com, np.broadcast_to(wanted['com_m'], com.shape), atol=1e-6)
        else:
            lo, hi = v2.HEAD_MASS_KG
            assert mass.min() >= lo - 1e-6 and mass.max() <= hi + 1e-6 and mass.max() - mass.min() > 0.8 * (hi - lo), (mass.min(), mass.max())
            shift = com - np.array(wanted['com_m'])
            for k, ax in enumerate('xyz'):
                a, b = v2.HEAD_COM_RANGE_M[ax]
                assert shift[:, k].min() >= a - 1e-6 and shift[:, k].max() <= b + 1e-6 and np.ptp(shift[:, k]) > 0.8 * (b - a), (ax, shift[:, k].min(), shift[:, k].max())
        np.testing.assert_allclose(inert, inert0 * mass[:, None] / wanted['mass_kg'], rtol=1e-4, atol=1e-8)
        print(f"[verify] Head_2 {mass.min():.3f}..{mass.max():.3f} kg over {len(mass)} envs, inertia scaled, "
              f"com {np.round(com[0] * 1000, 2).tolist()} mm (file {np.round(np.array(wanted['com_m']) * 1000, 2).tolist()})  OK")

        head = robot.actuators['head']
        np.testing.assert_allclose(head.stiffness.cpu().numpy(), v2.NECK_KP); np.testing.assert_allclose(head.damping.cpu().numpy(), v2.NECK_KD)
        term = env.action_manager.get_term('joint_pos'); names = term._joint_names
        ids = [names.index(n) for n in ('AAHead_yaw', 'Head_pitch')]
        scale = term._scale[0, ids].cpu().numpy() if torch.is_tensor(term._scale) else np.full(2, term._scale)
        np.testing.assert_allclose(scale, 0.25 * 6.0 / v2.NECK_KP, rtol=1e-6)
        big = torch.zeros(env.num_envs, len(names), device=env.device); big[:, ids] = -100.0
        term.process_actions(big); low = term.processed_actions[0, ids].cpu().numpy()
        np.testing.assert_allclose(low, [v2.NECK_TARGET_CLIP[n][0] for n in ('AAHead_yaw', 'Head_pitch')], atol=1e-6)
        print(f'[verify] neck kp {v2.NECK_KP}, kd {v2.NECK_KD}, action scale {scale[0]:.4f} rad, lower clip {low.round(4).tolist()}  OK')

        rm = env.reward_manager
        got = {n: rm.get_term_cfg(n).weight for n in rm.active_terms}
        if 'HeadOri' in args.task:
            want = {**GENERIC, 'action_rate_l2': -2.0, 'undesired_contacts': -10.0, **EMPHASIS, 'motion_head_ori': fix.HEAD_ORI_WEIGHT}
            head_cfg = rm.get_term_cfg('motion_head_ori')
            assert head_cfg.params['body_names'] == ['Head_2'] and head_cfg.params['std'] == fix.HEAD_ORI_STD, head_cfg.params
        else:
            want = GENERIC
            assert 'left_hand_link' not in str(rm.get_term_cfg('undesired_contacts').params['sensor_cfg'].body_names) or \
                '(?!left_hand_link$)' in str(rm.get_term_cfg('undesired_contacts').params['sensor_cfg'].body_names)
        assert got == want, {'missing': set(want) - set(got), 'extra': set(got) - set(want),
                             'weights': {k: (got[k], want[k]) for k in set(got) & set(want) if got[k] != want[k]}}
        print(f'[verify] reward terms {got}  OK')
    finally:
        env.close()


if __name__ == '__main__':
    main(); app.close()
