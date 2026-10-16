"""Check in PhysX that Booster-K1-Pikachu-IdleShuffle-v0 builds its reference and starts on it.

Reference alone (one environment, a command that stands 2 s, ramps to 6 cm/s forward over 1 s, holds 3 s, ramps down 1 s and
stands 3 s):
1. While standing the reference is the Idle motion frame by frame (0.5 s hold, then the 205-frame cycle); walking, the neck
   and arm references still are (the Idle motion keeps running).
2. Walking starts on the first step with a non-zero command; after the command is back at zero the reference returns to
   the Idle motion at the next double support (within half a gait cycle) and equals the running Idle frame after the blend.
3. The blends add no jump: the largest per-step change of the reference joint positions in a blend is no more than in
   steady walking at the same command (printed side by side).
Environment:
4. Standing starts are on an Idle frame and walking starts on the gait reference (within the reset joint noise); both kinds
   occur. The action offset is Idle frame 0 with the arms 3 deg inside their limits.
5. Over 10 s of commands both modes and both switches occur, the arm references stay 3 deg inside the limits and nothing is
   NaN; with zero actions for 0.3 s from standing starts the arm and neck joints stay within 15 deg rms of the reference.
6. Head settings (neck gains, Head_2 mass range), the reward terms and the observation sizes.

    env -u PYTHONPATH <isaaclab python> scripts/verify_idle_shuffle.py --headless --num_envs 64
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='Booster-K1-Pikachu-IdleShuffle-v0')
parser.add_argument('--num_envs', type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); app = AppLauncher(args).app

import math
import gymnasium as gym
import numpy as np
import torch
import booster_train.tasks  # noqa: F401  (registers the tasks)
from booster_train.tasks.manager_based.locomotion.robots.k1.pikachu_idle_shuffle import env_cfg as task
from booster_train.tasks.manager_based.locomotion.robots.k1.pikachu_idle_shuffle.reference import BLEND_S, IdleWalkReference
from booster_train.tasks.manager_based.locomotion.robots.k1.pikachu_shuffle_walk.env_cfg import UPPER_JOINTS
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v1sim_headori import env_cfg as v1
from booster_train.tasks.manager_based.beyond_mimic.robots.k1.pikachu_idle_scale0842_head_v2sim import env_cfg as v2
from isaaclab_tasks.utils import parse_env_cfg


def reference_alone(term, dt):
    ref = IdleWalkReference(term.library, term.motion, 1, term.device)
    ref.reset(torch.tensor([0], device=term.device), torch.tensor([False], device=term.device),
              torch.zeros(1, device=term.device), torch.zeros(1, dtype=torch.long, device=term.device))
    def mj(u):
        return u * u * u * (10 - 15 * u + 6 * u * u)
    times = np.arange(0, 10.0, dt); cmds = []
    for t in times:
        v = 0.06 * (mj(min(max(t - 2.0, 0), 1.0)) if t < 6.0 else 1 - mj(min(t - 6.0, 1.0)))
        cmds.append(0.0 if t < 2.0 or t >= 7.0 else v)
    rows = []
    for c in cmds:
        cmd = torch.tensor([[c, 0.0, 0.0]], dtype=torch.float32, device=term.device)
        ref.step(cmd, dt); r = ref.sample(cmd)
        rows.append((bool(ref.walking[0]), float(ref.alpha[0]), int(ref.idle_k[0]), float(ref.phase[0]), r["joint_pos"][0].clone()))
    walking = np.array([w for w, *_ in rows]); jp = torch.stack([r[-1] for r in rows])
    k_start = int(np.argmax(np.array(cmds) > 1e-4)); assert walking[k_start] and not walking[:k_start].any(), 'walking start'
    for k in range(k_start):
        assert torch.allclose(jp[k], term.motion.sample(torch.tensor([k + 1], device=term.device))["joint_pos"][0]), ('idle frame', k)
    k_zero = int(np.argmax((np.array(cmds) == 0) & (np.arange(len(cmds)) > k_start)))
    k_stop = k_zero + int(np.argmax(~walking[k_zero:]))
    up = term.motion.upper_ids
    for k in range(k_start, k_stop + 1):
        exp = term.motion.sample(torch.tensor([rows[k][2]], device=term.device))["joint_pos"][0]
        assert torch.allclose(jp[k, up], exp[up]), ('neck and arms on the Idle motion while walking', k)
    zero_cmd = torch.zeros((1, 3), device=term.device)
    contact = term.library.sample(zero_cmd, torch.tensor([rows[k_stop][3]], device=term.device))["contact"][0]
    print(f'[verify] reference: walk starts at {times[k_start]:.2f} s with the command; command zero at {times[k_zero]:.2f} s, '
          f'back to Idle at {times[k_stop]:.2f} s (double support {bool((contact > 0.5).all())}); neck and arms on the Idle '
          f'motion throughout')
    assert (k_stop - k_zero) * dt <= term.library.cycle_s / 2 + dt and bool((contact > 0.5).all())
    k_idle = k_stop + int(math.ceil(BLEND_S / dt)) + 1
    for k in range(k_idle, len(rows)):
        exp = term.motion.sample(torch.tensor([rows[k][2]], device=term.device))["joint_pos"][0]
        assert torch.allclose(jp[k], exp, atol=1e-6), ('idle after blend', k)
    step = (jp[1:] - jp[:-1]).abs().amax(-1).cpu().numpy()
    blend_in = step[k_start:k_start + int(BLEND_S / dt) + 2].max(); blend_out = step[k_stop - 1:k_stop + int(BLEND_S / dt) + 2].max()
    steady = step[k_start + int(1.5 / dt):k_zero - int(1.0 / dt)].max(); stand = step[:k_start - 1].max()
    print(f'[verify] largest per-step reference change: Idle {np.degrees(stand):.2f} deg, steady walk {np.degrees(steady):.2f}, '
          f'blend in {np.degrees(blend_in):.2f}, blend out {np.degrees(blend_out):.2f}')
    assert max(blend_in, blend_out) <= max(steady, stand) + 1e-6


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.events.push_robot = None
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        env.reset(); robot = env.scene['robot']; term = env.command_manager.get_term('gait')
        reference_alone(term, env.step_dt)
        walking = term.reference.walking.clone()
        err = (robot.data.joint_pos - term.ref['joint_pos']).abs().amax(-1)
        print(f'[verify] reset: {int((~walking).sum())} standing starts, {int(walking.sum())} walking starts, joint error to '
              f'the reference max {err.max():.3f} rad (noise {term.cfg.reset_joint_noise})')
        assert (~walking).any() and walking.any() and err.max() <= term.cfg.reset_joint_noise + 1e-4
        lim = robot.data.soft_joint_pos_limits[0]; arms = term.arm_ids; margin = math.radians(term.cfg.arm_limit_margin_deg)
        dflt = robot.data.default_joint_pos[0]
        inside = torch.min(dflt[arms] - lim[arms, 0], lim[arms, 1] - dflt[arms])
        print(f'[verify] action offset: arms {np.degrees(inside.min().item()):.2f} deg inside their limits at least (target 3 '
              f'deg, less the 0.57 deg calibration randomization)')
        assert inside.min() > margin - 0.011
        action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
        stand = ~walking; up = robot.find_joints(UPPER_JOINTS, preserve_order=True)[0]
        for _ in range(int(round(0.3 / env.step_dt))):
            env.step(action)
        rms = (robot.data.joint_pos - term.ref['joint_pos'])[:, up].square().mean(-1).sqrt()[stand]
        print(f'[verify] zero action 0.3 s from standing starts: arm and neck joints off the reference by '
              f'{np.degrees(rms.median().item()):.1f} deg rms (median)')
        assert rms.median() < math.radians(15)
        was, switches, arm_ok, seen = term.reference.walking.clone(), [0, 0], True, [False, False]
        for _ in range(int(round(10.0 / env.step_dt))):
            obs, *_ = env.step(action)
            now = term.reference.walking
            switches[0] += int((now & ~was).sum()); switches[1] += int((~now & was).sum()); was = now.clone()
            seen = [seen[0] or bool((~now).any()), seen[1] or bool(now.any())]
            a = term.ref['joint_pos'][:, arms]; l = robot.data.soft_joint_pos_limits[:, arms]
            arm_ok &= bool(((a >= l[..., 0] + margin - 1e-5) & (a <= l[..., 1] - margin + 1e-5)).all())
            assert all(torch.isfinite(v).all() for v in obs.values())
        print(f'[verify] 10 s of commands: {switches[0]} switches to walking, {switches[1]} to standing; arm references '
              f'inside the margin {arm_ok}')
        assert switches[0] > 0 and switches[1] > 0 and all(seen) and arm_ok
        head = robot.actuators['head']; h2 = robot.body_names.index('Head_2')
        np.testing.assert_allclose(head.stiffness.cpu().numpy(), v2.NECK_KP); np.testing.assert_allclose(head.damping.cpu().numpy(), v2.NECK_KD)
        mass = robot.root_physx_view.get_masses()[:, h2].numpy(); lo, hi = v1.HEAD_MASS_KG
        assert mass.min() >= lo - 1e-6 and mass.max() <= hi + 1e-6, (mass.min(), mass.max())
        print(f'[verify] neck kp {v2.NECK_KP} kd {v2.NECK_KD}, Head_2 {mass.min():.2f}..{mass.max():.2f} kg  OK')
        rm = env.reward_manager
        print('[verify] rewards', {n: rm.get_term_cfg(n).weight for n in rm.active_terms})
        assert task.ObservationsCfg.PolicyCfg().joint_pos.history_length == 5
        print('[verify] observation dims', {k: v.shape for k, v in env.observation_manager.compute().items()})
    finally:
        env.close()


if __name__ == '__main__':
    main(); app.close()
