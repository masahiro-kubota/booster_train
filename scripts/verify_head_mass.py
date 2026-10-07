"""Check in PhysX that a head-mass task loaded the inertials of its mass file (links.<name>.combined).

    env -u PYTHONPATH <isaaclab python> scripts/verify_head_mass.py --task Booster-K1-Pikachu_Idle-Scale0842-HeadR27-v0-Play --headless
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); app = AppLauncher(args).app

import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation
import booster_train.tasks  # noqa: F401  (registers the tasks)
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        env.reset(); robot = env.scene['robot']
        links = json.loads(Path(cfg.scene.robot.spawn.mass_file).read_text())['links']
        inertias = robot.root_physx_view.get_inertias().cpu().numpy()[0].reshape(-1, 3, 3)
        for name, row in links.items():
            i = robot.body_names.index(name); wanted = row['combined']
            m = float(robot.data.default_mass[0, i]); c = robot.data.body_com_pos_b[0, i].cpu().numpy()
            q = robot.data.body_com_quat_b[0, i].cpu().numpy(); rot = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            np.testing.assert_allclose(m, wanted['mass_kg'], rtol=2e-6)
            np.testing.assert_allclose(c, wanted['com_m'], atol=2e-7)
            np.testing.assert_allclose(rot @ np.diag(wanted['principal_inertia_kg_m2']) @ rot.T, wanted['inertia_kg_m2'], rtol=1e-4, atol=2e-7)
            np.testing.assert_allclose(inertias[i], wanted['inertia_kg_m2'], rtol=1e-4, atol=2e-7)
            print(f'[verify] {name}: mass {m:.4f} kg, com {np.round(c, 4).tolist()} m, inertia diag {np.round(np.diag(inertias[i]), 5).tolist()}  OK')
        print(f'[verify] robot total mass {robot.data.default_mass.sum().item():.4f} kg')
    finally:
        env.close()


if __name__ == '__main__':
    main(); app.close()
