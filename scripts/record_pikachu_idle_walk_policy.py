"""Record an unchanged trained policy in Isaac Lab, retaining resets and telemetry."""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seconds", type=float, default=30.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app

import hashlib
import json
import subprocess
import sys
import traceback

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
import booster_train.tasks  # noqa: F401
from booster_train.tasks.manager_based.locomotion.robots.k1.pikachu_idle_walk import mdp


def encoder(path):
    return subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720",
        "-r", "25", "-i", "-", "-an", "-c:v", "libx264",
        "-threads", "2", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)


def main():
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    task = "Booster-K1-Pikachu-IdleWalk-Scale0842-v0-Play"
    cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    cfg.seed = 842
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.eye = (1.5, -1.7, .70)
    cfg.viewer.lookat = (0., 0., -.02)
    cfg.viewer.resolution = (1280, 720)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args.device
    agent_cfg.seed = cfg.seed
    env = gym.make(task, cfg=cfg, render_mode="rgb_array")
    raw_env = env.unwrapped
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    writers = []
    try:
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=args.device)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=args.device)
        obs, _ = wrapped.reset()
        robot = raw_env.scene["robot"]
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
        def render_frame():
            # Explicit render: this installed Lab version's recompute=True
            # reads the RGB annotator without rendering another frame.
            root = robot.data.root_pos_w[0].cpu().numpy()
            raw_env.sim.set_camera_view(root + np.array(cfg.viewer.eye), root + np.array(cfg.viewer.lookat))
            raw_env.sim.render()
            return raw_env.render(recompute=True)
        for _ in range(24):
            pixels = render_frame()
        assert pixels.shape == (720, 1280, 3) and pixels.std() > 2, (pixels.shape, pixels.std())
        raw_writer = encoder(out / "isaac_raw.mp4")
        review_writer = encoder(out / "isaac_idle_walk_final.mp4")
        writers = [raw_writer, review_writer]
        rows, resets = [], []
        steps = round(args.seconds / raw_env.step_dt)
        assert steps % 2 == 0 and abs(raw_env.step_dt - .02) < 1e-8
        episode = 0
        last_reset_time = -100.
        last_reset_reason = ""
        episode_start = 0.
        with (out / "telemetry.jsonl").open("w") as telemetry:
            for step in range(steps):
                with torch.inference_mode():
                    t = step * raw_env.step_dt
                    command = raw_env.command_manager.get_command("velocity")[0]
                    velocity = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w)[0]
                    reference = mdp.idle(raw_env)
                    upper_error = robot.data.joint_pos[:, reference.upper_ids] - reference.sampled["joint_pos"][:, reference.upper_ids]
                    row = {
                        "time_s": t, "episode": episode,
                        "episode_time_s": float(raw_env.episode_length_buf[0]) * raw_env.step_dt,
                        "command_mps_radps": command.cpu().tolist(),
                        "velocity_heading_mps": velocity.cpu().tolist(),
                        "root_position_m": robot.data.root_pos_w[0].cpu().tolist(),
                        "height_error_m": float(mdp.height_error(raw_env)[0]),
                        "upper_joint_rmse_rad": float(upper_error.square().mean().sqrt()),
                        "knees_deg": mdp.knees(raw_env)[0].rad2deg().cpu().tolist(),
                        "idle_phase": float(reference.sampled["phase"][0]),
                    }
                    rows.append(row)
                    if step % 2 == 0:
                        pixels = render_frame()[:, :, :3]
                        raw_writer.stdin.write(pixels.tobytes())
                        frame = Image.fromarray(pixels)
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 1280, 48), fill=(22, 27, 35))
                        draw.text((18, 11), "Isaac Lab | final policy: model_5999 | 1x speed | camera follows robot", font=font, fill="white")
                        draw.rectangle((0, 626, 1280, 720), fill=(22, 27, 35))
                        draw.text((18, 635), f"t={t:05.2f}s   episode={episode+1}   resets={len(resets)}   forward command={command[0]:+.3f} m/s", font=font, fill="white")
                        draw.text((18, 666), f"Actual forward={velocity[0]:+.3f} m/s   height error={row['height_error_m']*1000:+.0f} mm   upper joint RMSE={np.degrees(row['upper_joint_rmse_rad']):.1f} deg", font=font_small, fill=(225, 231, 239))
                        if t-last_reset_time < 1.:
                            draw.rectangle((15, 65, 1040, 108), fill=(137, 39, 30))
                            draw.text((26, 75), "AUTO RESET: " + last_reset_reason, font=font, fill="white")
                        review_writer.stdin.write(np.asarray(frame).tobytes())
                        if step in (0, 250, 750, 1250):
                            frame.save(out / f"frame_{t:05.2f}s.png")
                    actions = policy(obs)
                    assert torch.isfinite(actions).all()
                    obs, reward, dones, extras = wrapped.step(actions)
                    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                    reasons = []
                    if dones[0]:
                        reasons = [name for name in raw_env.termination_manager.active_terms
                                   if raw_env.termination_manager.get_term(name)[0]]
                        last_reset_time = (step+1)*raw_env.step_dt
                        last_reset_reason = ", ".join(reasons)
                        resets.append({"time_s": last_reset_time, "episode": episode,
                                       "duration_s": last_reset_time-episode_start, "reasons": reasons})
                        episode_start = last_reset_time
                        episode += 1
                    row["reset_after_step"] = bool(dones[0])
                    row["termination_reasons"] = reasons
                    telemetry.write(json.dumps(row)+"\n")
                    if (step+1) % 250 == 0:
                        telemetry.flush()
                        print(f"CAPTURE {step+1}/{steps}; sim time={(step+1)*raw_env.step_dt:.1f}s; resets={len(resets)}", flush=True)
        for writer in writers:
            writer.stdin.close()
            assert writer.wait(timeout=45) == 0
        writers = []
        moving = [r for r in rows if abs(r["command_mps_radps"][0]) > .04]
        result = {
            "status": "CAPTURE_COMPLETE", "task": task,
            "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "policy": "deterministic inference (no action sampling)",
            "seed": cfg.seed, "num_envs": 1, "duration_s": args.seconds,
            "control_dt_s": raw_env.step_dt, "physics_dt_s": raw_env.physics_dt, "video_fps": 25,
            "mass_scenario": cfg.scene.robot.spawn.mass_scenario,
            "partition_construction_z_m": .4,
            "play_overrides": ["no observation noise", "no startup material randomization", "65 s episode limit", "forward target +0.05 m/s; lateral/yaw zero"],
            "resets": resets, "last_episode_duration_s": args.seconds-episode_start,
            "sample_mean_abs_height_error_m": float(np.mean([abs(r["height_error_m"]) for r in rows])),
            "sample_mean_upper_joint_rmse_deg": float(np.mean([np.degrees(r["upper_joint_rmse_rad"]) for r in rows])),
            "moving_sample_count": len(moving),
            "moving_mean_velocity_xy_error_mps": float(np.mean([np.linalg.norm(np.array(r["velocity_heading_mps"][:2])-r["command_mps_radps"][:2]) for r in moving])) if moving else None,
            "limitations": ["Single rollout is not a statistical evaluation.", "Telemetry is sampled before actions; termination rows identify subsequent auto resets.", "Costume head/arms/foam are not visible; their estimated masses are included in the training asset.", "No learned behavior or termination thresholds were changed for this video."],
        }
        (out / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        for writer in writers:
            if writer.stdin and not writer.stdin.closed:
                writer.stdin.close()
            try:
                writer.wait(timeout=10)
            except subprocess.TimeoutExpired:
                writer.terminate()
                writer.wait(timeout=10)
        wrapped.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.close()
