#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import torch

sys.path.append(
    str(
        Path(__file__).resolve().parents[2]
        / "source"
        / "booster_train"
        / "booster_train"
        / "tasks"
        / "manager_based"
        / "beyond_mimic"
    )
)

from diffusion import (  # noqa: E402
    ConditionalActionVAE,
    DatasetManifest,
    StateLatentDenoiser,
    VAEDecoder,
    build_state_projection,
    paper_state_layout,
    root_emphasis_indices,
    save_state_projection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BeyondMimic diffusion artifacts for booster_deploy.")
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-joint-names", nargs="+", required=True)
    parser.add_argument("--policy-dt", type=float, default=0.04)
    parser.add_argument("--num-bodies", type=int, default=23)
    parser.add_argument("--projection-random-dim", type=int, default=None)
    parser.add_argument("--projection-root-weight", type=float, default=None)
    parser.add_argument("--projection-seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vae_checkpoint = torch.load(args.vae_checkpoint, map_location=args.device)
    diffusion_checkpoint = torch.load(args.diffusion_checkpoint, map_location=args.device)
    raw_state_dim = int(vae_checkpoint["state_dim"])
    action_dim = int(vae_checkpoint["action_dim"])
    latent_dim = int(vae_checkpoint["latent_dim"])
    if len(args.action_joint_names) != action_dim:
        raise ValueError("--action-joint-names length must match VAE action_dim")

    vae = ConditionalActionVAE(state_dim=raw_state_dim, action_dim=action_dim, latent_dim=latent_dim).to(args.device)
    vae.load_state_dict(vae_checkpoint["state_dict"])
    vae.eval()
    decoder = VAEDecoder(vae.decoder).to(args.device).eval()
    traced_decoder = torch.jit.trace(
        decoder,
        (
            torch.zeros(1, raw_state_dim, device=args.device),
            torch.zeros(1, latent_dim, device=args.device),
        ),
    )
    traced_decoder.save(str(output_dir / "vae_decoder.pt"))

    denoiser = StateLatentDenoiser(
        input_dim=int(diffusion_checkpoint["input_dim"]),
        horizon_length=int(diffusion_checkpoint["window_length"]),
        embedding_dim=int(diffusion_checkpoint["embedding_dim"]),
        num_heads=int(diffusion_checkpoint["num_heads"]),
        num_layers=int(diffusion_checkpoint["num_layers"]),
    ).to(args.device)
    denoiser.load_state_dict(diffusion_checkpoint.get("ema_state_dict", diffusion_checkpoint["state_dict"]))
    denoiser.eval()
    traced_diffusion = torch.jit.trace(
        denoiser,
        (
            torch.zeros(1, int(diffusion_checkpoint["window_length"]), int(diffusion_checkpoint["input_dim"]), device=args.device),
            torch.zeros(1, int(diffusion_checkpoint["window_length"]), 2, dtype=torch.long, device=args.device),
        ),
    )
    traced_diffusion.save(str(output_dir / "diffusion.pt"))
    shutil.copyfile(args.normalizer, output_dir / "normalizer.json")

    state_layout = diffusion_checkpoint.get("state_layout") or paper_state_layout(args.num_bodies, action_dim)
    projection_info = diffusion_checkpoint.get("projection", {})
    projection_random_dim = (
        int(projection_info.get("random_dim", 0))
        if args.projection_random_dim is None
        else int(args.projection_random_dim)
    )
    projection_root_weight = (
        float(projection_info.get("root_weight", 6.0))
        if args.projection_root_weight is None
        else float(args.projection_root_weight)
    )
    projection_seed = (
        int(projection_info.get("seed", 1))
        if args.projection_seed is None
        else int(args.projection_seed)
    )
    state_projection = build_state_projection(
        raw_state_dim,
        list(projection_info.get("root_indices", root_emphasis_indices(state_layout))),
        random_dim=projection_random_dim,
        root_weight=projection_root_weight,
        seed=projection_seed,
    )
    if int(diffusion_checkpoint["state_dim"]) != state_projection.projected_state_dim:
        raise ValueError(
            "Diffusion checkpoint projected state_dim does not match rebuilt projection. "
            f"checkpoint={diffusion_checkpoint['state_dim']}, projection={state_projection.projected_state_dim}"
        )
    save_state_projection(state_projection, str(output_dir / "state_projection.pt"))

    manifest = DatasetManifest(
        version=2,
        state_dim=int(diffusion_checkpoint["state_dim"]),
        action_dim=action_dim,
        action_joint_names=list(args.action_joint_names),
        state_layout=state_layout,
        policy_dt=args.policy_dt,
        source="beyond_mimic_diffusion_export",
    )
    manifest_data = {
        **manifest.__dict__,
        "raw_state_dim": raw_state_dim,
        "projected_state_dim": int(diffusion_checkpoint["state_dim"]),
        "latent_dim": latent_dim,
        "history_length": int(diffusion_checkpoint["history_length"]),
        "current_index": int(diffusion_checkpoint.get("current_index", diffusion_checkpoint["history_length"])),
        "horizon": int(diffusion_checkpoint["horizon"]),
        "window_length": int(diffusion_checkpoint["window_length"]),
        "denoising_steps": int(diffusion_checkpoint["denoising_steps"]),
        "num_bodies": int(diffusion_checkpoint.get("num_bodies", args.num_bodies)),
        "projection_artifact": "state_projection.pt",
        "body_names": diffusion_checkpoint.get("body_names", []),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
