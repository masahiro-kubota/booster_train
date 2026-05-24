#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

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
    DiffusionSchedule,
    StateLatentDenoiser,
    TrajectoryWindowDataset,
    build_state_projection,
    load_normalizer,
    paper_state_layout,
    root_emphasis_indices,
    save_state_projection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BeyondMimic state-latent diffusion model.")
    parser.add_argument("--shards", nargs="+", required=True, help="Input rollout .npz shards.")
    parser.add_argument("--vae-checkpoint", required=True, help="Path to vae.pt from train_vae.py.")
    parser.add_argument("--normalizer", required=True, help="Path to normalizer.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for diffusion checkpoints.")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--denoising-steps", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--current-index", type=int, default=None)
    parser.add_argument("--num-bodies", type=int, default=23)
    parser.add_argument("--projection-random-dim", type=int, default=0)
    parser.add_argument("--projection-root-weight", type=float, default=6.0)
    parser.add_argument("--projection-seed", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def encode_actions(
    vae: ConditionalActionVAE,
    state_window: torch.Tensor,
    action_window: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Encode each state/action pair into a latent intent."""

    batch, length, _ = state_window.shape
    flat_state = state_window.reshape(batch * length, -1).to(device)
    flat_action = action_window.reshape(batch * length, -1).to(device)
    flat_action = (flat_action - action_mean) / action_std
    mu, _ = vae.encode(flat_state, flat_action)
    return mu.reshape(batch, length, -1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vae_checkpoint = torch.load(args.vae_checkpoint, map_location=args.device)
    raw_state_dim = int(vae_checkpoint["state_dim"])
    action_dim = int(vae_checkpoint["action_dim"])
    latent_dim = int(vae_checkpoint["latent_dim"])
    vae = ConditionalActionVAE(state_dim=raw_state_dim, action_dim=action_dim, latent_dim=latent_dim).to(args.device)
    vae.load_state_dict(vae_checkpoint["state_dict"])
    vae.eval()

    normalizer = load_normalizer(args.normalizer)
    action_mean = torch.as_tensor(normalizer["action_mean"], dtype=torch.float32, device=args.device)
    action_std = torch.as_tensor(normalizer["action_std"], dtype=torch.float32, device=args.device)
    current_index = args.history_length if args.current_index is None else int(args.current_index)
    window_length = args.history_length + 1 + args.horizon
    if current_index != args.history_length:
        raise ValueError("current_index must equal history_length for the paper-style history/current/future window")
    dataset = TrajectoryWindowDataset(
        args.shards,
        window_length=window_length,
        state_mean=normalizer["state_mean"],
        state_std=normalizer["state_std"],
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    state_layout = paper_state_layout(args.num_bodies, action_dim)
    if state_layout["last_action"][1] != raw_state_dim:
        raise ValueError(
            "VAE state_dim does not match paper_state_layout. "
            f"state_dim={raw_state_dim}, expected={state_layout['last_action'][1]} "
            f"for num_bodies={args.num_bodies}, action_dim={action_dim}"
        )
    state_projection = build_state_projection(
        raw_state_dim,
        root_emphasis_indices(state_layout),
        random_dim=args.projection_random_dim,
        root_weight=args.projection_root_weight,
        seed=args.projection_seed,
    )
    save_state_projection(state_projection, str(output_dir / "state_projection.pt"))
    projection = state_projection.projection.to(args.device)
    projected_state_dim = int(state_projection.projected_state_dim)

    input_dim = projected_state_dim + latent_dim
    denoiser = StateLatentDenoiser(
        input_dim=input_dim,
        horizon_length=window_length,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(args.device)
    schedule = DiffusionSchedule(num_steps=args.denoising_steps).to(args.device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    ema_state_dict = {name: value.detach().clone() for name, value in denoiser.state_dict().items()}

    history = []
    for epoch in range(args.epochs):
        denoiser.train()
        total_loss = 0.0
        total_state_loss = 0.0
        total_latent_loss = 0.0
        total_count = 0
        for batch in dataloader:
            state_window = batch["state"].to(args.device)
            action_window = batch["action"].to(args.device)
            latent_window = encode_actions(vae, state_window, action_window, action_mean, action_std, args.device)
            projected_state_window = torch.matmul(state_window, projection.t())
            clean_state = projected_state_window
            clean_latent = latent_window
            state_steps = torch.randint(0, schedule.num_steps, clean_state.shape[:2], device=args.device)
            latent_steps = torch.randint(0, schedule.num_steps, clean_latent.shape[:2], device=args.device)
            noisy_state = schedule.q_sample(clean_state, state_steps)
            noisy_latent = schedule.q_sample(clean_latent, latent_steps)
            clean = torch.cat((clean_state, clean_latent), dim=-1)
            noisy = torch.cat((noisy_state, noisy_latent), dim=-1)
            steps = torch.stack((state_steps, latent_steps), dim=-1)
            pred = denoiser(noisy, steps)
            state_loss = torch.mean((pred[..., :projected_state_dim] - clean_state) ** 2)
            latent_loss = torch.mean((pred[..., projected_state_dim:] - clean_latent) ** 2)
            loss = float(args.state_loss_weight) * state_loss + float(args.latent_loss_weight) * latent_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if args.ema_decay > 0.0:
                decay = float(args.ema_decay)
                with torch.no_grad():
                    current_state_dict = denoiser.state_dict()
                    for name, value in ema_state_dict.items():
                        if torch.is_floating_point(value):
                            value.mul_(decay).add_(current_state_dict[name].detach(), alpha=1.0 - decay)
                        else:
                            value.copy_(current_state_dict[name])
            total_loss += float(loss.detach().cpu()) * clean.shape[0]
            total_state_loss += float(state_loss.detach().cpu()) * clean.shape[0]
            total_latent_loss += float(latent_loss.detach().cpu()) * clean.shape[0]
            total_count += clean.shape[0]

        metrics = {
            "epoch": epoch + 1,
            "loss": total_loss / max(total_count, 1),
            "state_loss": total_state_loss / max(total_count, 1),
            "latent_loss": total_latent_loss / max(total_count, 1),
        }
        history.append(metrics)
        print(json.dumps(metrics), flush=True)

    checkpoint = {
        "state_dict": denoiser.state_dict(),
        "ema_state_dict": ema_state_dict,
        "raw_state_dim": raw_state_dim,
        "state_dim": projected_state_dim,
        "action_dim": action_dim,
        "latent_dim": latent_dim,
        "input_dim": input_dim,
        "history_length": args.history_length,
        "current_index": current_index,
        "horizon": args.horizon,
        "window_length": window_length,
        "denoising_steps": args.denoising_steps,
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "state_layout": state_layout,
        "num_bodies": int(args.num_bodies),
        "projection": {
            "raw_state_dim": state_projection.raw_state_dim,
            "projected_state_dim": state_projection.projected_state_dim,
            "root_indices": state_projection.root_indices,
            "random_dim": state_projection.random_dim,
            "root_weight": state_projection.root_weight,
            "seed": state_projection.seed,
        },
        "history": history,
    }
    torch.save(checkpoint, output_dir / "diffusion.pt")

    denoiser.eval()
    example_trajectory = torch.zeros(1, window_length, input_dim, device=args.device)
    example_steps = torch.zeros(1, window_length, 2, dtype=torch.long, device=args.device)
    traced = torch.jit.trace(denoiser, (example_trajectory, example_steps))
    traced.save(str(output_dir / "diffusion_model.pt"))


if __name__ == "__main__":
    main()
