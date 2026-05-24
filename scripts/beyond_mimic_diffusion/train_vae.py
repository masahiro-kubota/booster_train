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
    RolloutDataset,
    VAEDecoder,
    compute_normalizer,
    load_normalizer,
    save_normalizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BeyondMimic conditional action VAE.")
    parser.add_argument("--shards", nargs="+", required=True, help="Input rollout .npz shards.")
    parser.add_argument("--output-dir", required=True, help="Directory for VAE checkpoints and normalizer.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizer_path = output_dir / "normalizer.json"
    normalizer = compute_normalizer(args.shards)
    save_normalizer(normalizer, normalizer_path)
    loaded_normalizer = load_normalizer(normalizer_path)

    dataset = RolloutDataset(
        args.shards,
        state_mean=loaded_normalizer["state_mean"],
        state_std=loaded_normalizer["state_std"],
        action_mean=loaded_normalizer["action_mean"],
        action_std=loaded_normalizer["action_std"],
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    state_dim = int(dataset.states.shape[1])
    action_dim = int(dataset.actions.shape[1])
    model = ConditionalActionVAE(state_dim=state_dim, action_dim=action_dim, latent_dim=args.latent_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_count = 0
        for batch in dataloader:
            state = batch["state"].to(args.device)
            action = batch["action"].to(args.device)
            recon, mu, logvar = model(state, action)
            recon_loss = torch.mean((recon - action) ** 2)
            kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + args.kl_weight * kl_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            count = state.shape[0]
            total_loss += float(loss.detach().cpu()) * count
            total_recon += float(recon_loss.detach().cpu()) * count
            total_kl += float(kl_loss.detach().cpu()) * count
            total_count += count

        metrics = {
            "epoch": epoch + 1,
            "loss": total_loss / max(total_count, 1),
            "reconstruction_loss": total_recon / max(total_count, 1),
            "kl_loss": total_kl / max(total_count, 1),
        }
        history.append(metrics)
        print(json.dumps(metrics), flush=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "latent_dim": args.latent_dim,
        "normalizer": normalizer,
        "history": history,
    }
    torch.save(checkpoint, output_dir / "vae.pt")

    model.eval()
    decoder = VAEDecoder(model.decoder).to(args.device).eval()
    example_state = torch.zeros(1, state_dim, device=args.device)
    example_latent = torch.zeros(1, args.latent_dim, device=args.device)
    traced_decoder = torch.jit.trace(decoder, (example_state, example_latent))
    traced_decoder.save(str(output_dir / "vae_decoder.pt"))


if __name__ == "__main__":
    main()
