from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: type[nn.Module] = nn.ELU) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class ConditionalActionVAE(nn.Module):
    """Conditional VAE that reconstructs policy actions from state and latent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = [2048, 1024, 512] if hidden_dims is None else hidden_dims
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = _mlp(self.state_dim + self.action_dim, hidden_dims, self.latent_dim * 2)
        self.decoder = _mlp(self.state_dim + self.latent_dim, hidden_dims, self.action_dim)

    def encode(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.encoder(torch.cat((state, action), dim=-1))
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, torch.clamp(logvar, min=-10.0, max=10.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat((state, latent), dim=-1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(state, action)
        latent = self.reparameterize(mu, logvar)
        return self.decode(state, latent), mu, logvar


class VAEDecoder(nn.Module):
    """TorchScript-friendly decoder wrapper."""

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, state: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat((state, latent), dim=-1))


class SinusoidalStepEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        device = steps.device
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -scale)
        args = steps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
        if emb.shape[-1] < self.embedding_dim:
            emb = F.pad(emb, (0, self.embedding_dim - emb.shape[-1]))
        return emb


class StateLatentDenoiser(nn.Module):
    """Transformer denoiser that predicts clean state-latent trajectories."""

    def __init__(
        self,
        input_dim: int,
        horizon_length: int,
        embedding_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        feedforward_dim: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.horizon_length = int(horizon_length)
        self.input_proj = nn.Linear(self.input_dim, embedding_dim)
        self.step_embed = SinusoidalStepEmbedding(embedding_dim)
        self.component_step_proj = nn.Linear(embedding_dim * 2, embedding_dim)
        self.pos_embed = nn.Parameter(torch.zeros(self.horizon_length, 1, embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embedding_dim, self.input_dim)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def _step_embedding(self, steps: torch.Tensor, batch: int, length: int) -> torch.Tensor:
        if steps.dim() == 1:
            return self.step_embed(steps).view(1, batch, -1).expand(length, batch, -1)
        if steps.dim() == 2:
            if steps.shape != (batch, length):
                raise RuntimeError("2D steps must have shape [B, L]")
            return self.step_embed(steps.reshape(-1)).view(batch, length, -1).transpose(0, 1)
        if steps.dim() == 3:
            if steps.shape[:2] != (batch, length) or steps.shape[-1] != 2:
                raise RuntimeError("3D steps must have shape [B, L, 2]")
            state_step = self.step_embed(steps[..., 0].reshape(-1))
            latent_step = self.step_embed(steps[..., 1].reshape(-1))
            merged = self.component_step_proj(torch.cat((state_step, latent_step), dim=-1))
            return merged.view(batch, length, -1).transpose(0, 1)
        raise RuntimeError("steps must have shape [B], [B, L], or [B, L, 2]")

    def forward(self, trajectory: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        # trajectory: [batch, horizon_length, input_dim]
        if trajectory.dim() != 3:
            raise RuntimeError("trajectory must have shape [B, L, D]")
        batch, length, _ = trajectory.shape
        if length > self.horizon_length:
            raise RuntimeError("trajectory length exceeds configured horizon_length")
        x = self.input_proj(trajectory).transpose(0, 1)
        step_embedding = self._step_embedding(steps, batch, length)
        x = x + self.pos_embed[:length] + step_embedding
        x = self.transformer(x)
        return self.output_proj(x.transpose(0, 1))


class DiffusionSchedule(nn.Module):
    """DDPM schedule helper for x0-prediction training and sampling."""

    def __init__(self, num_steps: int = 20, beta_start: float = 1.0e-4, beta_end: float = 0.02) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat((torch.ones(1, dtype=torch.float32), alpha_bars[:-1]), dim=0)

        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars)
        posterior_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("posterior_variance", torch.clamp(posterior_var, min=1.0e-20))
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    @property
    def num_steps(self) -> int:
        return int(self.betas.numel())

    def q_sample(self, clean: torch.Tensor, steps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        noise = torch.randn_like(clean) if noise is None else noise
        shape = steps.shape + (1,) * (clean.dim() - steps.dim())
        sqrt_ab = self.sqrt_alpha_bars[steps].view(shape)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[steps].view(shape)
        return sqrt_ab * clean + sqrt_omab * noise

    def p_mean_variance(self, noisy: torch.Tensor, clean_pred: torch.Tensor, steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = steps.shape + (1,) * (noisy.dim() - steps.dim())
        coef1 = self.posterior_mean_coef1[steps].view(shape)
        coef2 = self.posterior_mean_coef2[steps].view(shape)
        variance = self.posterior_variance[steps].view(shape)
        return coef1 * clean_pred + coef2 * noisy, variance
