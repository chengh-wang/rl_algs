"""Continuous-time velocity DiT, implemented using only PyTorch.

Architecture adapted from SUZ-tsinghua/smp at commit
28d1dfd1b7b4447977fbdb2062167d9f87a2faaa, src/smp/pretrain/model.py.
See ../smp/vendor/PROVENANCE.md for original source attribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, fields
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FlowModelConfig:
  feature_dim: int
  window_size: int
  d_model: int = 256
  nhead: int = 4
  num_layers: int = 2
  dropout: float = 0.0
  head_dim: int | None = None
  time_embedding_scale: float = 1.0

  def __post_init__(self):
    for name in ("feature_dim", "window_size", "d_model", "nhead", "num_layers"):
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if self.head_dim is not None and (isinstance(self.head_dim, bool) or not isinstance(self.head_dim, int) or self.head_dim <= 0):
      raise ValueError("head_dim must be a positive integer or None")
    if self.d_model % 2 or self.d_model % self.nhead or (self.head_dim is not None and self.nhead*self.head_dim != self.d_model):
      raise ValueError("d_model must be even and equal nhead * head_dim")
    for name in ("dropout", "time_embedding_scale"):
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not 0 <= self.dropout <= 1 or self.time_embedding_scale <= 0:
      raise ValueError("dropout must be in [0,1] and time_embedding_scale positive")

  def to_dict(self):
    return asdict(self)

  @classmethod
  def from_dict(cls, value):
    if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(cls)}:
      raise ValueError("Model configuration must contain exactly all FlowModelConfig fields")
    return cls(**dict(value))


class _Timesteps(nn.Module):
  """Sinusoidal timestep features, ``flip_sin_to_cos`` (cos first, sin second)."""

  def __init__(self, num_channels: int = 256) -> None:
    super().__init__()
    if num_channels % 2 != 0:
      msg = f"_Timesteps requires even num_channels, got {num_channels}"
      raise ValueError(msg)
    self.num_channels = num_channels

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    half = self.num_channels // 2
    exponent = -math.log(10000.0) * torch.arange(
      half, dtype=torch.float32, device=t.device
    )
    exponent = exponent / half
    emb = t.float()[:, None] * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class _TimestepEmbedding(nn.Module):
  def __init__(self, in_channels: int, time_embed_dim: int) -> None:
    super().__init__()
    self.linear_1 = nn.Linear(in_channels, time_embed_dim)
    self.act = nn.SiLU()
    self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear_2(self.act(self.linear_1(x)))


class _AdaLayerNormSingle(nn.Module):
  """PixArt-α adaLN-single: produce (B, 1, 6·D) timestep modulation."""

  def __init__(self, embedding_dim: int) -> None:
    super().__init__()
    self.time_proj = _Timesteps(num_channels=256)
    self.timestep_embedder = _TimestepEmbedding(256, embedding_dim)
    self.silu = nn.SiLU()
    self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    t_emb = self.timestep_embedder(self.time_proj(t)).unsqueeze(1)
    return self.linear(self.silu(t_emb))


class _SinusoidalPositionalEmbedding(nn.Module):
  pe: torch.Tensor

  def __init__(self, embed_dim: int, max_seq_length: int = 32) -> None:
    super().__init__()
    position = torch.arange(max_seq_length).unsqueeze(1)
    div_term = torch.exp(
      torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(1, max_seq_length, embed_dim)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    self.register_buffer("pe", pe, persistent=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.pe[:, : x.shape[1]]


class _SwiGLU(nn.Module):
  def __init__(self, dim: int, inner_dim: int, bias: bool = True) -> None:
    super().__init__()
    self.proj = nn.Linear(dim, inner_dim * 2, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    a, b = self.proj(x).chunk(2, dim=-1)
    return F.silu(a) * b


class _FeedForward(nn.Module):
  def __init__(
    self,
    dim: int,
    mult: int = 4,
    dropout: float = 0.0,
    bias: bool = True,
  ) -> None:
    super().__init__()
    inner_dim = dim * mult
    self.act = _SwiGLU(dim, inner_dim, bias=bias)
    self.dropout = nn.Dropout(dropout)
    self.proj_out = nn.Linear(inner_dim, dim, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.proj_out(self.dropout(self.act(x)))


class _DiTBlock(nn.Module):
  """Self-attention + SwiGLU FFN, both modulated by adaLN-single."""

  def __init__(
    self,
    dim: int,
    num_heads: int,
    head_dim: int,
    dropout: float = 0.0,
    norm_eps: float = 1e-5,
  ) -> None:
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.attn_dim = num_heads * head_dim

    self.norm1 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
    self.to_q = nn.Linear(dim, self.attn_dim, bias=False)
    self.to_k = nn.Linear(dim, self.attn_dim, bias=False)
    self.to_v = nn.Linear(dim, self.attn_dim, bias=False)
    self.to_out = nn.Linear(self.attn_dim, dim, bias=False)
    self.attn_dropout = nn.Dropout(dropout)

    self.norm2 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
    self.ff = _FeedForward(dim, mult=4, dropout=dropout)

    self.scale_shift_table = nn.Parameter(torch.randn(1, 1, 6, dim) / dim**0.5)

  def _attn(self, x: torch.Tensor) -> torch.Tensor:
    B, N, _ = x.shape
    h, d = self.num_heads, self.head_dim
    q = self.to_q(x).reshape(B, N, h, d).transpose(1, 2)
    k = self.to_k(x).reshape(B, N, h, d).transpose(1, 2)
    v = self.to_v(x).reshape(B, N, h, d).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = out.transpose(1, 2).reshape(B, N, h * d)
    return self.attn_dropout(self.to_out(out))

  def forward(self, x: torch.Tensor, time_hidden_states: torch.Tensor) -> torch.Tensor:
    B = x.shape[0]
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
      self.scale_shift_table + time_hidden_states.reshape(B, 1, 6, -1)
    ).chunk(6, dim=-2)

    h = self.norm1(x)
    h = h * (1 + scale_msa.squeeze(-2)) + shift_msa.squeeze(-2)
    x = x + gate_msa.squeeze(-2) * self._attn(h)

    h = self.norm2(x)
    h = h * (1 + scale_mlp.squeeze(-2)) + shift_mlp.squeeze(-2)
    x = x + gate_mlp.squeeze(-2) * self.ff(h)
    return x


class FlowVelocityModel(nn.Module):
  """Predict conditional velocity from path points [B,W,F] and tau [B].

  tau is always continuous in [0,1]. The explicit time_embedding_scale only
  rescales sinusoidal embedding inputs; its default is 1 and is checkpointed.
  """

  def __init__(
    self,
    feature_dim: int,
    window_size: int,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.0,
    head_dim: int | None = None,
    time_embedding_scale: float = 1.0,
  ) -> None:
    super().__init__()
    self.config = FlowModelConfig(feature_dim, window_size, d_model, nhead, num_layers, dropout, head_dim, time_embedding_scale)
    self.feature_dim = feature_dim
    self.window_size = window_size

    if head_dim is None:
      if d_model % nhead != 0:
        msg = (
          f"d_model ({d_model}) must be divisible by nhead ({nhead}) "
          f"when head_dim is unspecified"
        )
        raise ValueError(msg)
      head_dim = d_model // nhead
    self.num_heads = nhead
    self.head_dim = head_dim
    self.inner_dim = nhead * head_dim
    if self.inner_dim != d_model:
      msg = (
        f"d_model ({d_model}) must equal nhead·head_dim "
        f"({nhead}·{head_dim} = {self.inner_dim})"
      )
      raise ValueError(msg)

    self.preprocess_conv = nn.Conv1d(feature_dim, feature_dim, 1, bias=False)
    self.proj_in = nn.Linear(feature_dim, self.inner_dim, bias=False)
    self.adaln_single = _AdaLayerNormSingle(self.inner_dim)
    self.sequence_pos_encoder = _SinusoidalPositionalEmbedding(
      self.inner_dim, max_seq_length=max(window_size, 32)
    )
    self.blocks = nn.ModuleList(
      [
        _DiTBlock(
          dim=self.inner_dim,
          num_heads=nhead,
          head_dim=head_dim,
          dropout=dropout,
        )
        for _ in range(num_layers)
      ]
    )
    self.proj_out = nn.Linear(self.inner_dim, feature_dim, bias=False)
    self.postprocess_conv = nn.Conv1d(feature_dim, feature_dim, 1, bias=False)

  def forward(self, x_t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    if not isinstance(x_t, torch.Tensor) or not x_t.is_floating_point() or x_t.ndim != 3 or x_t.shape[1:] != (self.window_size, self.feature_dim) or x_t.shape[0] == 0 or not torch.isfinite(x_t).all():
      raise ValueError("Expected finite floating nonempty path points [B,W,F]")
    if not isinstance(tau, torch.Tensor) or not tau.is_floating_point() or tau.shape != (len(x_t),) or tau.device != x_t.device or not torch.isfinite(tau).all() or ((tau < 0) | (tau > 1)).any():
      raise ValueError("Expected floating tau [B] on input device in [0,1]")
    h = x_t.transpose(1, 2)
    h = self.preprocess_conv(h) + h
    h = h.transpose(1, 2)

    h = self.proj_in(h)
    time_hidden_states = self.adaln_single(tau * self.config.time_embedding_scale)
    h = self.sequence_pos_encoder(h)

    for block in self.blocks:
      h = block(h, time_hidden_states)

    h = self.proj_out(h)
    h = h.transpose(1, 2)
    h = self.postprocess_conv(h) + h
    return h.transpose(1, 2)
