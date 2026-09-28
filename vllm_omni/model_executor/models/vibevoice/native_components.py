# SPDX-License-Identifier: Apache-2.0
"""Inference-only VibeVoice side modules used by the native AR runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class VibeVoiceBatchedStreamingCache:
    """Slot-indexed GPU cache for batched streaming convolutions.

    The preserved tokenizer cache stores one dictionary entry per
    ``(layer, request)`` and calls ``sample_indices.tolist()`` in every layer.
    This implementation stores one dense tensor per layer and performs
    gather/scatter entirely on device, so requests can join/leave a batch
    without a Python per-request loop on the codec hot path.
    """

    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self.cache: dict[str, torch.Tensor] = {}

    def get(self, layer_id: str, sample_indices: torch.Tensor) -> torch.Tensor | None:
        states = self.cache.get(layer_id)
        if states is None:
            return None
        return states.index_select(0, sample_indices.to(device=states.device, dtype=torch.long))

    def set(
        self,
        layer_id: str,
        sample_indices: torch.Tensor,
        states: torch.Tensor,
    ) -> None:
        detached = states.detach()
        storage = self.cache.get(layer_id)
        if storage is None:
            storage = torch.zeros(
                (self.capacity, *detached.shape[1:]),
                device=detached.device,
                dtype=detached.dtype,
            )
            self.cache[layer_id] = storage
        elif tuple(storage.shape[1:]) != tuple(detached.shape[1:]):
            raise RuntimeError(
                f"VibeVoice codec cache shape changed for {layer_id}: "
                f"{tuple(storage.shape[1:])} -> {tuple(detached.shape[1:])}"
            )
        storage.index_copy_(
            0,
            sample_indices.to(device=storage.device, dtype=torch.long),
            detached,
        )

    def set_to_zero(self, sample_indices: torch.Tensor) -> None:
        for states in self.cache.values():
            indices = sample_indices.to(device=states.device, dtype=torch.long)
            states.index_fill_(0, indices, 0)

    def clear(
        self,
        layer_id: str | None = None,
        sample_indices: torch.Tensor | None = None,
    ) -> None:
        if layer_id is None and sample_indices is None:
            self.cache.clear()
            return
        targets = (
            [self.cache[layer_id]]
            if layer_id is not None and layer_id in self.cache
            else list(self.cache.values())
        )
        if sample_indices is None:
            if layer_id is not None:
                self.cache.pop(layer_id, None)
            return
        for states in targets:
            indices = sample_indices.to(device=states.device, dtype=torch.long)
            states.index_fill_(0, indices, 0)


class VibeVoicePhase(str, Enum):
    PREFILL = "prefill"
    TEXT = "text"
    SPEECH = "speech"
    FINISHED = "finished"


@dataclass
class VibeVoiceRequestState:
    request_id: str
    phase: VibeVoicePhase = VibeVoicePhase.PREFILL
    positive_num_tokens: int = 0
    negative_num_tokens: int = 0
    negative_block_anchor: int = -1
    generated_steps: int = 0
    max_generation_steps: int = 0
    cfg_scale: float = 1.3
    ddpm_steps: int = 10
    rng_seed: int = 0
    rng_offset: int = 0
    noise_pool_position: int = 0
    noise_pool_valid: int = 0
    cache_slot: int = -1
    finished: bool = False
    reached_max_length: bool = False

    def as_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["phase"] = self.phase.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VibeVoiceRequestState":
        values = dict(data)
        values["phase"] = VibeVoicePhase(values.get("phase", VibeVoicePhase.PREFILL))
        return cls(**{key: value for key, value in values.items() if key in cls.__dataclass_fields__})


class VibeVoiceRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states.float()
        output = output * torch.rsqrt(output.square().mean(-1, keepdim=True) + self.eps)
        output = output.to(hidden_states.dtype)
        return output if self.weight is None else output * self.weight


class SpeechConnector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = VibeVoiceRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.norm(self.fc1(features)))


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding.to(timesteps.dtype))


class DiffusionFeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DiffusionHeadLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, eps: float):
        super().__init__()
        self.ffn = DiffusionFeedForward(hidden_size, intermediate_size)
        self.norm = VibeVoiceRMSNorm(hidden_size, eps=eps)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(condition).chunk(3, dim=-1)
        return hidden_states + gate * self.ffn(_modulate(self.norm(hidden_states), shift, scale))


class DiffusionFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, eps: float):
        super().__init__()
        self.norm_final = VibeVoiceRMSNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm_final(hidden_states), shift, scale))


class VibeVoiceDiffusionHead(nn.Module):
    """Numerically compatible inference form of the released diffusion head."""

    def __init__(self, config: Any):
        super().__init__()
        hidden_size = int(config.hidden_size)
        latent_size = int(config.latent_size)
        intermediate_size = int(hidden_size * float(config.head_ffn_ratio))
        self.noisy_images_proj = nn.Linear(latent_size, hidden_size, bias=False)
        self.cond_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.layers = nn.ModuleList(
            [
                DiffusionHeadLayer(hidden_size, intermediate_size, float(config.rms_norm_eps))
                for _ in range(int(config.head_layers))
            ]
        )
        self.final_layer = DiffusionFinalLayer(hidden_size, latent_size, float(config.rms_norm_eps))

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.noisy_images_proj(noisy_images)
        conditioning = self.cond_proj(condition) + self.t_embedder(timesteps)
        for layer in self.layers:
            hidden_states = layer(hidden_states, conditioning)
        return self.final_layer(hidden_states, conditioning)


__all__ = [
    "SpeechConnector",
    "VibeVoiceBatchedStreamingCache",
    "VibeVoiceDiffusionHead",
    "VibeVoicePhase",
    "VibeVoiceRequestState",
]
