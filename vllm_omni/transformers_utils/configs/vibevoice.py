# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hugging Face configuration for the original VibeVoice TTS checkpoints."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig, Qwen2Config


class VibeVoiceAcousticTokenizerConfig(PretrainedConfig):
    model_type = "vibevoice_acoustic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0.5,
        std_dist_type: str = "gaussian",
        mixer_layer: str = "depthwise_conv",
        conv_norm: str = "none",
        pad_mode: str = "constant",
        disable_last_norm: bool = True,
        layernorm: str = "RMSNorm",
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        encoder_n_filters: int = 32,
        encoder_ratios: list[int] | None = None,
        encoder_depths: str = "3-3-3-3-3-3-8",
        decoder_n_filters: int = 32,
        decoder_ratios: list[int] | None = None,
        decoder_depths: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type
        self.mixer_layer = mixer_layer
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_eps = layernorm_eps
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios or [8, 5, 5, 4, 2, 2]
        self.encoder_depths = encoder_depths
        self.decoder_n_filters = decoder_n_filters
        self.decoder_ratios = decoder_ratios or list(self.encoder_ratios)
        self.decoder_depths = decoder_depths


class VibeVoiceSemanticTokenizerConfig(VibeVoiceAcousticTokenizerConfig):
    model_type = "vibevoice_semantic_tokenizer"

    def __init__(self, vae_dim: int = 128, fix_std: float = 0.0, std_dist_type: str = "none", **kwargs: Any):
        # The semantic tokenizer has no decoder in the official checkpoint.
        kwargs.pop("decoder_n_filters", None)
        kwargs.pop("decoder_ratios", None)
        kwargs.pop("decoder_depths", None)
        super().__init__(vae_dim=vae_dim, fix_std=fix_std, std_dist_type=std_dist_type, **kwargs)


class VibeVoiceDiffusionHeadConfig(PretrainedConfig):
    model_type = "vibevoice_diffusion_head"

    def __init__(
        self,
        hidden_size: int = 1536,
        head_layers: int = 4,
        head_ffn_ratio: float = 3.0,
        rms_norm_eps: float = 1e-5,
        latent_size: int = 64,
        speech_vae_dim: int | None = 64,
        prediction_type: str = "v_prediction",
        diffusion_type: str = "ddpm",
        ddpm_num_steps: int = 1000,
        ddpm_num_inference_steps: int = 20,
        ddpm_beta_schedule: str = "cosine",
        ddpm_batch_mul: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.head_layers = head_layers
        self.head_ffn_ratio = head_ffn_ratio
        self.rms_norm_eps = rms_norm_eps
        self.latent_size = latent_size
        self.speech_vae_dim = speech_vae_dim
        self.prediction_type = prediction_type
        self.diffusion_type = diffusion_type
        self.ddpm_num_steps = ddpm_num_steps
        self.ddpm_num_inference_steps = ddpm_num_inference_steps
        self.ddpm_beta_schedule = ddpm_beta_schedule
        self.ddpm_batch_mul = ddpm_batch_mul


class VibeVoiceConfig(PretrainedConfig):
    """Composite config used by ``microsoft/VibeVoice-1.5B``."""

    model_type = "vibevoice"
    is_composition = True
    sub_configs = {
        "acoustic_tokenizer_config": VibeVoiceAcousticTokenizerConfig,
        "semantic_tokenizer_config": VibeVoiceSemanticTokenizerConfig,
        "decoder_config": Qwen2Config,
        "diffusion_head_config": VibeVoiceDiffusionHeadConfig,
    }

    def __init__(
        self,
        acoustic_tokenizer_config: dict[str, Any] | PretrainedConfig | None = None,
        semantic_tokenizer_config: dict[str, Any] | PretrainedConfig | None = None,
        decoder_config: dict[str, Any] | PretrainedConfig | None = None,
        diffusion_head_config: dict[str, Any] | PretrainedConfig | None = None,
        codec_frame_rate_hz: float = 7.5,
        **kwargs: Any,
    ) -> None:
        self.acoustic_tokenizer_config = _coerce_subconfig(acoustic_tokenizer_config, VibeVoiceAcousticTokenizerConfig)
        self.semantic_tokenizer_config = _coerce_subconfig(semantic_tokenizer_config, VibeVoiceSemanticTokenizerConfig)
        self.decoder_config = _coerce_subconfig(decoder_config, Qwen2Config)
        self.diffusion_head_config = _coerce_subconfig(diffusion_head_config, VibeVoiceDiffusionHeadConfig)
        self.acoustic_vae_dim = int(getattr(self.acoustic_tokenizer_config, "vae_dim", 64))
        self.semantic_vae_dim = int(getattr(self.semantic_tokenizer_config, "vae_dim", 128))
        self.codec_frame_rate_hz = float(codec_frame_rate_hz)
        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        del decoder
        return self.decoder_config

    @property
    def vocab_size(self) -> int:
        return int(self.decoder_config.vocab_size)

    @property
    def hidden_size(self) -> int:
        return int(self.decoder_config.hidden_size)

    @property
    def num_attention_heads(self) -> int:
        return int(self.decoder_config.num_attention_heads)

    @property
    def num_key_value_heads(self) -> int:
        return int(self.decoder_config.num_key_value_heads)

    @property
    def num_hidden_layers(self) -> int:
        return int(self.decoder_config.num_hidden_layers)

    @property
    def head_dim(self) -> int:
        return int(getattr(self.decoder_config, "head_dim", self.hidden_size // self.num_attention_heads))


def _coerce_subconfig(value: dict[str, Any] | PretrainedConfig | None, cls: type[PretrainedConfig]):
    if value is None:
        return cls()
    if isinstance(value, dict):
        value = dict(value)
        value.pop("model_type", None)
        return cls(**value)
    return value


try:
    AutoConfig.register("vibevoice", VibeVoiceConfig, exist_ok=True)
except TypeError:  # compatibility with older Transformers without exist_ok
    try:
        AutoConfig.register("vibevoice", VibeVoiceConfig)
    except ValueError:
        pass


__all__ = [
    "VibeVoiceAcousticTokenizerConfig",
    "VibeVoiceConfig",
    "VibeVoiceDiffusionHeadConfig",
    "VibeVoiceSemanticTokenizerConfig",
]
