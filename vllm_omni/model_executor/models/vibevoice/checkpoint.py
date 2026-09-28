# SPDX-License-Identifier: Apache-2.0
"""Cheap, allocation-free validation for the released VibeVoice-1.5B files."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_EXPECTED_COMPONENT_COUNTS = {
    "model.acoustic_connector": 5,
    "model.acoustic_tokenizer": 552,
    "model.language_model": 338,
    "model.prediction_head": 26,
    "model.semantic_connector": 5,
    "model.semantic_tokenizer": 276,
    "model.speech_bias_factor": 1,
    "model.speech_scaling_factor": 1,
}
_EXPECTED_DECODER = {
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "max_position_embeddings": 65536,
    "num_attention_heads": 12,
    "num_hidden_layers": 28,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
}


@dataclass(frozen=True)
class VibeVoiceCheckpointReport:
    model_dir: str
    architecture: str
    tensor_count: int
    shard_count: int
    total_size: int | None
    component_counts: dict[str, int]
    headers_checked: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _language_model_keys(num_layers: int) -> set[str]:
    keys = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    }
    for layer in range(num_layers):
        prefix = f"model.language_model.layers.{layer}"
        keys.update(
            {
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.v_proj.weight",
            }
        )
    return keys


def _prediction_head_keys(num_layers: int) -> set[str]:
    keys = {
        "model.prediction_head.cond_proj.weight",
        "model.prediction_head.noisy_images_proj.weight",
        "model.prediction_head.t_embedder.mlp.0.weight",
        "model.prediction_head.t_embedder.mlp.2.weight",
        "model.prediction_head.final_layer.adaLN_modulation.1.weight",
        "model.prediction_head.final_layer.linear.weight",
    }
    for layer in range(num_layers):
        prefix = f"model.prediction_head.layers.{layer}"
        keys.update(
            {
                f"{prefix}.adaLN_modulation.1.weight",
                f"{prefix}.ffn.down_proj.weight",
                f"{prefix}.ffn.gate_proj.weight",
                f"{prefix}.ffn.up_proj.weight",
                f"{prefix}.norm.weight",
            }
        )
    return keys


def _connector_keys(name: str) -> set[str]:
    return {
        f"model.{name}.fc1.bias",
        f"model.{name}.fc1.weight",
        f"model.{name}.fc2.bias",
        f"model.{name}.fc2.weight",
        f"model.{name}.norm.weight",
    }


def _validate_header_keys(model_dir: Path, weight_map: dict[str, str]) -> None:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError(
            "safetensors is required to validate checkpoint shard headers"
        ) from exc

    indexed_by_shard: dict[str, set[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        indexed_by_shard.setdefault(shard_name, set()).add(tensor_name)
    for shard_name, indexed_names in sorted(indexed_by_shard.items()):
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing VibeVoice checkpoint shard: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            header_names = set(shard.keys())
        if header_names != indexed_names:
            missing = sorted(indexed_names - header_names)
            unexpected = sorted(header_names - indexed_names)
            raise RuntimeError(
                f"VibeVoice shard/index mismatch in {shard_name}: "
                f"missing={missing[:10]} unexpected={unexpected[:10]}"
            )


def validate_vibevoice_checkpoint(
    model_dir: str | Path,
    *,
    inspect_headers: bool = True,
) -> VibeVoiceCheckpointReport:
    """Validate the official 1.5B manifest without allocating model tensors."""
    root = Path(model_dir).expanduser().resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing VibeVoice config: {config_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing VibeVoice safetensors index: {index_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    if "VibeVoiceForConditionalGeneration" not in architectures:
        raise RuntimeError(f"Unsupported VibeVoice architecture list: {architectures!r}")
    if config.get("model_type") != "vibevoice":
        raise RuntimeError(f"Expected model_type='vibevoice', got {config.get('model_type')!r}")

    decoder = config.get("decoder_config") or {}
    mismatched_decoder = {
        key: (expected, decoder.get(key))
        for key, expected in _EXPECTED_DECODER.items()
        if decoder.get(key) != expected
    }
    if mismatched_decoder:
        raise RuntimeError(f"Unsupported VibeVoice-1.5B decoder config: {mismatched_decoder}")
    if decoder.get("tie_word_embeddings") is not True:
        raise RuntimeError("VibeVoice-1.5B requires tied input/output embeddings")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("VibeVoice safetensors index has no weight_map")
    names = set(weight_map)
    if "lm_head.weight" in names or "model.lm_head.weight" in names:
        raise RuntimeError("Unexpected untied lm_head tensor in VibeVoice-1.5B checkpoint")

    expected_language = _language_model_keys(int(decoder["num_hidden_layers"]))
    actual_language = {name for name in names if name.startswith("model.language_model.")}
    if actual_language != expected_language:
        raise RuntimeError(
            "VibeVoice language-model manifest mismatch: "
            f"missing={sorted(expected_language - actual_language)[:10]} "
            f"unexpected={sorted(actual_language - expected_language)[:10]}"
        )

    diffusion = config.get("diffusion_head_config") or {}
    expected_prediction = _prediction_head_keys(int(diffusion.get("head_layers", -1)))
    actual_prediction = {name for name in names if name.startswith("model.prediction_head.")}
    if actual_prediction != expected_prediction:
        raise RuntimeError(
            "VibeVoice diffusion-head manifest mismatch: "
            f"missing={sorted(expected_prediction - actual_prediction)[:10]} "
            f"unexpected={sorted(actual_prediction - expected_prediction)[:10]}"
        )

    required_exact = (
        _connector_keys("acoustic_connector")
        | _connector_keys("semantic_connector")
        | {"model.speech_bias_factor", "model.speech_scaling_factor"}
    )
    missing_exact = sorted(required_exact - names)
    if missing_exact:
        raise RuntimeError(f"VibeVoice checkpoint is missing tensors: {missing_exact}")

    component_counts = Counter(".".join(name.split(".")[:2]) for name in names)
    if dict(component_counts) != _EXPECTED_COMPONENT_COUNTS:
        raise RuntimeError(
            "VibeVoice-1.5B component tensor counts differ from the released checkpoint: "
            f"expected={_EXPECTED_COMPONENT_COUNTS} actual={dict(component_counts)}"
        )

    shard_names = set(weight_map.values())
    if inspect_headers:
        _validate_header_keys(root, weight_map)
    else:
        missing_shards = sorted(name for name in shard_names if not (root / name).is_file())
        if missing_shards:
            raise FileNotFoundError(f"Missing VibeVoice checkpoint shards: {missing_shards}")

    metadata = index.get("metadata") or {}
    return VibeVoiceCheckpointReport(
        model_dir=str(root),
        architecture="VibeVoiceForConditionalGeneration",
        tensor_count=len(names),
        shard_count=len(shard_names),
        total_size=metadata.get("total_size"),
        component_counts=dict(sorted(component_counts.items())),
        headers_checked=inspect_headers,
    )


__all__ = ["VibeVoiceCheckpointReport", "validate_vibevoice_checkpoint"]
