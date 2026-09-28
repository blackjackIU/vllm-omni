# SPDX-License-Identifier: Apache-2.0
"""Serving adapter for ``microsoft/VibeVoice-1.5B``."""

from __future__ import annotations

import copy
import hashlib
import re
from typing import TYPE_CHECKING, Any

from vllm.inputs import tokens_input

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, conditioning_cache_salt
from vllm_omni.model_executor.models.vibevoice.prompting import (
    build_prompt_token_ids,
    resample_and_normalize_reference,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

_SPEAKER_RE = re.compile(r"(?im)^\s*speaker\s+(\d+)\s*:")
_ALLOWED_EXTRA_PARAMS = frozenset(
    {
        "cfg_scale",
        "ddpm_steps",
        "disable_prefill",
        "max_length_times",
        "max_new_tokens",
        "seed",
    }
)


def _script_for_vibevoice(text: str) -> str:
    text = text.strip().replace("’", "'")
    return text if _SPEAKER_RE.search(text) else f"Speaker 1: {text}"


@register_tts_adapter
class VibeVoiceAdapter(ARTTSAdapter):
    name = "vibevoice"
    stage_keys = frozenset({"vibevoice", "vibevoice_streaming", "vibevoice_legacy"})
    model_archs = frozenset(
        {
            "VibeVoiceForConditionalGeneration",
            "VibeVoiceStreamingForConditionalGeneration",
            "VibeVoiceNativeForConditionalGeneration",
            "LegacyVibeVoiceForConditionalGeneration",
            "LegacyVibeVoiceStreamingForConditionalGeneration",
        }
    )
    max_new_tokens_max = 65536

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        server = self.ctx.server
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"

        extra = request.extra_params or {}
        unknown = set(extra) - _ALLOWED_EXTRA_PARAMS
        if unknown:
            return f"Unsupported VibeVoice extra_params: {', '.join(sorted(unknown))}"
        try:
            cfg_scale = float(extra.get("cfg_scale", 1.3))
            ddpm_steps = int(extra.get("ddpm_steps", 10))
            max_length_times = float(extra.get("max_length_times", 2.0))
        except (TypeError, ValueError):
            return "VibeVoice cfg_scale, ddpm_steps, and max_length_times must be numeric"
        if not 1.0 <= cfg_scale <= 5.0:
            return "VibeVoice extra_params.cfg_scale must be between 1.0 and 5.0"
        if not 1 <= ddpm_steps <= 100:
            return "VibeVoice extra_params.ddpm_steps must be between 1 and 100"
        if not 1.0 <= max_length_times <= 10.0:
            return "VibeVoice extra_params.max_length_times must be between 1.0 and 10.0"

        requested_max_tokens = (
            request.max_new_tokens
            if request.max_new_tokens is not None
            else extra.get("max_new_tokens")
        )
        if requested_max_tokens is not None:
            try:
                requested_max_tokens = int(requested_max_tokens)
            except (TypeError, ValueError):
                return "VibeVoice extra_params.max_new_tokens must be an integer"
            if requested_max_tokens < self.max_new_tokens_min:
                return f"max_new_tokens must be at least {self.max_new_tokens_min}"
            if requested_max_tokens > self.max_new_tokens_max:
                return f"max_new_tokens cannot exceed {self.max_new_tokens_max}"
        if extra.get("seed") is not None:
            try:
                int(extra["seed"])
            except (TypeError, ValueError):
                return "VibeVoice extra_params.seed must be an integer"

        disable_prefill = bool(extra.get("disable_prefill", False))
        if request.ref_audio is None and not disable_prefill:
            err = server._apply_uploaded_speaker(request)
            if err:
                return err
        if request.ref_audio is None and not disable_prefill:
            return "VibeVoice voice cloning requires 'ref_audio' (one clip per speaker)"

        references = request.ref_audio if isinstance(request.ref_audio, list) else [request.ref_audio]
        references = [item for item in references if item is not None]
        if len(references) > 4:
            return "VibeVoice supports at most four reference speakers"
        for ref_audio in references:
            err = server._validate_ref_audio_format(ref_audio)
            if err:
                return err

        speaker_ids = list(dict.fromkeys(_SPEAKER_RE.findall(request.input)))
        if len(speaker_ids) > 4:
            return "VibeVoice scripts support at most four distinct speakers"
        if any(speaker_id not in {"1", "2", "3", "4"} for speaker_id in speaker_ids):
            return "VibeVoice speaker labels must be between Speaker 1 and Speaker 4"
        if speaker_ids:
            expected_ids = {str(index) for index in range(1, len(speaker_ids) + 1)}
            if set(speaker_ids) != expected_ids:
                return "VibeVoice speaker labels must be contiguous and start at Speaker 1"
        if speaker_ids and references and len(speaker_ids) != len(references):
            return (
                "VibeVoice requires one reference clip per speaker; "
                f"found {len(speaker_ids)} speakers but {len(references)} reference clips"
            )
        if not speaker_ids and len(references) > 1:
            return "Multi-speaker VibeVoice requests must label lines as 'Speaker N: ...'"
        return None

    async def build(
        self,
        request: OpenAICreateSpeechRequest,
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        del has_inline_ref_audio
        server = self.ctx.server
        if isinstance(request.ref_audio, list):
            ref_audio_data = await server._resolve_ref_audio_many(request.ref_audio)
        elif request.ref_audio is not None:
            ref_audio_data = [await server._resolve_ref_audio(request.ref_audio)]
        else:
            ref_audio_data = None

        extra: dict[str, Any] = dict(request.extra_params or {})
        params: dict[str, Any] = {
            "text": _script_for_vibevoice(request.input),
            "ref_audio_data": ref_audio_data,
            "cfg_scale": float(extra.get("cfg_scale", 1.3)),
            "ddpm_steps": int(extra.get("ddpm_steps", 10)),
            "disable_prefill": bool(extra.get("disable_prefill", False)),
            "max_length_times": float(extra.get("max_length_times", 2.0)),
        }
        requested_max_tokens = (
            request.max_new_tokens
            if request.max_new_tokens is not None
            else extra.get("max_new_tokens")
        )
        if requested_max_tokens is not None:
            params["max_new_tokens"] = int(requested_max_tokens)
        if request.seed is not None:
            params["seed"] = int(request.seed)
        elif extra.get("seed") is not None:
            params["seed"] = int(extra["seed"])
        elif sampling_params_list and getattr(sampling_params_list[0], "seed", None) is not None:
            params["seed"] = int(sampling_params_list[0].seed)

        stage_key = getattr(
            getattr(getattr(server, "_tts_stage", None), "engine_args", None),
            "model_stage",
            None,
        )
        if stage_key == "vibevoice_legacy":
            prompt_ids = [1]
        else:
            tokenizer = server._get_usage_text_tokenizer()
            if tokenizer is None:
                raise RuntimeError("Unable to load the Qwen tokenizer required by native VibeVoice")
            reference_lengths: list[int] = []
            normalized_references: list[Any] = []
            for reference in ref_audio_data or []:
                waveform, sample_rate = reference
                normalized = resample_and_normalize_reference(waveform, int(sample_rate))
                normalized_references.append(normalized)
                reference_lengths.append(len(normalized))
            prompt_ids, _, _ = build_prompt_token_ids(
                tokenizer,
                params["text"],
                reference_lengths if not params["disable_prefill"] else None,
            )
            params["native_prompt_len"] = len(prompt_ids)
        prompt = tokens_input(prompt_token_ids=prompt_ids)
        prompt["additional_information"] = params
        # Content tokens are already part of vLLM's prefix hash. Salt only the
        # continuous conditioning which cannot be represented by token ids.
        if ref_audio_data and stage_key != "vibevoice_legacy":
            digest = hashlib.sha256()
            digest.update(b"vibevoice-conditioning-v1\x00")
            model_config = server.engine_client.model_config
            digest.update(str(model_config.model).encode("utf-8"))
            digest.update(b"\x00")
            digest.update(str(getattr(model_config, "revision", None) or "main").encode("utf-8"))
            digest.update(b"\x00")
            digest.update(str(params.get("seed", 0)).encode("ascii"))
            for speaker_index, normalized in enumerate(normalized_references):
                digest.update(speaker_index.to_bytes(2, "little"))
                digest.update(hashlib.sha256(normalized.tobytes()).digest())
            prompt["cache_salt"] = digest.hexdigest()[:32]
        elif stage_key != "vibevoice_legacy":
            # Content already participates in vLLM's token-block hash. Keep
            # only non-token conditioning in the salt so repeated system/text
            # prefixes can genuinely hit the native prefix cache.
            model_config = server.engine_client.model_config
            digest = hashlib.sha256()
            digest.update(b"vibevoice-no-conditioning-v1\x00")
            digest.update(str(model_config.model).encode("utf-8"))
            digest.update(b"\x00")
            digest.update(str(getattr(model_config, "revision", None) or "main").encode("utf-8"))
            digest.update(b"\x00")
            # Generated continuous acoustic embeddings are not represented by
            # token IDs; keep different RNG streams out of the same KV prefix.
            digest.update(str(params.get("seed", 0)).encode("ascii"))
            prompt["cache_salt"] = digest.hexdigest()[:32]
        else:
            prompt["cache_salt"] = conditioning_cache_salt(request, params)
        return PreparedRequest(prompt=prompt, tts_params=params, model_type=self.name)

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: OpenAICreateSpeechRequest,
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        del request_id
        if prompt is None or len(prompt.get("prompt_token_ids", [])) <= 1:
            # Explicit legacy profile retains its one-shot scheduler contract.
            return sampling_params_list
        params = copy.deepcopy(sampling_params_list)
        prompt_len = len(prompt["prompt_token_ids"])
        extra = request.extra_params or {}
        requested = (
            request.max_new_tokens
            if request.max_new_tokens is not None
            else extra.get("max_new_tokens")
        )
        if requested is not None:
            requested = int(requested)
        derived = max(1, int(float(extra.get("max_length_times", 2.0)) * prompt_len))
        params[0].max_tokens = min(requested if requested is not None else derived, 65536)
        params[0].temperature = 0.0
        params[0].top_p = 1.0
        params[0].top_k = -1
        return params


__all__ = ["VibeVoiceAdapter"]
