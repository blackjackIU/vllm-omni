# SPDX-License-Identifier: Apache-2.0
"""Deterministic prompt construction for VibeVoice TTS.

The original processor builds a token prompt whose speech-placeholder count is
derived only from the normalized waveform length.  Keeping that logic here
lets the API server submit the *real* prompt token ids to vLLM (so prefix-cache
hashes are correct) while the GPU worker independently builds the matching
continuous prompt embeddings.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

SAMPLE_RATE = 24_000
SPEECH_COMPRESSION_RATIO = 3_200
SYSTEM_PROMPT = (
    " Transform the text provided by various speakers into speech output, "
    "utilizing the distinct voice of each respective speaker.\n"
)

_SPEAKER_LINE_RE = re.compile(r"^Speaker\s+(\d+)\s*:\s*(.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class VibeVoiceSpecialTokens:
    speech_start_id: int
    speech_end_id: int
    speech_diffusion_id: int
    eos_id: int
    pad_id: int
    bos_id: int | None


def configure_vibevoice_tokenizer(tokenizer: Any) -> VibeVoiceSpecialTokens:
    """Install/resolve the three Qwen vision tokens reused by VibeVoice."""
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<|vision_start|>",
                "<|vision_end|>",
                "<|vision_pad|>",
            ]
        }
    )
    image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_pad is None or int(image_pad) < 0:
        image_pad = tokenizer.pad_token_id
    return VibeVoiceSpecialTokens(
        speech_start_id=int(tokenizer.convert_tokens_to_ids("<|vision_start|>")),
        speech_end_id=int(tokenizer.convert_tokens_to_ids("<|vision_end|>")),
        speech_diffusion_id=int(tokenizer.convert_tokens_to_ids("<|vision_pad|>")),
        eos_id=int(tokenizer.eos_token_id),
        pad_id=int(image_pad),
        bos_id=None if tokenizer.bos_token_id is None else int(tokenizer.bos_token_id),
    )


def parse_script(script: str) -> list[tuple[int, str]]:
    parsed: list[tuple[int, str]] = []
    for raw_line in script.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _SPEAKER_LINE_RE.match(line)
        # Match the released processor: malformed/unlabelled lines are not
        # silently assigned to a speaker. The API adapter adds ``Speaker 1:``
        # when the complete input is plain text, so valid public requests keep
        # the convenient shorthand while mixed malformed scripts fail closed.
        if match is not None:
            parsed.append((int(match.group(1)), match.group(2).strip()))
    if not parsed:
        raise ValueError("VibeVoice script contains no speakable text")
    # The released processor converts user-facing Speaker 1..N labels to the
    # zero-based speaker indices used in both its text and voice prompts.
    if min(speaker_id for speaker_id, _ in parsed) > 0:
        parsed = [(speaker_id - 1, text) for speaker_id, text in parsed]
    return parsed


def resampled_num_samples(num_samples: int, sample_rate: int) -> int:
    if sample_rate <= 0:
        raise ValueError(f"Invalid reference sample rate: {sample_rate}")
    return int(math.ceil(int(num_samples) * SAMPLE_RATE / int(sample_rate)))


def reference_frame_count(num_samples_24khz: int) -> int:
    return int(math.ceil(max(0, int(num_samples_24khz)) / SPEECH_COMPRESSION_RATIO))


def build_prompt_token_ids(
    tokenizer: Any,
    script: str,
    reference_num_samples_24khz: list[int] | None = None,
) -> tuple[list[int], list[bool], VibeVoiceSpecialTokens]:
    """Mirror the released VibeVoice processor without loading its package."""
    special = configure_vibevoice_tokenizer(tokenizer)
    parsed = parse_script(script)
    token_ids = list(tokenizer.encode(SYSTEM_PROMPT))
    speech_mask = [False] * len(token_ids)

    references = reference_num_samples_24khz or []
    if references:
        prefix = list(tokenizer.encode(" Voice input:\n", add_special_tokens=False))
        token_ids.extend(prefix)
        speech_mask.extend([False] * len(prefix))
        for speaker_index, num_samples in enumerate(references):
            speaker_prefix = list(
                tokenizer.encode(f" Speaker {speaker_index}:", add_special_tokens=False)
            )
            frames = reference_frame_count(num_samples)
            segment = (
                speaker_prefix
                + [special.speech_start_id]
                + [special.speech_diffusion_id] * frames
                + [special.speech_end_id]
                + list(tokenizer.encode("\n", add_special_tokens=False))
            )
            segment_mask = (
                [False] * len(speaker_prefix)
                + [False]
                + [True] * frames
                + [False]
                + [False]
            )
            token_ids.extend(segment)
            speech_mask.extend(segment_mask)

    text_header = list(tokenizer.encode(" Text input:\n", add_special_tokens=False))
    token_ids.extend(text_header)
    speech_mask.extend([False] * len(text_header))
    for speaker_id, text in parsed:
        # VibeVoiceProcessor prepends one space to ``speaker_text`` before
        # interpolating it directly after the colon.  Our parser stores clean
        # text, so add that space here to produce the same final prompt bytes.
        line = list(
            tokenizer.encode(f" Speaker {speaker_id}: {text}\n", add_special_tokens=False)
        )
        token_ids.extend(line)
        speech_mask.extend([False] * len(line))

    output_header = list(tokenizer.encode(" Speech output:\n", add_special_tokens=False))
    token_ids.extend(output_header)
    token_ids.append(special.speech_start_id)
    speech_mask.extend([False] * (len(output_header) + 1))
    return token_ids, speech_mask, special


def normalize_reference_audio(audio: np.ndarray, target_db_fs: float = -25.0) -> np.ndarray:
    """Match the released processor's RMS normalization and clip guard."""
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise ValueError("Reference audio is empty")
    eps = 1e-6
    rms = float(np.sqrt(np.mean(waveform**2)))
    waveform = waveform * (10 ** (target_db_fs / 20.0) / (rms + eps))
    peak = float(np.max(np.abs(waveform)))
    if peak > 1.0:
        waveform = waveform / (peak + eps)
    return waveform.astype(np.float32, copy=False)


def resample_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Resample mono PCM with an optional torchaudio fast/high-quality path.

    A torch interpolation fallback keeps reference conditioning available on
    environments whose PyTorch and torchaudio CUDA wheels do not match.
    """
    sample_rate = int(sample_rate)
    target_sample_rate = int(target_sample_rate)
    if sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError(
            "VibeVoice sample rates must be positive, got "
            f"sample_rate={sample_rate}, target_sample_rate={target_sample_rate}"
        )
    waveform = waveform.reshape(-1)
    if waveform.numel() == 0:
        raise ValueError("VibeVoice reference waveform is empty")
    if sample_rate == target_sample_rate:
        return waveform
    try:
        import torchaudio

        return torchaudio.functional.resample(
            waveform,
            sample_rate,
            target_sample_rate,
        )
    except (ImportError, OSError, RuntimeError):
        output_length = int(
            math.ceil(waveform.shape[-1] * target_sample_rate / sample_rate)
        )
        return torch.nn.functional.interpolate(
            waveform.reshape(1, 1, -1),
            size=output_length,
            mode="linear",
            align_corners=False,
        ).reshape(-1)


def resample_and_normalize_reference(audio: Any, sample_rate: int) -> np.ndarray:
    """Canonical 24 kHz PCM used by prompt and conditioning cache keys."""
    waveform = torch.as_tensor(np.asarray(audio), dtype=torch.float32)
    if waveform.ndim == 2:
        channel_axis = 0 if waveform.shape[0] <= 8 else 1
        waveform = waveform.mean(dim=channel_axis)
    waveform = waveform.reshape(-1)
    waveform = resample_waveform(waveform, int(sample_rate), SAMPLE_RATE)
    return normalize_reference_audio(waveform.cpu().numpy())


__all__ = [
    "SAMPLE_RATE",
    "SPEECH_COMPRESSION_RATIO",
    "VibeVoiceSpecialTokens",
    "build_prompt_token_ids",
    "configure_vibevoice_tokenizer",
    "normalize_reference_audio",
    "parse_script",
    "reference_frame_count",
    "resample_and_normalize_reference",
    "resample_waveform",
    "resampled_num_samples",
]
