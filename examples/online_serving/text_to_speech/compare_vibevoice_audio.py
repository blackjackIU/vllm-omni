#!/usr/bin/env python3
"""Gate native VibeVoice WAV parity against a running legacy endpoint."""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import wave

import numpy as np


def _request(url: str, body: dict, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _pcm(wav_bytes: bytes) -> tuple[int, np.ndarray]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as stream:
        if stream.getsampwidth() != 2:
            raise ValueError("Parity tool currently requires 16-bit PCM WAV output")
        sample_rate = stream.getframerate()
        channels = stream.getnchannels()
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
        return sample_rate, samples.reshape(-1, channels).astype(np.float32) / 32768.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-url", default="http://127.0.0.1:8092/v1/audio/speech")
    parser.add_argument("--native-url", default="http://127.0.0.1:8091/v1/audio/speech")
    parser.add_argument("--model", default="microsoft/VibeVoice-1.5B")
    parser.add_argument("--input", default="Speaker 1: This is a VibeVoice parity test.")
    parser.add_argument("--ref-audio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ddpm-steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=1.3)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    body = {
        "model": args.model,
        "input": args.input,
        "response_format": "wav",
        "seed": args.seed,
        "extra_params": {
            "disable_prefill": args.ref_audio is None,
            "cfg_scale": args.cfg_scale,
            "ddpm_steps": args.ddpm_steps,
        },
    }
    if args.ref_audio:
        body["ref_audio"] = args.ref_audio

    legacy_rate, legacy = _pcm(_request(args.legacy_url, body, args.timeout))
    native_rate, native = _pcm(_request(args.native_url, body, args.timeout))
    shape_equal = legacy.shape == native.shape
    rate_equal = legacy_rate == native_rate
    finite = bool(np.isfinite(native).all())
    unclipped = bool((np.abs(native) <= 1.0).all())
    close = bool(shape_equal and np.allclose(native, legacy, rtol=args.rtol, atol=args.atol))
    report = {
        "sample_rate_equal": rate_equal,
        "sample_count_equal": shape_equal,
        "legacy_shape": list(legacy.shape),
        "native_shape": list(native.shape),
        "native_finite": finite,
        "native_unclipped": unclipped,
        "waveform_allclose": close,
        "max_abs_error": float(np.max(np.abs(native - legacy))) if shape_equal and native.size else None,
        "passed": rate_equal and shape_equal and finite and unclipped and close,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
