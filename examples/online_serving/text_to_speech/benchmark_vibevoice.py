#!/usr/bin/env python3
"""Small concurrent benchmark for the OpenAI-compatible VibeVoice endpoint."""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import threading
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Result:
    latency_s: float
    audio_s: float
    bytes_received: int


def _audio_duration(payload: bytes) -> float:
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
    except (EOFError, wave.Error, ZeroDivisionError):
        return 0.0


def _request(
    url: str,
    body: dict,
    timeout: float,
    start_barrier: threading.Barrier | None = None,
) -> Result:
    if start_barrier is not None:
        start_barrier.wait()
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    latency = time.perf_counter() - started
    return Result(latency, _audio_duration(payload), len(payload))


def _capabilities_url(speech_url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    parsed = urlsplit(speech_url)
    return urlunsplit(
        (parsed.scheme, parsed.netloc, "/v1/audio/capabilities", "", "")
    )


def _fetch_capabilities(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8091/v1/audio/speech")
    parser.add_argument("--model", default="microsoft/VibeVoice-1.5B")
    parser.add_argument("--input", default="This is a concurrent VibeVoice benchmark request.")
    parser.add_argument("--ref-audio")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--ddpm-steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument(
        "--capabilities-url",
        help="Defaults to /v1/audio/capabilities on the speech server",
    )
    parser.add_argument("--min-speedup", type=float, default=0.0)
    parser.add_argument(
        "--max-latency-ratio",
        type=float,
        default=0.0,
        help="Fail when mean latency / baseline mean latency exceeds this value",
    )
    args = parser.parse_args()

    if args.requests < 1 or args.concurrency < 1 or args.warmup < 0:
        parser.error("requests/concurrency must be positive and warmup cannot be negative")

    body = {
        "model": args.model,
        "input": args.input,
        "response_format": "wav",
        "seed": args.seed,
        "extra_params": {
            "cfg_scale": args.cfg_scale,
            "ddpm_steps": args.ddpm_steps,
            "disable_prefill": args.ref_audio is None,
        },
    }
    if args.ref_audio:
        body["ref_audio"] = args.ref_audio

    for _ in range(args.warmup):
        _request(args.url, body, args.timeout)

    started = time.perf_counter()
    results: list[Result] = []
    worker_count = min(args.concurrency, args.requests)
    start_barrier = threading.Barrier(worker_count + 1)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = [
            pool.submit(
                _request,
                args.url,
                body,
                args.timeout,
                start_barrier if index < worker_count else None,
            )
            for index in range(args.requests)
        ]
        # Release the first full worker wave together, so the benchmark
        # measures server batching rather than client submission jitter.
        start_barrier.wait()
        for future in as_completed(futures):
            results.append(future.result())
    wall_s = time.perf_counter() - started

    latencies = [result.latency_s for result in results]
    audio_s = sum(result.audio_s for result in results)
    audio_per_second = audio_s / wall_s if wall_s > 0 else 0.0
    summary = {
        "requests": len(results),
        "concurrency": args.concurrency,
        "wall_s": wall_s,
        "latency_mean_s": statistics.fmean(latencies),
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
        "requests_per_second": len(results) / wall_s,
        "audio_s": audio_s,
        "audio_seconds_per_second": audio_per_second,
        "aggregate_rtf": wall_s / audio_s if audio_s > 0 else float("inf"),
        "bytes": sum(result.bytes_received for result in results),
    }
    capabilities_url = args.capabilities_url or _capabilities_url(args.url)
    try:
        capabilities = _fetch_capabilities(
            capabilities_url,
            min(args.timeout, 30.0),
        )
        scheduler_evidence = capabilities.get("scheduler_evidence", {})
        summary["scheduler_evidence"] = scheduler_evidence
        summary["runtime_capabilities"] = {
            key: capabilities.get(key)
            for key in (
                "continuous_batching",
                "positive_cuda_graph_captured",
                "negative_cuda_graph_enabled",
                "negative_cuda_graph_disabled_reason",
                "diffusion_compiled",
                "codec_compiled",
            )
        }
    except Exception as exc:
        summary["capabilities_error"] = str(exc)
    print(f"requests={len(results)} concurrency={args.concurrency} wall_s={wall_s:.3f}")
    print(
        f"latency_s mean={summary['latency_mean_s']:.3f} "
        f"p50={summary['latency_p50_s']:.3f} p95={summary['latency_p95_s']:.3f}"
    )
    print(
        f"throughput_rps={summary['requests_per_second']:.4f} "
        f"audio_s_per_s={audio_per_second:.4f} aggregate_rtf={summary['aggregate_rtf']:.4f}"
    )
    print(f"audio_s={audio_s:.3f} bytes={summary['bytes']}")
    if scheduler_evidence := summary.get("scheduler_evidence"):
        print(
            "scheduler_max_batch_size="
            f"{scheduler_evidence.get('scheduler_max_batch_size', 0)} "
            "continuous_join_steps="
            f"{scheduler_evidence.get('continuous_join_steps', 0)}"
        )
    elif capabilities_error := summary.get("capabilities_error"):
        print(f"capabilities_warning={capabilities_error}")

    if args.baseline_json is not None:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        baseline_throughput = float(baseline["audio_seconds_per_second"])
        speedup = audio_per_second / baseline_throughput if baseline_throughput > 0 else 0.0
        summary["baseline_audio_seconds_per_second"] = baseline_throughput
        summary["speedup"] = speedup
        summary["speedup_gate"] = speedup >= args.min_speedup
        baseline_latency = float(baseline["latency_mean_s"])
        latency_ratio = summary["latency_mean_s"] / baseline_latency if baseline_latency > 0 else float("inf")
        summary["baseline_latency_mean_s"] = baseline_latency
        summary["latency_ratio"] = latency_ratio
        summary["latency_gate"] = (
            args.max_latency_ratio <= 0 or latency_ratio <= args.max_latency_ratio
        )
        print(f"speedup={speedup:.4f} required={args.min_speedup:.4f}")
        if args.max_latency_ratio > 0:
            print(
                f"latency_ratio={latency_ratio:.4f} "
                f"maximum={args.max_latency_ratio:.4f}"
            )

    if args.json_output is not None:
        args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if summary.get("speedup_gate") is False or summary.get("latency_gate") is False:
        sys.exit(2)


if __name__ == "__main__":
    main()
