"""Prometheus metrics emitted by the native VibeVoice execution path."""

from __future__ import annotations

import contextlib
import time
from collections import deque
from collections.abc import Iterator

import torch
from prometheus_client import Counter, Gauge, Histogram

_batch_size = Histogram(
    "vllm_omni_vibevoice_substage_batch_size",
    "Native VibeVoice active row count by GPU substage.",
    ("substage",),
    buckets=(1, 2, 4, 8, 16, 32, 64),
)
_active_requests = Gauge(
    "vllm_omni_vibevoice_active_requests",
    "Requests retaining native VibeVoice state in this worker.",
)
_negative_tokens = Gauge(
    "vllm_omni_vibevoice_negative_kv_logical_tokens",
    "Logical tokens currently retained across negative CFG paged-KV streams.",
)
_negative_blocks = Gauge(
    "vllm_omni_vibevoice_negative_kv_logical_blocks",
    "Logical paged-KV blocks addressed by active negative CFG streams.",
)
_cleanup = Counter(
    "vllm_omni_vibevoice_request_cleanup_total",
    "Native VibeVoice request states and codec slots reclaimed.",
)
_conditioning_cache = Counter(
    "vllm_omni_vibevoice_conditioning_cache_total",
    "Reference-conditioning cache lookups.",
    ("result",),
)
_substage_seconds = Histogram(
    "vllm_omni_vibevoice_substage_seconds",
    "Asynchronous native VibeVoice GPU substage duration.",
    ("substage",),
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
_speech_tokens = Counter(
    "vllm_omni_vibevoice_generated_speech_tokens_total",
    "Generated native VibeVoice diffusion/acoustic tokens.",
)
_audio_samples = Counter(
    "vllm_omni_vibevoice_generated_audio_samples_total",
    "PCM samples emitted by native VibeVoice.",
)
_negative_graph = Counter(
    "vllm_omni_vibevoice_negative_cuda_graph_total",
    "Native VibeVoice negative-CFG graph/compile execution outcomes.",
    ("result",),
)
_scheduler_batch_size = Histogram(
    "vllm_omni_vibevoice_scheduler_batch_size",
    "Requests scheduled together in a native VibeVoice engine step.",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)
_continuous_join = Counter(
    "vllm_omni_vibevoice_continuous_join_total",
    "Scheduler steps that admit new VibeVoice requests while cached requests continue.",
)
_prefix_cache = Counter(
    "vllm_omni_vibevoice_prefix_cache_requests_total",
    "Native VibeVoice requests classified from scheduler-computed prefix tokens.",
    ("result",),
)
_prefix_cache_tokens = Counter(
    "vllm_omni_vibevoice_prefix_cache_tokens_total",
    "Prompt tokens supplied by the vLLM KV prefix cache to native VibeVoice.",
)
_prefill_cycles = Histogram(
    "vllm_omni_vibevoice_prefill_cycles",
    "Scheduler prefill cycles used by each completed native VibeVoice request.",
    buckets=(1, 2, 3, 4, 8, 16, 32, 64),
)
_chunked_prefill = Counter(
    "vllm_omni_vibevoice_chunked_prefill_requests_total",
    "Native VibeVoice requests observed in more than one prefill scheduler cycle.",
)


class VibeVoiceMetrics:
    def __init__(self) -> None:
        self._pending_cuda_timings: deque[tuple[str, torch.cuda.Event, torch.cuda.Event]] = deque()
        self.scheduler_steps = 0
        self.scheduler_max_batch_size = 0
        self.continuous_join_steps = 0
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0
        self.prefix_cache_tokens = 0
        self.chunked_prefill_requests = 0

    def _drain_cuda_timings(self) -> None:
        while self._pending_cuda_timings and self._pending_cuda_timings[0][2].query():
            substage, started, finished = self._pending_cuda_timings.popleft()
            _substage_seconds.labels(substage=substage).observe(started.elapsed_time(finished) / 1000.0)

    @contextlib.contextmanager
    def timer(self, substage: str, device: torch.device) -> Iterator[None]:
        if device.type == "cuda":
            self._drain_cuda_timings()
            if torch.cuda.is_current_stream_capturing():
                yield
                return
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            try:
                yield
            finally:
                finished.record()
                self._pending_cuda_timings.append((substage, started, finished))
            return
        started_cpu = time.perf_counter()
        try:
            yield
        finally:
            _substage_seconds.labels(substage=substage).observe(time.perf_counter() - started_cpu)

    def observe_batch(self, substage: str, size: int) -> None:
        if size > 0:
            _batch_size.labels(substage=substage).observe(size)

    def set_state(
        self,
        *,
        active_requests: int,
        negative_tokens: int,
        negative_blocks: int = 0,
    ) -> None:
        _active_requests.set(active_requests)
        _negative_tokens.set(negative_tokens)
        _negative_blocks.set(negative_blocks)

    def cleanup(self, count: int) -> None:
        if count > 0:
            _cleanup.inc(count)

    def conditioning_cache(self, hit: bool) -> None:
        _conditioning_cache.labels(result="hit" if hit else "miss").inc()

    def generated(self, *, speech_tokens: int, audio_samples: int) -> None:
        if speech_tokens > 0:
            _speech_tokens.inc(speech_tokens)
        if audio_samples > 0:
            _audio_samples.inc(audio_samples)

    def negative_graph(self, result: str) -> None:
        _negative_graph.labels(result=result).inc()

    def scheduler_step(self, *, batch_size: int, new_requests: int, cached_requests: int) -> None:
        if batch_size <= 0:
            return
        _scheduler_batch_size.observe(batch_size)
        self.scheduler_steps += 1
        self.scheduler_max_batch_size = max(self.scheduler_max_batch_size, batch_size)
        if new_requests > 0 and cached_requests > 0:
            _continuous_join.inc()
            self.continuous_join_steps += 1

    def prefix_cache(self, *, computed_tokens: int) -> None:
        computed_tokens = max(0, int(computed_tokens))
        hit = computed_tokens > 0
        _prefix_cache.labels(result="hit" if hit else "miss").inc()
        if hit:
            self.prefix_cache_hits += 1
            self.prefix_cache_tokens += computed_tokens
            _prefix_cache_tokens.inc(computed_tokens)
        else:
            self.prefix_cache_misses += 1

    def finish_prefill(self, cycles: int) -> None:
        cycles = max(0, int(cycles))
        if cycles <= 0:
            return
        _prefill_cycles.observe(cycles)
        if cycles > 1:
            _chunked_prefill.inc()
            self.chunked_prefill_requests += 1

    def scheduler_snapshot(self) -> dict[str, int]:
        return {
            "scheduler_steps": self.scheduler_steps,
            "scheduler_max_batch_size": self.scheduler_max_batch_size,
            "continuous_join_steps": self.continuous_join_steps,
            "prefix_cache_hits": self.prefix_cache_hits,
            "prefix_cache_misses": self.prefix_cache_misses,
            "prefix_cache_tokens": self.prefix_cache_tokens,
            "chunked_prefill_requests": self.chunked_prefill_requests,
        }


__all__ = ["VibeVoiceMetrics"]
