# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optimized runtimes for ``microsoft/VibeVoice-1.5B``.

VibeVoice interleaves Qwen token generation with next-token diffusion and
immediately decodes each 7.5 Hz acoustic latent.  That loop cannot be expressed
as a stock vLLM token sampler without porting its positive/negative KV pair and
continuous-latent feedback into the AR runner.

This file preserves that one-shot implementation only as the explicit legacy
reference backend. Production architecture names are rebound at the bottom of
the module to the native scheduler-driven implementation. Legacy requests
admitted in the same scheduler step are grouped by compatible generation
parameters for A/B testing and rollback.

The original Microsoft repository no longer publishes the 1.5B inference
module.  A compatible ``vibevoice`` package which exposes
``VibeVoiceForConditionalGenerationInference`` and ``VibeVoiceProcessor`` must
be installed; the community preservation repository provides that API.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import queue
import sys
import threading
import types
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform

from .prompting import resample_waveform

logger = init_logger(__name__)

_SAMPLE_RATE = 24000
_DEFAULT_CFG_SCALE = 1.3
_DEFAULT_DDPM_STEPS = 10
_STREAM_TIMEOUT_SECONDS = 300.0
_STREAM_END = object()
_RUNTIME_INSTALL = (
    "pip install --no-deps "
    "git+https://github.com/vibevoice-community/VibeVoice.git@631804b9c1f042e381207fe87c54603fe6accbc1"
)


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _generation_group_key(info: dict[str, Any]) -> tuple[Any, ...]:
    """Return parameters which the preserved ``generate`` accepts per batch.

    The upstream loop exposes these as scalars, not per-row tensors. Keeping
    unlike requests in separate groups is required for correctness; requests
    with the common defaults still form one efficient GPU batch.
    """
    max_new_tokens = _pick(info, "max_new_tokens", None)
    seed = _pick(info, "seed", None)
    return (
        float(_pick(info, "cfg_scale", _DEFAULT_CFG_SCALE)),
        int(_pick(info, "ddpm_steps", _DEFAULT_DDPM_STEPS)),
        float(_pick(info, "max_length_times", 2.0)),
        None if max_new_tokens is None else int(max_new_tokens),
        bool(_pick(info, "disable_prefill", False)),
        None if seed is None else int(seed),
    )


def _group_generation_infos(infos: list[dict[str, Any]]) -> list[tuple[list[int], list[dict[str, Any]]]]:
    """Stable grouping used by the one-shot generation worker."""
    groups: dict[tuple[Any, ...], tuple[list[int], list[dict[str, Any]]]] = {}
    for index, info in enumerate(infos):
        key = _generation_group_key(info)
        indices, entries = groups.setdefault(key, ([], []))
        indices.append(index)
        entries.append(info)
    return list(groups.values())


def _pick(info: dict[str, Any], key: str, default: Any = None) -> Any:
    value = info.get(key, default)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _mono_float32(chunk: Any) -> torch.Tensor:
    audio = torch.as_tensor(chunk, dtype=torch.float32).detach().cpu()
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    if audio.ndim == 2:
        # VibeVoice normally emits [1, samples]. Be defensive about a
        # channels-last tensor supplied by a downstream implementation.
        channel_axis = 0 if audio.shape[0] <= audio.shape[1] else 1
        audio = audio.mean(dim=channel_axis)
    return audio.reshape(-1).contiguous()


def _resample_reference(waveform: Any, sample_rate: int) -> np.ndarray:
    audio = torch.as_tensor(np.asarray(waveform), dtype=torch.float32)
    if audio.ndim == 2:
        channel_axis = 0 if audio.shape[0] <= 8 else 1
        audio = audio.mean(dim=channel_axis)
    audio = audio.reshape(-1)
    audio = resample_waveform(audio, int(sample_rate), _SAMPLE_RATE)
    return audio.cpu().numpy().astype(np.float32, copy=False)


@contextlib.contextmanager
def _community_runtime_import_compat():
    """Bridge the preserved TTS runtime to the Transformers 5 API.

    Transformers 5 already maps VibeVoice's ASR/acoustic tokenizer. The
    preserved package registers its TTS versions under the same config names,
    and also imports the pre-5.0 Qwen2 fast-tokenizer module. Scope both shims
    to package import; the resulting TTS auto-model mappings intentionally stay.
    """
    from transformers import AutoModel, AutoModelForCausalLM

    patched_mappings: list[tuple[Any, Any]] = []
    for auto_cls in (AutoModel, AutoModelForCausalLM):
        mapping = auto_cls._model_mapping
        original_register = mapping.register

        def register(key, value, exist_ok=False, *, _original=original_register):
            del exist_ok
            return _original(key, value, exist_ok=True)

        mapping.register = register
        patched_mappings.append((mapping, original_register))

    fast_module_name = "transformers.models.qwen2.tokenization_qwen2_fast"
    inserted_fast_alias = False
    if fast_module_name not in sys.modules:
        try:
            __import__(fast_module_name)
        except ImportError:
            from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

            fast_module = types.ModuleType(fast_module_name)
            fast_module.Qwen2TokenizerFast = Qwen2Tokenizer
            sys.modules[fast_module_name] = fast_module
            inserted_fast_alias = True
    try:
        with warnings.catch_warnings():
            # The package imports its ASR processor from the TTS package. Its
            # optional ffmpeg helper is not shipped at the preserved revision;
            # TTS requests use already-decoded arrays and never need it.
            warnings.filterwarnings(
                "ignore",
                message="audio_utils not available, will fall back to soundfile for audio loading",
                category=UserWarning,
            )
            yield
    finally:
        for mapping, original_register in patched_mappings:
            mapping.register = original_register
        if inserted_fast_alias:
            sys.modules.pop(fast_module_name, None)


def _patch_community_runtime_class(model_cls: type[nn.Module]) -> None:
    """Bridge renamed Transformers 5 model capability hooks."""
    # Transformers 5 renamed the capability flag checked by
    # ``_flash_attn_can_dispatch``.  The preserved runtime correctly declares
    # the pre-5.0 spelling, so mirror it to the new spelling before model init.
    if getattr(model_cls, "_supports_flash_attn_2", False):
        model_cls._supports_flash_attn = True
    if getattr(model_cls, "_vllm_omni_transformers5_compat", False):
        return
    original_tie_weights = model_cls.tie_weights

    def tie_weights(self, *args, **kwargs):
        del args, kwargs
        return original_tie_weights(self)

    model_cls.tie_weights = tie_weights

    # The checkpoint intentionally omits the LM head because it shares the
    # decoder embedding matrix. Transformers 5 reports that omission unless
    # the expected tied key is explicitly ignored during loading.
    ignored_missing = set(getattr(model_cls, "_keys_to_ignore_on_load_missing", ()) or ())
    ignored_missing.add(r"lm_head\.weight")
    model_cls._keys_to_ignore_on_load_missing = ignored_missing

    # The preserved composite config predates ``get_text_config``. Without
    # this override Transformers 5 hands the outer VibeVoice config to
    # DynamicCache instead of its Qwen decoder config.
    config_cls = getattr(model_cls, "config_class", None)
    if config_cls is not None:

        def get_text_config(self, *args, **kwargs):
            del args, kwargs
            return self.decoder_config

        config_cls.get_text_config = get_text_config

    # Transformers 5.10 moved dynamic KV tensors into per-layer objects. The
    # preserved CFG loop still mutates the former key_cache/value_cache lists
    # in place, so expose read-through compatibility properties.
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "key_cache"):

        def key_cache(self):
            return [layer.keys for layer in self.layers]

        DynamicCache.key_cache = property(key_cache)
    if not hasattr(DynamicCache, "value_cache"):

        def value_cache(self):
            return [layer.values for layer in self.layers]

        DynamicCache.value_cache = property(value_cache)

    # Transformers 5 omits ``inputs_embeds`` from prepared inputs when it is
    # unused. VibeVoice's classifier-free-guidance loop indexes that key
    # directly before deciding whether it needs to replace the value.
    original_prepare_inputs = model_cls.prepare_inputs_for_generation

    def prepare_inputs_for_generation(self, *args, **kwargs):
        prepared = original_prepare_inputs(self, *args, **kwargs)
        prepared.setdefault("inputs_embeds", None)
        return prepared

    model_cls.prepare_inputs_for_generation = prepare_inputs_for_generation

    # The preserved runtime passes the former positional
    # ``use_model_defaults`` argument. Transformers 5 removed that argument
    # from GenerationMixin._prepare_generation_config, while keeping the rest
    # of the call contract unchanged.
    original_prepare_generation_config = model_cls._prepare_generation_config
    prepare_signature = inspect.signature(original_prepare_generation_config)
    if "use_model_defaults" not in prepare_signature.parameters:

        def prepare_generation_config(self, generation_config, *args, **kwargs):
            if len(args) > 1 or (args and not isinstance(args[0], bool)):
                raise TypeError("Unexpected positional arguments for _prepare_generation_config")
            # The community helper has already constructed a fresh config.
            # Fold generation parameters into it first so Transformers 5 does
            # not warn about mixing a config object with generation kwargs.
            if generation_config is not None:
                kwargs = generation_config.update(**kwargs)
            return original_prepare_generation_config(self, generation_config, **kwargs)

        model_cls._prepare_generation_config = prepare_generation_config

    # Transformers 5 also removed the trailing ``device`` argument from cache
    # preparation. The preserved 4.51 runtime still supplies it positionally.
    original_prepare_cache = model_cls._prepare_cache_for_generation
    cache_signature = inspect.signature(original_prepare_cache)
    if "device" not in cache_signature.parameters:

        def prepare_cache_for_generation(
            self,
            generation_config,
            model_kwargs,
            generation_mode,
            batch_size,
            max_cache_length,
            *args,
            **kwargs,
        ):
            if len(args) > 1:
                raise TypeError("Unexpected positional arguments for _prepare_cache_for_generation")
            kwargs.pop("device", None)
            return original_prepare_cache(
                self,
                generation_config,
                model_kwargs,
                generation_mode,
                batch_size,
                max_cache_length,
                **kwargs,
            )

        model_cls._prepare_cache_for_generation = prepare_cache_for_generation
    model_cls._vllm_omni_transformers5_compat = True


def _patch_community_scheduler_class(scheduler_cls: type[Any]) -> None:
    """Keep VibeVoice's non-module diffusion schedule off the meta device.

    Transformers 5 constructs pretrained models under ``torch.device("meta")``.
    VibeVoice's scheduler is regular Python state rather than an ``nn.Module``,
    so Transformers cannot materialize its tensors while loading the state
    dict.  Its constructor also immediately moves ``sigmas`` to CPU, which
    raises ``NotImplementedError`` for a meta tensor.  Build only this small
    schedule under an inner CPU device context.
    """
    if getattr(scheduler_cls, "_vllm_omni_transformers5_compat", False):
        return
    original_init = scheduler_cls.__init__

    def init(self, *args, **kwargs):
        with torch.device("cpu"):
            original_init(self, *args, **kwargs)

    scheduler_cls.__init__ = init
    scheduler_cls._vllm_omni_transformers5_compat = True


class _AudioQueueStreamer:
    """Small duck-typed replacement for VibeVoice's ``AudioStreamer``."""

    def __init__(self, output_queue: queue.Queue[Any]) -> None:
        self.output_queue = output_queue
        self.finished_flags = [False]

    def put(self, audio_chunks: torch.Tensor, sample_indices: torch.Tensor) -> None:
        for row, sample_idx in enumerate(sample_indices):
            if int(sample_idx.item()) == 0 and not self.finished_flags[0]:
                self.output_queue.put(_mono_float32(audio_chunks[row]))

    def end(self, sample_indices: torch.Tensor | None = None) -> None:
        if sample_indices is not None and not any(int(x.item()) == 0 for x in sample_indices):
            return
        if not self.finished_flags[0]:
            self.finished_flags[0] = True
            self.output_queue.put(_STREAM_END)


@dataclass
class _GenerationState:
    output_queue: queue.Queue[Any]
    stop_event: threading.Event
    thread: threading.Thread | None


class LegacyVibeVoiceStreamingForConditionalGeneration(nn.Module):
    """Incremental PCM wrapper around the preserved VibeVoice runtime."""

    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model_path = vllm_config.model_config.model
        self._device = current_omni_platform.get_torch_device()
        self._states: dict[str, _GenerationState] = {}
        self._ar_last_chunk_flags: list[bool] = []

        try:
            with _community_runtime_import_compat():
                from vibevoice.modular.modeling_vibevoice_inference import (
                    VibeVoiceForConditionalGenerationInference,
                )
                from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
                from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler
            _patch_community_runtime_class(VibeVoiceForConditionalGenerationInference)
            _patch_community_scheduler_class(DPMSolverMultistepScheduler)
        except ImportError as exc:
            raise ImportError(
                "VibeVoice 1.5B serving requires the preserved inference package. "
                f"Install the tested revision with `{_RUNTIME_INSTALL}`."
            ) from exc

        dtype = self._select_dtype()
        try:
            from transformers.utils import is_flash_attn_2_available

            flash_attention_available = is_flash_attn_2_available()
        except ImportError:
            flash_attention_available = False
        attention_impl = "flash_attention_2" if self._device.type == "cuda" and flash_attention_available else "sdpa"
        logger.info("Loading VibeVoice model from %s (dtype=%s, attention=%s)", self.model_path, dtype, attention_impl)
        try:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                dtype=dtype,
                attn_implementation=attention_impl,
                low_cpu_mem_usage=True,
            )
        except (ImportError, RuntimeError, ValueError):
            if attention_impl != "flash_attention_2":
                raise
            logger.warning("FlashAttention is unavailable for VibeVoice; falling back to SDPA")
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                self.model_path,
                dtype=dtype,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            )
        if getattr(model.config.decoder_config, "tie_word_embeddings", False):
            model.lm_head.weight = model.model.language_model.embed_tokens.weight
        model.to(self._device)
        model.eval()
        self._model = model
        self._processor = VibeVoiceProcessor.from_pretrained(self.model_path)
        self._configure_runtime_kernels()

    def _configure_runtime_kernels(self) -> None:
        """Apply opt-in hot-path compilation after checkpoint loading.

        Compiling only the small diffusion prediction head avoids wrapping the
        stateful Transformers generation loop. Inductor then reuses the same
        graph for every DDPM step of a fixed batch shape. Failures are
        deliberately non-fatal because PyTorch/driver combinations differ.
        """
        if self._device.type != "cuda":
            return
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if _env_enabled("VLLM_OMNI_VIBEVOICE_CUDNN_BENCHMARK"):
            torch.backends.cudnn.benchmark = True
        if not _env_enabled("VLLM_OMNI_VIBEVOICE_COMPILE_DIFFUSION"):
            return
        # This legacy backend also mutates generation state across calls. Keep
        # graph-tree ownership opt-in for the same portability reason as the
        # native scheduler-driven backend.
        mode = os.environ.get("VLLM_OMNI_VIBEVOICE_COMPILE_MODE", "default")
        try:
            prediction_head = self._model.model.prediction_head
            self._model.model.prediction_head = torch.compile(
                prediction_head,
                mode=mode,
                dynamic=True,
                fullgraph=False,
            )
            logger.info("VibeVoice diffusion head enabled torch.compile mode=%s", mode)
        except Exception as exc:  # pragma: no cover - backend specific
            logger.warning("VibeVoice diffusion-head compilation unavailable; using eager kernels: %s", exc)

    def _select_dtype(self) -> torch.dtype:
        requested = getattr(self.vllm_config.model_config, "dtype", None)
        if isinstance(requested, torch.dtype):
            return requested
        if self._device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if self._device.type == "mps":
            return torch.float16
        return torch.float32

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The reference model owns its composite checkpoint loading. vLLM still
        # supplies an iterator, which must be drained to release loader state.
        for _ in weights:
            pass
        return {name for name, _ in self.named_parameters()}

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, Any]]:
        return [{"text": "Speaker 1: hello", "_is_dummy": True} for _ in range(num_reqs)]

    def _prepare_inputs(self, info: dict[str, Any]) -> dict[str, Any]:
        script = str(_pick(info, "text", "") or "").strip()
        references = _pick(info, "ref_audio_data", None)
        voices: list[np.ndarray] | None = None
        if references and not bool(_pick(info, "disable_prefill", False)):
            if isinstance(references, tuple) and len(references) == 2 and isinstance(references[1], int):
                references = [references]
            voices = [_resample_reference(waveform, sample_rate) for waveform, sample_rate in references]

        inputs = self._processor(
            text=[script],
            voice_samples=[voices] if voices else None,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {key: value.to(self._device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    def _prepare_batch_inputs(self, infos: list[dict[str, Any]]) -> dict[str, Any]:
        scripts: list[str] = []
        voice_samples: list[list[np.ndarray] | None] = []
        for info in infos:
            scripts.append(str(_pick(info, "text", "") or "").strip())
            references = _pick(info, "ref_audio_data", None)
            voices: list[np.ndarray] | None = None
            if references and not bool(_pick(info, "disable_prefill", False)):
                if isinstance(references, tuple) and len(references) == 2 and isinstance(references[1], int):
                    references = [references]
                voices = [_resample_reference(waveform, sample_rate) for waveform, sample_rate in references]
            voice_samples.append(voices)

        inputs = self._processor(
            text=scripts,
            voice_samples=voice_samples,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {key: value.to(self._device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    def _generation_worker(
        self,
        state: _GenerationState,
        info: dict[str, Any],
        streamer: _AudioQueueStreamer,
    ) -> None:
        cpu_rng_state = None
        cuda_rng_state = None
        try:
            seed = _pick(info, "seed", None)
            if seed is not None:
                cpu_rng_state = torch.get_rng_state()
                cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))

            inputs = self._prepare_inputs(info)
            ddpm_steps = int(_pick(info, "ddpm_steps", _DEFAULT_DDPM_STEPS))
            self._model.set_ddpm_inference_steps(num_steps=ddpm_steps)
            self._model.generate(
                **inputs,
                max_new_tokens=_pick(info, "max_new_tokens", None),
                max_length_times=float(_pick(info, "max_length_times", 2.0)),
                cfg_scale=float(_pick(info, "cfg_scale", _DEFAULT_CFG_SCALE)),
                tokenizer=self._processor.tokenizer,
                generation_config={"do_sample": False},
                is_prefill=not bool(_pick(info, "disable_prefill", False)),
                return_speech=False,
                audio_streamer=streamer,
                stop_check_fn=state.stop_event.is_set,
                show_progress_bar=False,
                verbose=False,
            )
        except BaseException as exc:  # propagate failures to the scheduler thread
            state.output_queue.put(exc)
        finally:
            if cpu_rng_state is not None:
                torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            streamer.end()

    def _start_generation(self, info: dict[str, Any]) -> _GenerationState:
        # Do not bound this queue: after client cancellation the scheduler no
        # longer drains it, and a bounded queue could strand the daemon thread
        # inside ``streamer.put`` before it observes ``stop_event``.
        output_queue: queue.Queue[Any] = queue.Queue()
        stop_event = threading.Event()
        # The state must exist before the worker starts because its target reads
        # the same object for cancellation and error delivery.
        state = _GenerationState(output_queue=output_queue, stop_event=stop_event, thread=None)
        streamer = _AudioQueueStreamer(output_queue)
        thread = threading.Thread(
            target=self._generation_worker,
            args=(state, dict(info), streamer),
            name="vibevoice-generation",
            daemon=True,
        )
        state.thread = thread
        thread.start()
        return state

    def _next_chunk(self, request_key: str, info: dict[str, Any]) -> tuple[torch.Tensor, bool]:
        state = self._states.get(request_key)
        if state is None:
            state = self._start_generation(info)
            self._states[request_key] = state
        try:
            event = state.output_queue.get(timeout=_STREAM_TIMEOUT_SECONDS)
        except queue.Empty as exc:
            state.stop_event.set()
            self._states.pop(request_key, None)
            raise TimeoutError("Timed out waiting for a VibeVoice audio chunk") from exc
        if event is _STREAM_END:
            self._states.pop(request_key, None)
            return torch.zeros((0,), dtype=torch.float32), True
        if isinstance(event, BaseException):
            state.stop_event.set()
            self._states.pop(request_key, None)
            raise RuntimeError("VibeVoice generation failed") from event
        return _mono_float32(event), False

    def _make_dummy_hidden(self, input_ids: torch.Tensor | None) -> torch.Tensor:
        decoder_config = getattr(self.config, "decoder_config", self.config)
        hidden_size = int(getattr(decoder_config, "hidden_size", 1536))
        rows = 1 if input_ids is None else max(1, int(input_ids.shape[0]))
        return torch.zeros((rows, hidden_size), device=self._device, dtype=torch.float32)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds, kwargs
        hidden = self._make_dummy_hidden(input_ids)
        empty = torch.zeros((0,), dtype=torch.float32)
        sr = torch.tensor(_SAMPLE_RATE, dtype=torch.int32)
        infos = runtime_additional_information or [{}]
        if not runtime_additional_information or all(info.get("_is_dummy") for info in infos):
            self._ar_last_chunk_flags = [True] * len(infos)
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={"model_outputs": [empty] * len(infos), "sr": [sr] * len(infos)},
            )

        outputs: list[torch.Tensor] = []
        last_flags: list[bool] = []
        for info in infos:
            if info.get("_is_dummy"):
                outputs.append(empty)
                last_flags.append(True)
                continue
            request_key = str(info.get("global_request_id") or info.get("_omni_req_id") or id(info))
            chunk, is_last = self._next_chunk(request_key, info)
            outputs.append(chunk)
            last_flags.append(is_last)
        self._ar_last_chunk_flags = last_flags
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs={"model_outputs": outputs, "sr": [sr] * len(outputs)},
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for req_id in finished_req_ids:
            state = self._states.pop(str(req_id), None)
            if state is not None:
                state.stop_event.set()

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> torch.Tensor:
        del sampling_metadata
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states.ndim == 1:
            hidden_states = hidden_states.unsqueeze(-1)
        elif hidden_states.ndim > 2:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        decoder_config = getattr(self.config, "decoder_config", self.config)
        vocab_size = int(getattr(decoder_config, "vocab_size", 151936))
        logits = torch.zeros((hidden_states.shape[0], vocab_size), device=hidden_states.device, dtype=torch.float32)
        eos_id, keepalive_id = 2, 1
        for row in range(hidden_states.shape[0]):
            is_last = self._ar_last_chunk_flags[row] if row < len(self._ar_last_chunk_flags) else True
            logits[row, eos_id if is_last else keepalive_id] = 1.0e6
            logits[row, keepalive_id if is_last else eos_id] = -1.0e9
        return logits

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Any = None,
        is_multimodal: Any = None,
    ) -> torch.Tensor:
        del multimodal_embeddings, is_multimodal
        decoder_config = getattr(self.config, "decoder_config", self.config)
        hidden_size = int(getattr(decoder_config, "hidden_size", 1536))
        return torch.zeros((input_ids.shape[0], hidden_size), device=input_ids.device, dtype=torch.float32)


class LegacyVibeVoiceForConditionalGeneration(LegacyVibeVoiceStreamingForConditionalGeneration):
    """Batched one-shot VibeVoice runtime for ``LLM_GENERATION``.

    Unlike the streaming compatibility path this class does not create one
    Python thread per request and does not pretend an audio chunk is an AR
    token. The Omni generation scheduler admits several requests together;
    compatible requests execute in one upstream ``generate`` batch and return
    one waveform per request.
    """

    def _generate_group(self, infos: list[dict[str, Any]]) -> list[torch.Tensor]:
        first = infos[0]
        inputs = self._prepare_batch_inputs(infos)
        ddpm_steps = int(_pick(first, "ddpm_steps", _DEFAULT_DDPM_STEPS))
        seed = _pick(first, "seed", None)

        cpu_rng_state = None
        cuda_rng_state = None
        try:
            if seed is not None:
                cpu_rng_state = torch.get_rng_state()
                cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))

            self._model.set_ddpm_inference_steps(num_steps=ddpm_steps)
            generated = self._model.generate(
                **inputs,
                max_new_tokens=_pick(first, "max_new_tokens", None),
                max_length_times=float(_pick(first, "max_length_times", 2.0)),
                cfg_scale=float(_pick(first, "cfg_scale", _DEFAULT_CFG_SCALE)),
                tokenizer=self._processor.tokenizer,
                generation_config={"do_sample": False},
                is_prefill=not bool(_pick(first, "disable_prefill", False)),
                return_speech=True,
                show_progress_bar=False,
                verbose=False,
            )
        finally:
            if cpu_rng_state is not None:
                torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

        speech_outputs = getattr(generated, "speech_outputs", None) or []
        if len(speech_outputs) != len(infos):
            raise RuntimeError(
                "VibeVoice returned "
                f"{len(speech_outputs)} waveforms for a batch of {len(infos)} requests"
            )
        return [
            torch.zeros((0,), dtype=torch.float32) if audio is None else _mono_float32(audio)
            for audio in speech_outputs
        ]

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del input_ids, positions, intermediate_tensors, inputs_embeds
        infos = runtime_additional_information
        if infos is None:
            infos = kwargs.get("model_intermediate_buffer")
        seq_token_counts = kwargs.get("seq_token_counts")
        num_reqs = len(seq_token_counts) if seq_token_counts is not None else len(infos or [])
        infos = list(infos or [{} for _ in range(num_reqs)])
        if len(infos) < num_reqs:
            infos.extend({} for _ in range(num_reqs - len(infos)))

        empty = torch.zeros((0,), dtype=torch.float32)
        outputs: list[torch.Tensor] = [empty] * num_reqs
        real_indices = [index for index, info in enumerate(infos[:num_reqs]) if not info.get("_is_dummy")]
        real_infos = [infos[index] for index in real_indices]
        if real_infos:
            for relative_indices, grouped_infos in _group_generation_infos(real_infos):
                grouped_outputs = self._generate_group(grouped_infos)
                for relative_index, audio in zip(relative_indices, grouped_outputs, strict=True):
                    outputs[real_indices[relative_index]] = audio

        sr = torch.tensor(_SAMPLE_RATE, dtype=torch.int32)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": outputs, "sr": [sr] * num_reqs},
        )


from .modeling_vibevoice_native import (  # noqa: E402
    VibeVoiceNativeForConditionalGeneration,
)

# Production architecture names resolve to the native, scheduler-driven
# implementation.  The preserved HF loop remains importable under explicit
# ``Legacy*`` names for parity tests and emergency rollback only.
VibeVoiceForConditionalGeneration = VibeVoiceNativeForConditionalGeneration
VibeVoiceStreamingForConditionalGeneration = VibeVoiceNativeForConditionalGeneration


__all__ = [
    "LegacyVibeVoiceForConditionalGeneration",
    "LegacyVibeVoiceStreamingForConditionalGeneration",
    "VibeVoiceForConditionalGeneration",
    "VibeVoiceNativeForConditionalGeneration",
    "VibeVoiceStreamingForConditionalGeneration",
]
