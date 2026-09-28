# SPDX-License-Identifier: Apache-2.0
"""Native vLLM autoregressive runtime for ``microsoft/VibeVoice-1.5B``.

The Qwen2 backbone is executed by vLLM layers and its positive/negative CFG
passes own distinct paged-KV namespaces.  Diffusion and streaming codec work is
performed as batched GPU side computation before the positive AR step that
consumes the newly generated continuous embedding.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.attention.auxiliary_stream import (
    AuxiliaryAttentionStreamSpec,
    AuxiliarySequenceState,
)
from vllm_omni.metrics.vibevoice import VibeVoiceMetrics
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform

from .checkpoint import validate_vibevoice_checkpoint
from .native_components import (
    SpeechConnector,
    VibeVoiceBatchedStreamingCache,
    VibeVoiceDiffusionHead,
    VibeVoicePhase,
    VibeVoiceRequestState,
)
from .negative_cfg_graph import NegativeCFGGraphRunner
from .prompting import (
    SAMPLE_RATE,
    build_prompt_token_ids,
    configure_vibevoice_tokenizer,
    resample_and_normalize_reference,
)

logger = init_logger(__name__)

_STATE_KEY = "vibevoice_native"
_DEFAULT_CFG_SCALE = 1.3
_DEFAULT_DDPM_STEPS = 10


def _pick(info: dict[str, Any], key: str, default: Any = None) -> Any:
    value = info.get(key, default)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _module_device(module: nn.Module) -> torch.device:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return current_omni_platform.get_torch_device()


def _scheduled_single_token(info: dict[str, Any], flat_ids: torch.Tensor, start: int, end: int) -> int:
    if end - start != 1:
        return -1
    scheduled = info.get("_omni_scheduled_token_ids_cpu")
    if isinstance(scheduled, (list, tuple)) and len(scheduled) == 1:
        return int(scheduled[0])
    # Compatibility fallback for third-party runners which do not implement
    # the generic auxiliary-stream CPU-token contract.
    return int(flat_ids[start].item())


def _share_module_parameters(target: nn.Module, source: nn.Module) -> None:
    """Point ``target`` parameters at ``source`` while retaining its modules.

    Attention modules keep distinct prefixes/cache bindings, while QKV, MLP,
    embedding and norm tensors consume memory only once.
    """
    source_parameters = dict(source.named_parameters(remove_duplicate=False))
    for name, source_parameter in source_parameters.items():
        path, _, leaf = name.rpartition(".")
        target_parent = target.get_submodule(path) if path else target
        if leaf in target_parent._parameters:
            target_parent._parameters[leaf] = source_parameter
    source_buffers = dict(source.named_buffers(remove_duplicate=False))
    for name, source_buffer in source_buffers.items():
        path, _, leaf = name.rpartition(".")
        target_parent = target.get_submodule(path) if path else target
        # KV-cache bindings and scale buffers belong to the Attention object;
        # share only immutable model buffers present on both trees.
        if leaf in target_parent._buffers and "kv_cache" not in name:
            target_parent._buffers[leaf] = source_buffer


def _native_checkpoint_target_name(name: str) -> str:
    """Resolve one HF checkpoint key to its native (possibly packed) key."""
    if not name.startswith("model.language_model."):
        return name
    replacements = {
        ".q_proj": ".qkv_proj",
        ".k_proj": ".qkv_proj",
        ".v_proj": ".qkv_proj",
        ".gate_proj": ".gate_up_proj",
        ".up_proj": ".gate_up_proj",
    }
    for source, target in replacements.items():
        if source in name:
            return name.replace(source, target)
    return name


class _VibeVoiceNativeModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config
        # Qwen2Model reads a flat text config. VibeVoice stores it under
        # decoder_config, while cache/parallel/compile settings stay inherited
        # from the composite parent config.
        backbone_vllm_config = copy.copy(vllm_config)
        backbone_model_config = copy.copy(vllm_config.model_config)
        backbone_model_config.hf_config = config.decoder_config
        backbone_model_config.hf_text_config = config.decoder_config
        backbone_vllm_config.model_config = backbone_model_config
        language_prefix = maybe_prefix(prefix, "language_model")
        negative_prefix = maybe_prefix(prefix, "negative_language_model")
        self.language_model = Qwen2Model(vllm_config=backbone_vllm_config, prefix=language_prefix)
        self.negative_language_model = Qwen2Model(vllm_config=backbone_vllm_config, prefix=negative_prefix)
        _share_module_parameters(self.negative_language_model, self.language_model)

        # Numerically compatible inference-only codec/scheduler definitions are
        # versioned in-tree. The native profile therefore never imports the
        # community package, constructs a Transformers Qwen, or calls generate.
        from .vendored_dpm_solver import DPMSolverMultistepScheduler
        from .vendored_tokenizer import (
            VibeVoiceAcousticTokenizerModel,
            VibeVoiceSemanticTokenizerModel,
        )

        self.acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(config.acoustic_tokenizer_config)
        self.semantic_tokenizer = VibeVoiceSemanticTokenizerModel(config.semantic_tokenizer_config)
        self.acoustic_connector = SpeechConnector(config.acoustic_vae_dim, config.decoder_config.hidden_size)
        self.semantic_connector = SpeechConnector(config.semantic_vae_dim, config.decoder_config.hidden_size)
        self.prediction_head = VibeVoiceDiffusionHead(config.diffusion_head_config)
        self.register_buffer("speech_scaling_factor", torch.tensor(float("nan")))
        self.register_buffer("speech_bias_factor", torch.tensor(float("nan")))
        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type,
        )
        cache_capacity = int(vllm_config.scheduler_config.max_num_seqs)
        self.acoustic_cache = VibeVoiceBatchedStreamingCache(cache_capacity)
        self.semantic_cache = VibeVoiceBatchedStreamingCache(cache_capacity)


class VibeVoiceNativeForConditionalGeneration(nn.Module):
    """Step-wise VibeVoice using native Qwen2 and dual vLLM paged KV."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = True
    enable_update_additional_information = True
    requires_full_prefix_cached_hidden_states = False
    # vLLM's scheduler-owned paged KV prefix cache remains enabled. The Omni
    # tensor prefix cache (hidden/multimodal payload snapshots) is unnecessary
    # for this model and would cache delta PCM, so opt out of that second cache.
    disable_omni_tensor_prefix_cache = True
    omni_pooler_payload_include_hidden = False
    # State updates consume GPU hidden states before snapshotting; PCM can then
    # use the runner's pinned-memory, dedicated-stream D2H path.
    use_async_omni_output = True
    eager_omni_postprocess_before_async_output = True
    postprocess_uses_multimodal_outputs = False
    supports_auxiliary_attention_streams = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:
            raise ValueError("Native VibeVoice does not yet support pipeline parallelism")
        compilation_config = getattr(vllm_config, "compilation_config", None)
        cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
        cudagraph_mode_name = str(cudagraph_mode).rsplit(".", 1)[-1].upper()
        cudagraph_enabled = cudagraph_mode is not None and cudagraph_mode_name not in {
            "NONE",
            "0",
        }
        # Every AR step may replace inputs_embeds with the latent produced by
        # the diffusion/codec side path. A replay that retains that transient
        # address can make paged FlashAttention dereference freed storage. Do
        # not let an inherited/shallow-merged deploy profile start unsafely.
        if cudagraph_enabled and not bool(
            getattr(compilation_config, "cudagraph_copy_inputs", False)
        ):
            raise ValueError(
                "Native VibeVoice CUDA Graph requires "
                "compilation_config.cudagraph_copy_inputs=true"
            )
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.decoder_config = self.config.decoder_config
        self.model_path = vllm_config.model_config.model
        self.model_revision = str(getattr(vllm_config.model_config, "revision", None) or "main")
        local_model_path = Path(str(self.model_path)).expanduser()
        if local_model_path.is_dir() and (
            local_model_path / "model.safetensors.index.json"
        ).is_file():
            report = validate_vibevoice_checkpoint(
                local_model_path,
                inspect_headers=False,
            )
            logger.info(
                "VibeVoice checkpoint preflight complete: tensors=%d shards=%d bytes=%s",
                report.tensor_count,
                report.shard_count,
                report.total_size,
            )
        self.model = _VibeVoiceNativeModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = self.model.language_model.embed_tokens
        self.logits_processor = LogitsProcessor(int(self.decoder_config.vocab_size))
        self.make_empty_intermediate_tensors = self.model.language_model.make_empty_intermediate_tensors

        tokenizer_path = getattr(vllm_config.model_config, "tokenizer", None) or "Qwen/Qwen2.5-1.5B"
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
        self._special = configure_vibevoice_tokenizer(self._tokenizer)
        valid_ids = [
            self._special.speech_start_id,
            self._special.speech_end_id,
            self._special.speech_diffusion_id,
            self._special.eos_id,
        ]
        if self._special.bos_id is not None:
            valid_ids.append(self._special.bos_id)
        valid_ids = sorted(set(valid_ids))
        self.register_buffer(
            "_valid_token_ids",
            torch.tensor(valid_ids, dtype=torch.long),
            persistent=False,
        )
        valid_mask = torch.zeros(int(self.decoder_config.vocab_size), dtype=torch.bool)
        valid_mask[self._valid_token_ids] = True
        self.register_buffer("_invalid_token_mask", ~valid_mask, persistent=False)

        self._batch_req_ids: list[str] = []
        self._step_audio: list[torch.Tensor] = []
        self._next_cache_slot = 0
        self._free_cache_slots: list[int] = []
        self._states_by_req: dict[str, VibeVoiceRequestState] = {}
        self._generators_by_req: dict[tuple[str, str], torch.Generator] = {}
        self._deferred_cleanup_ids: set[str] = set()
        self._conditioning_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._conditioning_cache_limit = max(
            0, int(os.getenv("VLLM_OMNI_VIBEVOICE_CONDITIONING_CACHE_SIZE", "128"))
        )
        self._log_token_trace = os.getenv(
            "VLLM_OMNI_VIBEVOICE_LOG_TOKEN_TRACE", "0"
        ).lower() not in {"0", "false", "off", "no"}
        self._token_trace_limit = max(
            0, int(os.getenv("VLLM_OMNI_VIBEVOICE_TOKEN_TRACE_LIMIT", "64"))
        )
        self._noise_pool_chunk_steps = max(
            1, int(os.getenv("VLLM_OMNI_VIBEVOICE_NOISE_POOL_STEPS", "256"))
        )
        self._noise_pool: torch.Tensor | None = None
        self._metrics = VibeVoiceMetrics()
        self._scheduler_prompt_lengths: dict[str, int] = {}
        self._scheduler_prefill_cycles: dict[str, int] = {}
        self._compile_diffusion_after_load = os.getenv(
            "VLLM_OMNI_VIBEVOICE_COMPILE_DIFFUSION", "1"
        ).lower() not in {"0", "false", "off", "no"}
        self._compile_codec_after_load = os.getenv(
            "VLLM_OMNI_VIBEVOICE_COMPILE_CODEC", "1"
        ).lower() not in {"0", "false", "off", "no"}
        self._diffusion_compiled = False
        self._codec_compiled = False
        self._side_compile_attempted = False
        self._eager_side_modules: dict[str, nn.Module] = {}
        self._cuda_graph_captured = False
        self._negative_cfg_graph = NegativeCFGGraphRunner(
            self.model.negative_language_model,
            layer_marker="negative_language_model",
            tensor_parallel_size=int(
                getattr(
                    getattr(self.vllm_config, "parallel_config", None),
                    "tensor_parallel_size",
                    1,
                )
            ),
        )
        self.gpu_resident_buffer_keys: set[tuple[str, ...]] = {
            (_STATE_KEY, "last_positive_hidden"),
            (_STATE_KEY, "negative_input_embedding"),
            (_STATE_KEY, "prompt_embeddings"),
        }

    def _allocate_cache_slot(self) -> int:
        if self._free_cache_slots:
            return self._free_cache_slots.pop()
        slot = self._next_cache_slot
        self._next_cache_slot += 1
        return slot

    def _request_generator(
        self,
        state: VibeVoiceRequestState,
        device: torch.device,
    ) -> torch.Generator:
        generator_key = (state.request_id, str(device))
        generator = self._generators_by_req.get(generator_key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(state.rng_seed))
            self._generators_by_req[generator_key] = generator
        return generator

    def _request_state(self, info: dict[str, Any], request_id: str) -> tuple[VibeVoiceRequestState, dict[str, Any]]:
        native = info.setdefault(_STATE_KEY, {})
        raw_state = native.get("state")
        if isinstance(raw_state, VibeVoiceRequestState):
            state = raw_state
        elif isinstance(raw_state, dict):
            state = VibeVoiceRequestState.from_dict(raw_state)
        else:
            seed = int(_pick(info, "seed", _pick(info, "_omni_seed", 0)) or 0)
            state = VibeVoiceRequestState(
                request_id=request_id,
                cfg_scale=float(_pick(info, "cfg_scale", _DEFAULT_CFG_SCALE)),
                ddpm_steps=int(_pick(info, "ddpm_steps", _DEFAULT_DDPM_STEPS)),
                rng_seed=seed,
            )
        native["state"] = state
        self._states_by_req[request_id] = state
        self._update_state_metrics()
        return state, native

    def _update_state_metrics(self) -> None:
        block_size = int(self.vllm_config.cache_config.block_size)
        negative_tokens = sum(
            item.negative_num_tokens for item in self._states_by_req.values()
        )
        negative_blocks = sum(
            (item.negative_num_tokens + block_size - 1) // block_size
            for item in self._states_by_req.values()
            if item.negative_num_tokens > 0
        )
        self._metrics.set_state(
            active_requests=len(self._states_by_req),
            negative_tokens=negative_tokens,
            negative_blocks=negative_blocks,
        )

    def _reset_codec_cache(self, slots: Sequence[int]) -> None:
        if not slots:
            return
        device = _module_device(self.model.acoustic_tokenizer)
        indices = torch.tensor(list(slots), device=device, dtype=torch.long)
        self.model.acoustic_cache.set_to_zero(indices)
        self.model.semantic_cache.set_to_zero(indices)

    def _ensure_request_slot(self, state: VibeVoiceRequestState) -> int:
        if state.cache_slot < 0:
            state.cache_slot = self._allocate_cache_slot()
        capacity = int(self.model.acoustic_cache.capacity)
        if state.cache_slot >= capacity:
            raise RuntimeError(
                "VibeVoice active request slots exceeded scheduler max_num_seqs; "
                f"slot={state.cache_slot} capacity={capacity}"
            )
        return state.cache_slot

    def _ensure_noise_pool(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        expected_shape = (
            int(self.model.acoustic_cache.capacity),
            self._noise_pool_chunk_steps,
            2,
            int(self.config.acoustic_vae_dim),
        )
        if (
            self._noise_pool is None
            or self._noise_pool.device != device
            or self._noise_pool.dtype != dtype
            or tuple(self._noise_pool.shape) != expected_shape
        ):
            self._noise_pool = torch.empty(expected_shape, device=device, dtype=dtype)
        return self._noise_pool

    def _refill_request_noise(self, state: VibeVoiceRequestState, pool: torch.Tensor) -> None:
        slot = self._ensure_request_slot(state)
        generator = self._request_generator(state, pool.device)
        torch.randn(
            pool[slot].shape,
            generator=generator,
            device=pool.device,
            dtype=pool.dtype,
            out=pool[slot],
        )
        state.rng_offset += int(pool[slot].numel())
        state.noise_pool_position = 0
        state.noise_pool_valid = self._noise_pool_chunk_steps

    def _cleanup_requests(self) -> None:
        if not self._deferred_cleanup_ids:
            return
        slots: list[int] = []
        for request_id in self._deferred_cleanup_ids:
            state = self._states_by_req.pop(request_id, None)
            for key in [key for key in self._generators_by_req if key[0] == request_id]:
                self._generators_by_req.pop(key, None)
            if state is None or state.cache_slot < 0:
                continue
            slots.append(state.cache_slot)
            self._free_cache_slots.append(state.cache_slot)
        self._reset_codec_cache(slots)
        self._metrics.cleanup(len(self._deferred_cleanup_ids))
        self._update_state_metrics()
        self._deferred_cleanup_ids.clear()

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, Any]]:
        return [
            {
                "text": "Speaker 1: hello",
                "disable_prefill": True,
                "_is_dummy": True,
                "request_id": f"dummy-{index}",
            }
            for index in range(num_reqs)
        ]

    def _prepare_reference_embeddings(
        self,
        references: Any,
        speech_input_mask: torch.Tensor,
        *,
        state: VibeVoiceRequestState,
        device: torch.device,
        dtype: torch.dtype,
        references_are_canonical: bool = False,
    ) -> torch.Tensor | None:
        if not references:
            return None
        if isinstance(references, tuple) and len(references) == 2:
            references = [references]
        waveforms: list[np.ndarray] = []
        cache_keys: list[str] = []
        for speaker_index, (waveform, sample_rate) in enumerate(references):
            if references_are_canonical:
                if int(sample_rate) != SAMPLE_RATE:
                    raise ValueError("Canonical VibeVoice references must be 24 kHz")
                normalized = np.asarray(waveform, dtype=np.float32).reshape(-1)
            else:
                normalized = resample_and_normalize_reference(waveform, int(sample_rate))
            waveforms.append(normalized)
            digest = hashlib.sha256(normalized.tobytes()).hexdigest()
            cache_keys.append(
                f"v1:{self.model_path}:{self.model_revision}:{speaker_index}:{digest}"
            )

        # Cache deterministic encoder means, not sampled/connected embeddings.
        # Sampling stays per request so seed parity and RNG offsets do not
        # change when a conditioning-cache hit occurs.
        means_by_speaker: list[torch.Tensor | None] = [None] * len(waveforms)
        missing: list[int] = []
        for index, cache_key in enumerate(cache_keys):
            cached = self._conditioning_cache.get(cache_key)
            if cached is None:
                self._metrics.conditioning_cache(False)
                missing.append(index)
                continue
            self._metrics.conditioning_cache(True)
            self._conditioning_cache.move_to_end(cache_key)
            means_by_speaker[index] = cached.to(device=device, dtype=dtype, non_blocking=True)

        if missing:
            max_samples = max(len(waveforms[index]) for index in missing)
            padded = torch.zeros((len(missing), max_samples), device=device, dtype=dtype)
            frame_lengths: list[int] = []
            for row, speaker_index in enumerate(missing):
                waveform = waveforms[speaker_index]
                padded[row, : len(waveform)] = torch.from_numpy(waveform).to(device=device, dtype=dtype)
                frame_lengths.append((len(waveform) + 3199) // 3200)

            encoded = self.model.acoustic_tokenizer.encode(padded.unsqueeze(1))
            for row, (speaker_index, frame_length) in enumerate(zip(missing, frame_lengths, strict=True)):
                value = encoded.mean[row, : min(frame_length, encoded.mean.shape[1])]
                means_by_speaker[speaker_index] = value
                if self._conditioning_cache_limit:
                    self._conditioning_cache[cache_keys[speaker_index]] = value.detach().to("cpu")
                    self._conditioning_cache.move_to_end(cache_keys[speaker_index])
                    while len(self._conditioning_cache) > self._conditioning_cache_limit:
                        self._conditioning_cache.popitem(last=False)

        means = [value for value in means_by_speaker if value is not None]
        max_frames = max(value.shape[0] for value in means)
        latent_size = int(means[0].shape[-1])
        padded_means = torch.zeros(
            (len(means), max_frames, latent_size),
            device=device,
            dtype=dtype,
        )
        for row, value in enumerate(means):
            padded_means[row, : value.shape[0]] = value

        generator = self._request_generator(state, device)
        dist_type = str(self.model.acoustic_tokenizer.std_dist_type)
        fixed_std = self.model.acoustic_tokenizer.fix_std.to(device=device, dtype=dtype)
        if dist_type == "gaussian":
            row_std = torch.randn(
                (len(means),), generator=generator, device=device, dtype=dtype
            ) * (fixed_std / 0.8)
            noise = torch.randn(
                padded_means.shape, generator=generator, device=device, dtype=dtype
            )
            sampled = padded_means + row_std[:, None, None] * noise
            state.rng_offset += len(means) + noise.numel()
        elif dist_type == "fix":
            noise = torch.randn(
                padded_means.shape, generator=generator, device=device, dtype=dtype
            )
            sampled = padded_means + fixed_std * noise
            state.rng_offset += noise.numel()
        else:
            sampled = padded_means
        sampled = (sampled + self.model.speech_bias_factor) * self.model.speech_scaling_factor
        connected_batch = self.model.acoustic_connector(sampled)
        connected = torch.cat(
            [connected_batch[row, : value.shape[0]] for row, value in enumerate(means)],
            dim=0,
        )
        expected = int(speech_input_mask.sum().item())
        if connected.shape[0] != expected:
            raise RuntimeError(
                f"VibeVoice prompt/reference frame mismatch: prompt={expected}, codec={connected.shape[0]}"
            )
        return connected

    @torch.inference_mode()
    def preprocess_batch(
        self,
        *,
        req_ids: list[str],
        model_intermediate_buffer: dict[str, dict[str, Any]],
        device: torch.device,
    ) -> None:
        embedding_dtype = getattr(self.vllm_config.model_config, "dtype", torch.bfloat16)
        for request_id in req_ids:
            info = model_intermediate_buffer.get(request_id)
            if not isinstance(info, dict):
                continue
            state, native = self._request_state(info, request_id)
            if isinstance(native.get("prompt_token_ids"), torch.Tensor):
                continue
            script = str(_pick(info, "text", "") or "").strip()
            references = None if bool(_pick(info, "disable_prefill", False)) else _pick(info, "ref_audio_data")
            reference_lengths: list[int] = []
            canonical_references: list[tuple[np.ndarray, int]] = []
            if references:
                ref_items = [references] if isinstance(references, tuple) and len(references) == 2 else references
                for waveform, sample_rate in ref_items:
                    normalized = resample_and_normalize_reference(waveform, int(sample_rate))
                    canonical_references.append((normalized, SAMPLE_RATE))
                    reference_lengths.append(len(normalized))
            prompt_ids, speech_mask, _ = build_prompt_token_ids(
                self._tokenizer,
                script,
                reference_lengths,
            )
            ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
            embeddings = self.embed_input_ids(ids).to(dtype=embedding_dtype)
            speech_mask_tensor = torch.tensor(speech_mask, dtype=torch.bool, device=device)
            if references:
                reference_embeddings = self._prepare_reference_embeddings(
                    canonical_references,
                    speech_mask_tensor,
                    state=state,
                    device=device,
                    dtype=embedding_dtype,
                    references_are_canonical=True,
                )
                if reference_embeddings is not None:
                    embeddings[speech_mask_tensor] = reference_embeddings
            self._ensure_request_slot(state)
            noise_pool = self._ensure_noise_pool(device=device, dtype=embedding_dtype)
            self._refill_request_noise(state, noise_pool)
            state.positive_num_tokens = 0
            state.max_generation_steps = int(
                min(
                    int(_pick(info, "max_new_tokens", self.decoder_config.max_position_embeddings)),
                    float(_pick(info, "max_length_times", 2.0)) * len(prompt_ids),
                )
            )
            native["prompt_token_ids"] = ids.detach().to("cpu")
            native["prompt_embeddings"] = embeddings.detach()
            native["prefill_offset"] = 0
            native["state"] = state

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        del input_embeds
        request_id = str(info.get("request_id") or info.get("global_request_id") or "unknown")
        state, native = self._request_state(info, request_id)
        is_prefill = bool(info.get("_omni_is_prefill", input_ids.numel() > 1))
        if is_prefill:
            prompt_ids = native.get("prompt_token_ids")
            prompt_embeddings = native.get("prompt_embeddings")
            if not isinstance(prompt_ids, torch.Tensor) or not isinstance(prompt_embeddings, torch.Tensor):
                raise RuntimeError("VibeVoice native prompt was not initialized by preprocess_batch")
            offset = max(0, int(info.get("_omni_num_computed_tokens", native.get("prefill_offset", 0))))
            span = int(input_ids.numel())
            ids = prompt_ids[offset : offset + span].to(device=input_ids.device)
            embeds = prompt_embeddings[offset : offset + span].to(
                device=input_ids.device,
                dtype=getattr(self.vllm_config.model_config, "dtype", torch.bfloat16),
            )
            if ids.shape[0] != span or embeds.shape[0] != span:
                raise RuntimeError(
                    f"VibeVoice scheduled prompt slice [{offset}:{offset + span}] exceeds prompt length "
                    f"{prompt_ids.shape[0]}"
                )
            native["prefill_offset"] = offset + span
            native["state"] = state
            return ids, embeds, {_STATE_KEY: native}

        state.phase = VibeVoicePhase.SPEECH if state.phase == VibeVoicePhase.PREFILL else state.phase
        state.generated_steps += int(input_ids.numel())
        native["state"] = state
        return input_ids, self.embed_input_ids(input_ids), {_STATE_KEY: native}

    def build_auxiliary_attention_streams(
        self,
        *,
        req_ids: list[str],
        input_ids: torch.Tensor,
        request_token_spans: list[tuple[int, int]],
        model_intermediate_buffer: dict[str, dict[str, Any]],
    ) -> tuple[AuxiliaryAttentionStreamSpec, ...]:
        sequences: list[AuxiliarySequenceState] = []
        flat_ids = input_ids.reshape(-1)
        for request_id, (start, end) in zip(req_ids, request_token_spans, strict=False):
            info = model_intermediate_buffer.get(request_id, {})
            state, _ = self._request_state(info, request_id)
            token = _scheduled_single_token(info, flat_ids, start, end)
            active = token == self._special.speech_diffusion_id and not state.finished
            if active and state.negative_block_anchor < 0:
                positive_write_position = int(info.get("_omni_num_computed_tokens", 0))
                state.negative_block_anchor = positive_write_position // int(
                    self.vllm_config.cache_config.block_size
                )
            query_len = max(1, end - start)
            logical_length = state.negative_num_tokens + 1 if active else max(query_len, state.negative_num_tokens)
            sequences.append(
                AuxiliarySequenceState(
                    sequence_length=logical_length,
                    write_position=state.negative_num_tokens if active else None,
                    active=active,
                    block_table_start=max(0, state.negative_block_anchor),
                )
            )
        if not any(sequence.active for sequence in sequences):
            return ()
        return (
            AuxiliaryAttentionStreamSpec(
                name="negative_cfg",
                layer_name_marker="negative_language_model",
                sequences=tuple(sequences),
                compact_active=True,
            ),
        )

    def _request_noise(
        self,
        states: list[VibeVoiceRequestState],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pool = self._ensure_noise_pool(device=device, dtype=dtype)
        for state in states:
            if state.noise_pool_position >= state.noise_pool_valid:
                # The default 256-step pool keeps this off the ordinary decode
                # hot path. Refill remains per request to preserve each
                # request's independent Philox stream across batch reorder.
                self._refill_request_noise(state, pool)
        slots = torch.tensor(
            [state.cache_slot for state in states],
            device=device,
            dtype=torch.long,
        )
        positions = torch.tensor(
            [state.noise_pool_position for state in states],
            device=device,
            dtype=torch.long,
        )
        pairs = pool[slots, positions]
        for state in states:
            state.noise_pool_position += 1
        # Preserve the released loop's 2*B draw and positive/negative row
        # layout; the solver replaces the second half after its first update.
        return torch.cat([pairs[:, 0], pairs[:, 1]], dim=0)

    def _sample_speech_tokens(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        states: list[VibeVoiceRequestState],
    ) -> torch.Tensor:
        if not states:
            return positive.new_empty((0, int(self.config.acoustic_vae_dim)))
        step_counts = {state.ddpm_steps for state in states}
        if len(step_counts) != 1:
            # Caller buckets by DDPM count; this guards accidental mixed use.
            raise ValueError(f"Mixed DDPM step counts in one diffusion batch: {sorted(step_counts)}")
        self.model.noise_scheduler.set_timesteps(next(iter(step_counts)))
        condition = torch.cat([positive, negative], dim=0).to(_module_device(self.model.prediction_head))
        speech = self._request_noise(states, device=condition.device, dtype=condition.dtype)
        cfg = torch.tensor(
            [state.cfg_scale for state in states],
            device=condition.device,
            dtype=condition.dtype,
        ).unsqueeze(1)
        batch_size = len(states)
        for timestep in self.model.noise_scheduler.timesteps:
            half = speech[:batch_size]
            combined = torch.cat([half, half], dim=0)
            timestep_batch = timestep.repeat(combined.shape[0]).to(combined)
            eps = self.model.prediction_head(combined, timestep_batch, condition)
            conditional_eps, unconditional_eps = eps.split(batch_size, dim=0)
            guided = unconditional_eps + cfg * (conditional_eps - unconditional_eps)
            speech = self.model.noise_scheduler.step(
                torch.cat([guided, guided], dim=0), timestep, speech
            ).prev_sample
        return speech[:batch_size]

    def _run_diffusion_sidepath(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        request_token_spans: list[tuple[int, int]],
        infos: list[dict[str, Any]],
    ) -> torch.Tensor:
        flat_ids = input_ids.reshape(-1)
        active_rows: list[int] = []
        active_info_indices: list[int] = []
        states: list[VibeVoiceRequestState] = []
        positive_conditions: list[torch.Tensor] = []
        self._step_audio = [torch.empty(0, dtype=torch.float32) for _ in infos]

        for info_index, (info, (start, end)) in enumerate(zip(infos, request_token_spans, strict=False)):
            request_id = str(info.get("request_id") or info.get("global_request_id") or info_index)
            state, native = self._request_state(info, request_id)
            if end - start != 1:
                native["state"] = state
                continue
            token = _scheduled_single_token(info, flat_ids, start, end)
            if self._log_token_trace and state.generated_steps <= self._token_trace_limit:
                logger.info(
                    "VIBEVOICE_TOKEN_TRACE request=%s step=%d token=%d "
                    "phase=%s speech_start=%d speech_diffusion=%d speech_end=%d eos=%d",
                    request_id,
                    state.generated_steps,
                    token,
                    state.phase.value,
                    self._special.speech_start_id,
                    self._special.speech_diffusion_id,
                    self._special.speech_end_id,
                    self._special.eos_id,
                )
            if token == self._special.speech_start_id:
                state.phase = VibeVoicePhase.SPEECH
                state.negative_num_tokens = 0
                state.negative_block_anchor = -1
                native.pop("negative_input_embedding", None)
                if state.cache_slot >= 0:
                    self._reset_codec_cache([state.cache_slot])
            elif token == self._special.speech_end_id:
                state.phase = VibeVoicePhase.TEXT
                state.negative_block_anchor = -1
                native.pop("negative_input_embedding", None)
                if state.cache_slot >= 0:
                    self._reset_codec_cache([state.cache_slot])
            elif token == self._special.eos_id:
                state.phase = VibeVoicePhase.FINISHED
                state.finished = True
            elif token == self._special.speech_diffusion_id and not state.finished:
                self._ensure_request_slot(state)
                last_hidden = native.get("last_positive_hidden")
                if not isinstance(last_hidden, torch.Tensor):
                    raise RuntimeError(
                        f"Missing positive hidden state before VibeVoice diffusion for request {request_id}"
                    )
                active_rows.append(start)
                active_info_indices.append(info_index)
                states.append(state)
                positive_conditions.append(last_hidden.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))
            native["state"] = state

        if not active_rows:
            return inputs_embeds

        self._metrics.observe_batch("negative_cfg", len(active_rows))
        self._metrics.observe_batch("diffusion", len(active_rows))
        self._metrics.observe_batch("acoustic_decode", len(active_rows))
        self._metrics.observe_batch("semantic_encode", len(active_rows))

        # Compact only active CFG rows. Auxiliary per-layer metadata carries a
        # matching compact query_start_loc/block table, so inactive requests do
        # not execute a fake negative forward or mutate their logical cache.
        negative_ids = torch.tensor(
            [
                self._special.speech_start_id
                if state.negative_num_tokens == 0
                else self._special.speech_diffusion_id
                for state in states
            ],
            device=flat_ids.device,
            dtype=flat_ids.dtype,
        )
        negative_positions = torch.tensor(
            [state.negative_num_tokens for state in states],
            device=flat_ids.device,
            dtype=torch.long,
        )
        negative_embeddings_by_row: list[torch.Tensor] = []
        for info_index, state in zip(active_info_indices, states, strict=True):
            native = infos[info_index][_STATE_KEY]
            prior_embedding = native.get("negative_input_embedding")
            if state.negative_num_tokens == 0:
                prior_embedding = self.model.negative_language_model.embed_input_ids(
                    negative_ids[len(negative_embeddings_by_row) : len(negative_embeddings_by_row) + 1]
                )[0]
            if not isinstance(prior_embedding, torch.Tensor):
                raise RuntimeError(
                    f"Missing negative CFG input embedding for request {state.request_id}"
                )
            negative_embeddings_by_row.append(
                prior_embedding.to(device=flat_ids.device, dtype=inputs_embeds.dtype)
            )
        negative_embeddings = torch.stack(negative_embeddings_by_row, dim=0)
        with self._metrics.timer("negative_cfg", flat_ids.device):
            # The auxiliary runner bypasses Qwen2Model's runner-owned wrapper,
            # compiles without Inductor graph trees, and owns a separate set of
            # CUDA graphs for compact CFG batch buckets. It keeps the current
            # auxiliary PagedAttention ForwardContext and safely falls back to
            # compiled/eager execution for unsupported metadata shapes.
            negative_hidden = self._negative_cfg_graph.run(
                negative_ids,
                negative_positions,
                negative_embeddings,
            )
            self._metrics.negative_graph(self._negative_cfg_graph.last_execution)
        if isinstance(negative_hidden, IntermediateTensors):
            raise RuntimeError("VibeVoice negative CFG is unsupported with pipeline parallelism")

        row_tensor = torch.tensor(active_rows, device=inputs_embeds.device, dtype=torch.long)
        positive = torch.stack(positive_conditions, dim=0)
        negative = negative_hidden
        speech_latents = torch.empty(
            (len(states), int(self.config.acoustic_vae_dim)),
            device=positive.device,
            dtype=positive.dtype,
        )
        buckets: dict[int, list[int]] = {}
        for index, state in enumerate(states):
            buckets.setdefault(state.ddpm_steps, []).append(index)
        with self._metrics.timer("diffusion", positive.device):
            for indices in buckets.values():
                index_tensor = torch.tensor(indices, device=positive.device, dtype=torch.long)
                bucket_states = [states[index] for index in indices]
                speech_latents.index_copy_(
                    0,
                    index_tensor,
                    self._sample_speech_tokens(
                        positive.index_select(0, index_tensor),
                        negative.index_select(0, index_tensor),
                        bucket_states,
                    ),
                )

        scaled = speech_latents / self.model.speech_scaling_factor - self.model.speech_bias_factor
        cache_slots = torch.tensor(
            [state.cache_slot for state in states],
            device=scaled.device,
            dtype=torch.long,
        )
        with self._metrics.timer("acoustic_decode", scaled.device):
            audio = self.model.acoustic_tokenizer.decode(
                scaled.unsqueeze(1),
                cache=self.model.acoustic_cache,
                sample_indices=cache_slots,
                use_cache=True,
            ).clone()
        with self._metrics.timer("semantic_encode", audio.device):
            semantic = self.model.semantic_tokenizer.encode(
                audio,
                cache=self.model.semantic_cache,
                sample_indices=cache_slots,
                use_cache=True,
            ).mean.clone()
        # ``reduce-overhead`` uses CUDA Graph Trees whose returned tensors are
        # backed by reusable static output buffers.  Own each result before
        # invoking the next independently compiled side module; otherwise the
        # next graph is allowed to overwrite a still-live connector output.
        acoustic_embedding = self.model.acoustic_connector(speech_latents.unsqueeze(1)).clone()
        semantic_embedding = self.model.semantic_connector(semantic).clone()
        next_embeddings = (acoustic_embedding + semantic_embedding).squeeze(1)
        # ``inputs_embeds`` is the runner-owned decode buffer consumed by the
        # positive CUDA Graph. Keep the update in place so both vLLM's explicit
        # copy-input mode and third-party runners that reuse this allocation see
        # the freshly generated continuous embedding.
        inputs_embeds.index_copy_(0, row_tensor, next_embeddings)

        for local_index, info_index in enumerate(active_info_indices):
            states[local_index].negative_num_tokens += 1
            infos[info_index][_STATE_KEY]["state"] = states[local_index]
            infos[info_index][_STATE_KEY]["negative_input_embedding"] = next_embeddings[
                local_index
            ].detach()
            self._step_audio[info_index] = audio[local_index].reshape(-1).float()
        self._update_state_metrics()
        self._metrics.generated(
            speech_tokens=len(states),
            audio_samples=int(audio.shape[0] * audio.shape[-1]),
        )
        return inputs_embeds

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.language_model.embed_input_ids(input_ids)

    def prepare_omni_forward_inputs(
        self,
        *,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        request_token_spans: list[tuple[int, int]],
        model_intermediate_buffer: list[dict[str, Any]],
    ) -> torch.Tensor | None:
        """Run dynamic CFG/diffusion work outside positive-Qwen graph replay."""
        if input_ids is None:
            raise RuntimeError("Native VibeVoice requires scheduled input token ids")
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        infos = list(model_intermediate_buffer or [])
        if not infos:
            self._step_audio = []
            return inputs_embeds
        self._metrics.observe_batch("positive_qwen", len(infos))
        return self._run_diffusion_sidepath(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            request_token_spans=request_token_spans,
            infos=infos,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        request_token_spans: list[tuple[int, int]] | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if input_ids is None:
            raise RuntimeError("Native VibeVoice requires scheduled input token ids")
        del model_intermediate_buffer, request_token_spans
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        with self._metrics.timer("positive_qwen", input_ids.device):
            return self.model.language_model(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        del sampling_metadata
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        # The released inference loop can only choose four special tokens (and
        # BOS when defined).  In the default unquantized TP=1 profile, avoid a
        # 151,936 x 1,536 full-vocabulary projection only to mask virtually all
        # of it immediately afterwards.  The selected rows are the tied input
        # embeddings, so this is algebraically the same projection.  Preserve
        # vLLM's standard distributed/quantized head path for TP and optional
        # quantization profiles where sharding or a custom kernel owns the
        # projection semantics.
        parallel_config = getattr(self.vllm_config, "parallel_config", None)
        tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1))
        if tp_size == 1 and getattr(self.vllm_config, "quant_config", None) is None:
            selected_weights = self.lm_head.weight.index_select(0, self._valid_token_ids)
            selected_logits = torch.matmul(hidden_states, selected_weights.transpose(0, 1))
            logits = selected_logits.new_full(
                (*selected_logits.shape[:-1], int(self.decoder_config.vocab_size)),
                float("-inf"),
            )
            logits.index_copy_(-1, self._valid_token_ids, selected_logits)
            return logits
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None:
            logits = logits.masked_fill(self._invalid_token_mask, float("-inf"))
        return logits

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | IntermediateTensors | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        infos = kwargs.get("model_intermediate_buffer") or []
        outputs = list(self._step_audio)
        if len(outputs) < len(infos):
            outputs.extend(torch.empty(0, dtype=torch.float32) for _ in range(len(infos) - len(outputs)))
        sample_rate = torch.tensor(SAMPLE_RATE, dtype=torch.int32)
        result = OmniOutput(
            text_hidden_states=model_outputs,
            multimodal_outputs={
                "model_outputs": outputs,
                "sr": [sample_rate] * len(outputs),
            },
        )
        return result

    def postprocess(self, hidden_states: torch.Tensor, **info: Any) -> dict[str, Any]:
        if hidden_states.numel() == 0:
            return {}
        native = dict(info.get(_STATE_KEY, {}))
        native["last_positive_hidden"] = hidden_states[-1].detach()
        raw_state = native.get("state")
        if isinstance(raw_state, VibeVoiceRequestState):
            raw_state.positive_num_tokens += int(hidden_states.shape[0])
            native["state"] = raw_state
        return {_STATE_KEY: native}

    def set_batch_req_ids(self, req_ids: Sequence[str]) -> None:
        self._batch_req_ids = list(req_ids)

    def observe_scheduler_step(self, scheduler_output: Any) -> None:
        """Record scheduler evidence without importing scheduler internals."""
        scheduled = dict(getattr(scheduler_output, "num_scheduled_tokens", {}) or {})
        new_reqs_value = getattr(scheduler_output, "scheduled_new_reqs", None)
        new_reqs = list(new_reqs_value) if new_reqs_value is not None else []
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        cached_ids_value = getattr(cached, "req_ids", None)
        cached_ids = list(cached_ids_value) if cached_ids_value is not None else []
        cached_computed_value = getattr(cached, "num_computed_tokens", None)
        cached_computed = (
            list(cached_computed_value)
            if cached_computed_value is not None
            else []
        )
        computed_before: dict[str, int] = {}

        for request in new_reqs:
            request_id = str(getattr(request, "req_id"))
            prompt_ids = getattr(request, "prompt_token_ids", None)
            prompt_length = len(prompt_ids) if prompt_ids is not None else 0
            computed = int(getattr(request, "num_computed_tokens", 0) or 0)
            self._scheduler_prompt_lengths[request_id] = prompt_length
            self._scheduler_prefill_cycles.setdefault(request_id, 0)
            computed_before[request_id] = computed
            self._metrics.prefix_cache(computed_tokens=computed)

        for index, request_id_value in enumerate(cached_ids):
            request_id = str(request_id_value)
            computed_before[request_id] = (
                int(cached_computed[index]) if index < len(cached_computed) else 0
            )

        self._metrics.scheduler_step(
            batch_size=len(scheduled),
            new_requests=len(new_reqs),
            cached_requests=len(cached_ids),
        )
        for request_id in scheduled:
            prompt_length = self._scheduler_prompt_lengths.get(str(request_id), 0)
            if prompt_length and computed_before.get(str(request_id), prompt_length) < prompt_length:
                self._scheduler_prefill_cycles[str(request_id)] = (
                    self._scheduler_prefill_cycles.get(str(request_id), 0) + 1
                )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        finished = {str(request_id) for request_id in finished_req_ids}
        for request_id in finished:
            self._metrics.finish_prefill(
                self._scheduler_prefill_cycles.pop(request_id, 0)
            )
            self._scheduler_prompt_lengths.pop(request_id, None)
        self._deferred_cleanup_ids.update(finished)
        self._batch_req_ids = [request_id for request_id in self._batch_req_ids if request_id not in finished]
        # Scheduler finish notifications belong to outputs materialized in the
        # preceding cycle. Cleanup must happen here: an abort/final EOS can
        # produce a zero-token scheduler tick which never calls forward() or
        # make_omni_output(). Deferring until then leaks KV-adjacent codec state.
        self._cleanup_requests()

    def _compile_side_modules(self) -> None:
        if self._side_compile_attempted or not torch.cuda.is_available():
            return
        self._side_compile_attempted = True
        # Stateful codec caches are mutated in place. The ordinary Inductor
        # mode is the portable default; profiles may opt into graph-tree modes
        # only after validating them on their target hardware/backend.
        mode = os.getenv("VLLM_OMNI_VIBEVOICE_COMPILE_MODE", "default")
        tensor_parallel_size = int(
            getattr(self.vllm_config.parallel_config, "tensor_parallel_size", 1)
        )
        if tensor_parallel_size > 1 and mode == "reduce-overhead":
            # CUDA Graph Trees own input addresses and are unsafe for these
            # stateful/mutating streaming-cache modules under TP collectives.
            logger.warning(
                "VibeVoice side-module compile mode reduce-overhead is unsafe "
                "with tensor_parallel_size=%d; forcing mode=default",
                tensor_parallel_size,
            )
            mode = "default"
        if self._compile_diffusion_after_load:
            try:
                self._eager_side_modules["prediction_head"] = self.model.prediction_head
                self.model.prediction_head = torch.compile(
                    self.model.prediction_head,
                    mode=mode,
                    dynamic=True,
                    fullgraph=False,
                )
                self._diffusion_compiled = True
                logger.info("VibeVoice native diffusion head compile configured: mode=%s", mode)
            except Exception as exc:  # pragma: no cover - driver/compiler dependent
                self._restore_eager_side_modules()
                logger.warning("VibeVoice native diffusion compile unavailable: %s", exc)
        if self._compile_codec_after_load:
            try:
                # Wrap only the numerically heavy submodules after checkpoint
                # loading. This preserves strict state-dict names and leaves
                # the slot-indexed streaming-cache orchestration in eager
                # Python while Dynamo compiles its convolutional GPU regions.
                self._eager_side_modules.update(
                    {
                        "acoustic_decoder": self.model.acoustic_tokenizer.decoder,
                        "semantic_encoder": self.model.semantic_tokenizer.encoder,
                        "acoustic_connector": self.model.acoustic_connector,
                        "semantic_connector": self.model.semantic_connector,
                    }
                )
                self.model.acoustic_tokenizer.decoder = torch.compile(
                    self.model.acoustic_tokenizer.decoder,
                    mode=mode,
                    dynamic=True,
                    fullgraph=False,
                )
                self.model.semantic_tokenizer.encoder = torch.compile(
                    self.model.semantic_tokenizer.encoder,
                    mode=mode,
                    dynamic=True,
                    fullgraph=False,
                )
                self.model.acoustic_connector = torch.compile(
                    self.model.acoustic_connector,
                    mode=mode,
                    dynamic=True,
                    fullgraph=False,
                )
                self.model.semantic_connector = torch.compile(
                    self.model.semantic_connector,
                    mode=mode,
                    dynamic=True,
                    fullgraph=False,
                )
                self._codec_compiled = True
                logger.info("VibeVoice native codec/connector compile configured: mode=%s", mode)
            except Exception as exc:  # pragma: no cover - driver/compiler dependent
                self._restore_eager_side_modules()
                logger.warning("VibeVoice native codec compile unavailable: %s", exc)

    def _restore_eager_side_modules(self) -> None:
        modules = self._eager_side_modules
        if "prediction_head" in modules:
            self.model.prediction_head = modules["prediction_head"]
        if "acoustic_decoder" in modules:
            self.model.acoustic_tokenizer.decoder = modules["acoustic_decoder"]
        if "semantic_encoder" in modules:
            self.model.semantic_tokenizer.encoder = modules["semantic_encoder"]
        if "acoustic_connector" in modules:
            self.model.acoustic_connector = modules["acoustic_connector"]
        if "semantic_connector" in modules:
            self.model.semantic_connector = modules["semantic_connector"]
        self._diffusion_compiled = False
        self._codec_compiled = False

    @torch.inference_mode()
    def profile_side_modules(self) -> None:
        """Materialize and profile the native diffusion/codec peak at startup.

        vLLM's standard memory dummy run exercises the positive language model,
        but its random token IDs do not enter VibeVoice's diffusion state.  This
        hook is called inside the worker's memory-profiling window, before KV
        block count is chosen, so persistent streaming-cache storage and compile
        workspaces are included in the actual VRAM budget.
        """
        if not torch.cuda.is_available():
            return
        batch_size = max(
            1,
            int(
                os.getenv(
                    "VLLM_OMNI_VIBEVOICE_PROFILE_BATCH_SIZE",
                    str(self.vllm_config.scheduler_config.max_num_seqs),
                )
            ),
        )
        batch_size = min(batch_size, int(self.vllm_config.scheduler_config.max_num_seqs))
        device = _module_device(self.model.prediction_head)
        dtype = getattr(self.vllm_config.model_config, "dtype", torch.bfloat16)
        states = [
            VibeVoiceRequestState(
                request_id=f"__vibevoice_profile_{index}",
                cfg_scale=_DEFAULT_CFG_SCALE,
                ddpm_steps=_DEFAULT_DDPM_STEPS,
                rng_seed=index,
                cache_slot=index,
            )
            for index in range(batch_size)
        ]
        conditions = torch.zeros(
            (batch_size, int(self.decoder_config.hidden_size)),
            device=device,
            dtype=dtype,
        )
        slots = torch.arange(batch_size, device=device, dtype=torch.long)

        def run_profile_path() -> None:
            latents = self._sample_speech_tokens(conditions, conditions, states)
            scaled = latents / self.model.speech_scaling_factor - self.model.speech_bias_factor
            audio = self.model.acoustic_tokenizer.decode(
                scaled.unsqueeze(1),
                cache=self.model.acoustic_cache,
                sample_indices=slots,
                use_cache=True,
            ).clone()
            semantic = self.model.semantic_tokenizer.encode(
                audio,
                cache=self.model.semantic_cache,
                sample_indices=slots,
                use_cache=True,
            ).mean.clone()
            acoustic_embedding = self.model.acoustic_connector(latents.unsqueeze(1)).clone()
            semantic_embedding = self.model.semantic_connector(semantic).clone()
            _ = acoustic_embedding + semantic_embedding

        def reset_profile_state(*, discard_cache: bool = False) -> None:
            if discard_cache:
                # A failed CUDA Graph capture can leave tensors whose storage
                # belongs to a reusable graph output pool. Dropping dictionary
                # references is safe; mutating those tensors with index_fill_
                # is not.
                self.model.acoustic_cache.clear()
                self.model.semantic_cache.clear()
            else:
                self._reset_codec_cache(range(batch_size))
            for state in states:
                state.noise_pool_position = 0
                state.noise_pool_valid = 0
                for key in [key for key in self._generators_by_req if key[0] == state.request_id]:
                    self._generators_by_req.pop(key, None)

        # Allocate every streaming-cache tensor eagerly before wrapping the
        # heavy modules. CUDA Graphs then mutate stable cache allocations
        # instead of capturing their initial torch.zeros() as graph outputs.
        run_profile_path()
        current_omni_platform.synchronize()
        reset_profile_state()

        # Weight loading and vLLM's post-load parameter processing have both
        # completed by the time the worker enters memory profiling. Wrap only
        # after the eager cache-allocation pass, while compile workspaces are
        # still charged to vLLM's memory-profiling window.
        self._compile_side_modules()
        self._negative_cfg_graph.configure_compile()

        try:
            run_profile_path()
        except Exception as compile_exc:
            if not (self._diffusion_compiled or self._codec_compiled):
                raise
            logger.warning(
                "VibeVoice side-module compile failed during warmup; retrying eager: %s",
                compile_exc,
            )
            self._restore_eager_side_modules()
            reset_profile_state(discard_cache=True)
            run_profile_path()
        current_omni_platform.synchronize()
        reset_profile_state()
        logger.info(
            "VibeVoice native side-path memory profile complete: batch_size=%d ddpm_steps=%d",
            batch_size,
            _DEFAULT_DDPM_STEPS,
        )

    def runtime_capabilities(self) -> dict[str, Any]:
        def attention_implementations(module: nn.Module) -> list[str]:
            implementations = {
                type(impl).__name__
                for child in module.modules()
                if (impl := getattr(child, "impl", None)) is not None
            }
            return sorted(implementations) or ["vllm_attention_unresolved"]

        positive_attention = attention_implementations(self.model.language_model)
        negative_attention = attention_implementations(self.model.negative_language_model)
        cache_config = getattr(self.vllm_config, "cache_config", None)
        scheduler_config = getattr(self.vllm_config, "scheduler_config", None)
        compilation_config = getattr(self.vllm_config, "compilation_config", None)
        cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
        compilation_mode = str(getattr(compilation_config, "mode", "NONE"))
        positive_qwen_compiled = "NONE" not in compilation_mode.upper()
        negative_qwen_compiled = self._negative_cfg_graph.compile_verified
        return {
            "backend": "vibevoice_native",
            "attention": "vllm_paged_attention",
            "positive_attention_impl": positive_attention,
            "negative_attention_impl": negative_attention,
            "cfg_kv_streams": 2,
            "kv_cache_dtype": str(getattr(cache_config, "cache_dtype", "auto")),
            "kv_block_size": int(getattr(cache_config, "block_size", 0) or 0),
            "continuous_batching": True,
            "chunked_prefill": bool(getattr(scheduler_config, "enable_chunked_prefill", False)),
            "prefix_caching": bool(getattr(cache_config, "enable_prefix_caching", False)),
            "cuda_graph_config": str(cudagraph_mode),
            "cuda_graph_copy_inputs": bool(
                getattr(compilation_config, "cudagraph_copy_inputs", False)
            ),
            "cuda_graph_capture_sizes": list(
                getattr(compilation_config, "cudagraph_capture_sizes", ()) or ()
            ),
            # The ordinary vLLM capture run owns positive Qwen. The speech
            # warmup lazily compiles/captures the compact auxiliary branch.
            "positive_qwen_compiled": positive_qwen_compiled,
            "positive_cuda_graph_captured": self._cuda_graph_captured,
            "negative_qwen_compiled": negative_qwen_compiled,
            "negative_cuda_graph": self._negative_cfg_graph.graph_ready,
            "negative_cuda_graph_requested": self._negative_cfg_graph.graph_requested,
            "negative_cuda_graph_enabled": self._negative_cfg_graph.graph_enabled,
            "negative_cuda_graph_disabled_reason": (
                self._negative_cfg_graph.graph_disabled_reason
            ),
            "negative_cuda_graph_batch_sizes": (
                self._negative_cfg_graph.captured_batch_sizes
            ),
            "negative_cuda_graph_configured_batch_sizes": list(
                self._negative_cfg_graph.capture_batch_sizes
            ),
            "negative_cuda_graph_hits": self._negative_cfg_graph.graph_hits,
            "negative_cuda_graph_misses": self._negative_cfg_graph.graph_misses,
            # This top-level field reports the runner-owned positive-Qwen
            # graph. Auxiliary negative CFG has its own explicit fields and is
            # intentionally compile-only on TP=2.
            "cuda_graph_captured": self._cuda_graph_captured,
            "diffusion_compiled": self._diffusion_compiled,
            "codec_compiled": self._codec_compiled,
            "tensor_parallel_size": int(
                getattr(getattr(self.vllm_config, "parallel_config", None), "tensor_parallel_size", 1)
            ),
            "hf_generate": False,
            "scheduler_evidence": self._metrics.scheduler_snapshot(),
        }

    def on_cuda_graph_capture_complete(self) -> None:
        compilation_config = getattr(self.vllm_config, "compilation_config", None)
        mode = str(getattr(compilation_config, "cudagraph_mode", "NONE"))
        self._cuda_graph_captured = "NONE" not in mode.upper()
        logger.info(
            "VIBEVOICE_CAPABILITIES_AFTER_CAPTURE %s",
            json.dumps(self.runtime_capabilities(), sort_keys=True),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        checkpoint_buffers = {
            "model.speech_scaling_factor": self.model.speech_scaling_factor,
            "model.speech_bias_factor": self.model.speech_bias_factor,
        }
        loaded_buffers: set[str] = set()
        loaded_side_parameters: set[str] = set()
        seen_checkpoint_names: set[str] = set()
        all_parameters = dict(self.named_parameters(remove_duplicate=False))
        native_targets = set(all_parameters) | set(checkpoint_buffers)

        def parameter_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if name in seen_checkpoint_names:
                    raise RuntimeError(f"Duplicate VibeVoice checkpoint tensor: {name}")
                seen_checkpoint_names.add(name)
                buffer = checkpoint_buffers.get(name)
                if buffer is None:
                    target_name = _native_checkpoint_target_name(name)
                    if target_name not in native_targets:
                        raise RuntimeError(
                            f"Unexpected VibeVoice checkpoint tensor {name!r}; "
                            f"mapped native target {target_name!r} does not exist"
                        )
                    if name.startswith("model.language_model."):
                        # Only the Qwen subtree uses packed QKV/gate-up
                        # parameters. Applying its substring mapper to the
                        # diffusion FFN would corrupt gate_proj/up_proj names.
                        yield name, weight
                        continue
                    parameter = all_parameters[target_name]
                    if tuple(parameter.shape) != tuple(weight.shape):
                        raise RuntimeError(
                            f"VibeVoice checkpoint parameter shape mismatch for {name}: "
                            f"expected={tuple(parameter.shape)} actual={tuple(weight.shape)}"
                        )
                    # Use the loader attached by vLLM when present.  Besides
                    # preserving quantized/custom parameter semantics this is
                    # important for device/meta initialization and CPU
                    # offloading; a raw ``parameter.data.copy_`` bypasses all
                    # of those paths.
                    weight_loader = getattr(parameter, "weight_loader", default_weight_loader)
                    weight_loader(parameter, weight)
                    loaded_side_parameters.add(target_name)
                    continue
                if tuple(buffer.shape) != tuple(weight.shape):
                    raise RuntimeError(
                        f"VibeVoice checkpoint buffer shape mismatch for {name}: "
                        f"expected={tuple(buffer.shape)} actual={tuple(weight.shape)}"
                    )
                buffer.copy_(weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded_buffers.add(name)

        loader = AutoWeightsLoader(self, skip_prefixes=["lm_head.", "model.negative_language_model."])
        loaded = loader.load_weights(parameter_weights(), mapper=self.hf_to_vllm_mapper)
        missing_buffers = sorted(checkpoint_buffers.keys() - loaded_buffers)
        if missing_buffers:
            raise RuntimeError(f"VibeVoice checkpoint is missing required buffers: {missing_buffers}")
        loaded.update(loaded_buffers)
        loaded.update(loaded_side_parameters)
        # The negative Qwen tree aliases every positive parameter. Mark aliases
        # loaded so vLLM's strict completeness check does not demand duplicate
        # checkpoint tensors.
        loaded.update(
            name
            for name, _ in self.named_parameters(remove_duplicate=False)
            if name.startswith("model.negative_language_model.") or name.startswith("lm_head.")
        )
        expected = {name for name, _ in self.named_parameters(remove_duplicate=True)}
        missing = sorted(expected - loaded)
        if missing:
            raise RuntimeError(
                "VibeVoice native checkpoint did not initialize every parameter; "
                f"missing={len(missing)}, first={missing[:20]}"
            )
        if not torch.isfinite(self.model.speech_scaling_factor).all() or not torch.isfinite(
            self.model.speech_bias_factor
        ).all():
            raise RuntimeError(
                "VibeVoice checkpoint did not initialize speech_scaling_factor "
                "and speech_bias_factor"
            )
        logger.info(
            "VibeVoice native weight coverage complete: parameters=%d loaded_names=%d",
            len(expected),
            len(loaded),
        )
        logger.info("VIBEVOICE_CAPABILITIES %s", json.dumps(self.runtime_capabilities(), sort_keys=True))
        return loaded


__all__ = ["VibeVoiceNativeForConditionalGeneration"]
