# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from vllm_omni.attention.auxiliary_stream import (
    AuxiliaryAttentionStreamSpec,
    AuxiliarySequenceState,
    apply_auxiliary_attention_streams,
)
from vllm_omni.model_executor.models.vibevoice.checkpoint import (
    _connector_keys,
    _language_model_keys,
    _prediction_head_keys,
    validate_vibevoice_checkpoint,
)
from vllm_omni.model_executor.models.vibevoice.modeling_vibevoice_native import (
    _native_checkpoint_target_name,
)
from vllm_omni.model_executor.models.vibevoice.native_components import (
    VibeVoiceBatchedStreamingCache,
    VibeVoicePhase,
    VibeVoiceRequestState,
)
from vllm_omni.model_executor.models.vibevoice.negative_cfg_graph import (
    NegativeCFGGraphRunner,
    _clone_metadata,
    _copy_metadata_tensors,
    _metadata_signature,
)
from vllm_omni.model_executor.models.vibevoice.prompting import (
    build_prompt_token_ids,
    parse_script,
    reference_frame_count,
    resample_waveform,
)
from vllm_omni.model_executor.models.vibevoice.vendored_dpm_solver import (
    DPMSolverMultistepScheduler,
)
from vllm_omni.model_executor.models.vibevoice.vendored_tokenizer import SConvTranspose1d


class _Tokenizer:
    eos_token_id = 9
    pad_token_id = 0
    bos_token_id = None

    def __init__(self) -> None:
        self.tokens = {
            "<|vision_start|>": 10,
            "<|vision_end|>": 11,
            "<|vision_pad|>": 12,
            "<|image_pad|>": 13,
        }

    def add_special_tokens(self, _: dict) -> None:
        pass

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.tokens[token]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        return [100 + (ord(char) % 17) for char in text]


def test_reference_resample_falls_back_when_torchaudio_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torchaudio", None)
    waveform = torch.linspace(-1.0, 1.0, 8000)

    resampled = resample_waveform(waveform, 8000, 24000)

    assert resampled.shape == (24000,)
    assert torch.isfinite(resampled).all()


def test_negative_cuda_graph_fails_closed_for_tensor_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_OMNI_VIBEVOICE_GRAPH_NEGATIVE_QWEN", "1")

    runner = NegativeCFGGraphRunner(
        torch.nn.Identity(),
        layer_marker="negative_language_model",
        tensor_parallel_size=2,
    )

    assert runner.graph_requested is True
    assert runner.graph_enabled is False
    assert runner.graph_disabled_reason == (
        "manual_negative_graph_unsupported_with_tensor_parallel"
    )


def test_negative_cuda_graph_fails_closed_on_pre_ampere_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_OMNI_VIBEVOICE_GRAPH_NEGATIVE_QWEN", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    runner = NegativeCFGGraphRunner(
        torch.nn.Identity(),
        layer_marker="negative_language_model",
    )

    assert runner.graph_enabled is False
    assert runner.device_capability == (7, 5)
    assert runner.graph_disabled_reason == (
        "manual_negative_graph_requires_compute_capability_80"
    )


def test_negative_cuda_graph_fails_closed_with_triton_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TritonAttentionImpl:
        pass

    model = torch.nn.Sequential(torch.nn.Identity())
    model[0].impl = TritonAttentionImpl()
    monkeypatch.setenv("VLLM_OMNI_VIBEVOICE_GRAPH_NEGATIVE_QWEN", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    runner = NegativeCFGGraphRunner(
        model,
        layer_marker="negative_language_model",
    )

    assert runner.graph_enabled is False
    assert runner.graph_disabled_reason == (
        "manual_negative_graph_unsupported_with_triton_attention"
    )


def test_negative_cuda_graph_remains_available_on_validated_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FlashAttentionImpl:
        pass

    model = torch.nn.Sequential(torch.nn.Identity())
    model[0].impl = FlashAttentionImpl()
    monkeypatch.setenv("VLLM_OMNI_VIBEVOICE_GRAPH_NEGATIVE_QWEN", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    runner = NegativeCFGGraphRunner(
        model,
        layer_marker="negative_language_model",
    )

    assert runner.graph_enabled is True
    assert runner.graph_disabled_reason is None
    assert runner.device_capability == (9, 0)


def test_diffusion_schedule_stays_on_cpu_inside_accelerator_default_context() -> None:
    # vLLM creates model objects under an accelerator/meta default-device
    # context.  The DPM scheduler is not an nn.Module and must explicitly keep
    # its NumPy-interoperable schedule tensors on CPU.
    with torch.device("meta"):
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)

    scheduler.set_timesteps(10)

    assert scheduler.lambda_t.device.type == "cpu"
    assert scheduler.alphas_cumprod.device.type == "cpu"
    assert scheduler.sigmas.device.type == "cpu"
    assert scheduler.timesteps.device.type == "cpu"
    assert len(scheduler.timesteps) == 10


def _contains_subsequence(values: list[int], expected: list[int]) -> bool:
    width = len(expected)
    return any(values[index : index + width] == expected for index in range(len(values) - width + 1))


@dataclass
class _CommonMetadata:
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table_tensor: torch.Tensor
    query_start_loc: torch.Tensor
    max_seq_len: int = 1
    max_query_len: int = 1
    num_actual_tokens: int = 0
    num_reqs: int = 0
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    seq_lens_cpu_upper_bound: torch.Tensor | None = None
    query_start_loc_cpu: torch.Tensor | None = None
    _seq_lens_cpu: torch.Tensor | None = None
    _num_computed_tokens_cpu: torch.Tensor | None = None


@dataclass
class _LayerMetadata:
    common_attn_metadata: _CommonMetadata


@dataclass
class _BackendMetadata:
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    max_seq_len: int = 1
    max_query_len: int = 1
    num_actual_tokens: int = 0
    num_reqs: int = 0
    common_prefix_len: int = 7
    use_cascade: bool = True
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    cu_prefix_query_lens: torch.Tensor | None = None
    prefix_kv_lens: torch.Tensor | None = None
    suffix_kv_lens: torch.Tensor | None = None
    num_decode_reqs: int = 0
    num_prefill_reqs: int = 1


def test_prompt_contains_exact_reference_placeholder_count() -> None:
    ids, mask, special = build_prompt_token_ids(
        _Tokenizer(),
        "Speaker 1: hello",
        [6401],
    )

    assert reference_frame_count(6401) == 3
    assert sum(mask) == 3
    assert ids.count(special.speech_diffusion_id) == 3
    assert ids[-1] == special.speech_start_id


def test_script_speaker_labels_are_normalized_like_reference_processor() -> None:
    assert parse_script("Speaker 1: hello\nSpeaker 2: world") == [
        (0, "hello"),
        (1, "world"),
    ]


def test_prompt_matches_reference_speaker_spacing_and_rejects_unlabelled_lines() -> None:
    tokenizer = _Tokenizer()
    ids, _, _ = build_prompt_token_ids(tokenizer, "Speaker 1: hello")
    expected_line = tokenizer.encode(" Speaker 0: hello\n", add_special_tokens=False)
    incorrect_line = tokenizer.encode(" Speaker 0:hello\n", add_special_tokens=False)

    assert _contains_subsequence(ids, expected_line)
    assert not _contains_subsequence(ids, incorrect_line)
    with pytest.raises(ValueError, match="no speakable text"):
        parse_script("this line has no speaker label")


def test_request_state_round_trip_preserves_cfg_sequence_state() -> None:
    state = VibeVoiceRequestState(
        request_id="r1",
        phase=VibeVoicePhase.SPEECH,
        positive_num_tokens=17,
        negative_num_tokens=3,
        negative_block_anchor=5,
        rng_seed=42,
        rng_offset=7,
        noise_pool_position=3,
        noise_pool_valid=256,
        cache_slot=2,
    )

    assert VibeVoiceRequestState.from_dict(state.as_dict()) == state


def test_packed_weight_mapping_is_scoped_to_qwen_only() -> None:
    assert (
        _native_checkpoint_target_name(
            "model.language_model.layers.0.self_attn.q_proj.weight"
        )
        == "model.language_model.layers.0.self_attn.qkv_proj.weight"
    )
    diffusion_gate = "model.prediction_head.layers.0.ffn.gate_proj.weight"
    assert _native_checkpoint_target_name(diffusion_gate) == diffusion_gate


def test_codec_cache_gathers_and_resets_request_slots_on_device() -> None:
    cache = VibeVoiceBatchedStreamingCache(capacity=4)
    initial = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    slots = torch.tensor([0, 2], dtype=torch.long)
    cache.set("decoder.layer", slots, initial)

    assert cache.get("decoder.layer", torch.tensor([2, 0])).tolist() == [
        [3.0, 4.0],
        [1.0, 2.0],
    ]
    cache.set_to_zero(torch.tensor([2]))
    assert cache.get("decoder.layer", slots).tolist() == [[1.0, 2.0], [0.0, 0.0]]


def test_streaming_transposed_conv_uses_static_causal_cache_shape() -> None:
    layer = SConvTranspose1d(
        in_channels=2,
        out_channels=1,
        kernel_size=4,
        stride=2,
        causal=True,
    )
    cache = VibeVoiceBatchedStreamingCache(capacity=2)
    slots = torch.tensor([0], dtype=torch.long)

    layer(torch.ones(1, 2, 1), cache=cache, sample_indices=slots, use_cache=True)
    first_shape = cache.cache[layer.layer_id].shape
    layer(torch.ones(1, 2, 1), cache=cache, sample_indices=slots, use_cache=True)

    assert first_shape[-1] == layer.context_size
    assert cache.cache[layer.layer_id].shape == first_shape


def test_auxiliary_stream_compacts_active_rows_and_uses_logical_slots() -> None:
    common = _CommonMetadata(
        seq_lens=torch.tensor([9, 11], dtype=torch.int32),
        slot_mapping=torch.tensor([5, 6], dtype=torch.int64),
        block_table_tensor=torch.tensor([[3, 4], [7, 8]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
    )
    name = "model.negative_language_model.layers.0.self_attn.attn"
    metadata = {name: _LayerMetadata(common)}
    slots: dict[str, torch.Tensor] = {}
    spec = AuxiliaryAttentionStreamSpec(
        name="negative_cfg",
        layer_name_marker="negative_language_model",
        compact_active=True,
        sequences=(
            AuxiliarySequenceState(sequence_length=1, write_position=None, active=False),
            AuxiliarySequenceState(sequence_length=18, write_position=17, active=True),
        ),
    )

    apply_auxiliary_attention_streams(
        attn_metadata=metadata,
        slot_mappings=slots,
        specs=(spec,),
        request_token_spans=[(0, 1), (1, 2)],
        block_size=16,
    )

    compact = metadata[name].common_attn_metadata
    assert compact.seq_lens.tolist() == [18]
    assert compact.block_table_tensor.tolist() == [[7, 8]]
    assert compact.slot_mapping.tolist() == [8 * 16 + 1]
    assert compact.query_start_loc.tolist() == [0, 1]
    assert compact.query_start_loc_cpu.tolist() == [0, 1]
    assert compact._seq_lens_cpu.tolist() == [18]
    assert compact._num_computed_tokens_cpu.tolist() == [17]
    assert compact.max_seq_len == 32
    assert compact.seq_lens_cpu_upper_bound.tolist() == [32]
    assert compact.num_reqs == 1
    assert slots[name].tolist() == compact.slot_mapping.tolist()


def test_auxiliary_stream_remaps_logical_zero_away_from_prefix_blocks() -> None:
    common = _CommonMetadata(
        seq_lens=torch.tensor([49], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        block_table_tensor=torch.tensor([[2, 3, 9, 10]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
    )
    name = "model.negative_language_model.layers.0.self_attn.attn"
    metadata = {name: _LayerMetadata(common)}
    spec = AuxiliaryAttentionStreamSpec(
        name="negative_cfg",
        layer_name_marker="negative_language_model",
        compact_active=True,
        sequences=(
            AuxiliarySequenceState(
                sequence_length=18,
                write_position=17,
                active=True,
                block_table_start=2,
            ),
        ),
    )

    apply_auxiliary_attention_streams(
        attn_metadata=metadata,
        slot_mappings={},
        specs=(spec,),
        request_token_spans=[(0, 1)],
        block_size=16,
    )

    compact = metadata[name].common_attn_metadata
    assert compact.block_table_tensor[0, :2].tolist() == [9, 10]
    assert compact.slot_mapping.tolist() == [10 * 16 + 1]


def test_auxiliary_stream_rewrites_materialized_backend_metadata() -> None:
    name = "model.negative_language_model.layers.0.self_attn.attn"
    metadata = {
        name: _BackendMetadata(
            seq_lens=torch.tensor([33], dtype=torch.int32),
            slot_mapping=torch.tensor([0], dtype=torch.int64),
            block_table=torch.tensor([[4, 5, 6]], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
            scheduler_metadata=torch.ones(9, dtype=torch.int32),
            prefix_scheduler_metadata=torch.ones(5, dtype=torch.int32),
            cu_prefix_query_lens=torch.tensor([0, 1], dtype=torch.int32),
            prefix_kv_lens=torch.tensor([7], dtype=torch.int32),
            suffix_kv_lens=torch.tensor([26], dtype=torch.int32),
        )
    }
    spec = AuxiliaryAttentionStreamSpec(
        name="negative_cfg",
        layer_name_marker="negative_language_model",
        compact_active=True,
        sequences=(
            AuxiliarySequenceState(
                sequence_length=2,
                write_position=1,
                active=True,
                block_table_start=2,
            ),
        ),
    )

    apply_auxiliary_attention_streams(
        attn_metadata=metadata,
        slot_mappings={},
        specs=(spec,),
        request_token_spans=[(0, 1)],
        block_size=16,
    )

    rewritten = metadata[name]
    assert rewritten.block_table[0, 0].item() == 6
    assert rewritten.slot_mapping.tolist() == [6 * 16 + 1]
    assert rewritten.common_prefix_len == 0
    assert rewritten.use_cascade is False
    assert rewritten.scheduler_metadata is None
    assert rewritten.prefix_scheduler_metadata is None
    assert rewritten.cu_prefix_query_lens is None
    assert rewritten.prefix_kv_lens is None
    assert rewritten.suffix_kv_lens is None
    assert rewritten.num_decode_reqs == 1
    assert rewritten.num_prefill_reqs == 0


def test_native_backend_source_does_not_call_hf_generate() -> None:
    source_path = (
        Path(__file__).parents[4]
        / "vllm_omni/model_executor/models/vibevoice/modeling_vibevoice_native.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    generate_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "generate"
    ]

    assert generate_calls == []


def test_negative_cfg_bypasses_runner_owned_model_compile_wrapper() -> None:
    source_path = (
        Path(__file__).parents[4]
        / "vllm_omni/model_executor/models/vibevoice/modeling_vibevoice_native.py"
    )
    source = source_path.read_text(encoding="utf-8")

    # The CFG branch executes in prepare_omni_forward_inputs, before the
    # runner-owned compiled model call.  Calling the decorated Qwen2Model via
    # __call__ here causes an invalid nested CUDA Graph capture/replay.
    assert "negative_hidden = self._negative_cfg_graph.run(" in source

    graph_source_path = source_path.with_name("negative_cfg_graph.py")
    graph_source = graph_source_path.read_text(encoding="utf-8")
    assert "return self.model.forward(input_ids, positions, None, inputs_embeds)" in graph_source
    assert '"triton.cudagraphs": False' in graph_source
    assert '"triton.cudagraph_trees": False' in graph_source


def test_negative_graph_metadata_clone_owns_and_refreshes_tensor_storage() -> None:
    shared = torch.tensor([1, 2], dtype=torch.int32)
    source = {
        "negative.layer.0": _BackendMetadata(
            seq_lens=shared,
            slot_mapping=torch.tensor([7], dtype=torch.int64),
            block_table=torch.tensor([[3, 4]], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
            max_seq_len=16,
        ),
        "negative.layer.1": {"shared_seq_lens": shared},
    }
    cloned = _clone_metadata(source)

    assert cloned["negative.layer.0"].seq_lens.data_ptr() != shared.data_ptr()
    assert cloned["negative.layer.0"].seq_lens is cloned["negative.layer.1"]["shared_seq_lens"]
    signature = _metadata_signature(cloned)

    shared.fill_(9)
    source["negative.layer.0"].slot_mapping.fill_(11)
    assert _copy_metadata_tensors(cloned, source)
    assert cloned["negative.layer.0"].seq_lens.tolist() == [9, 9]
    assert cloned["negative.layer.0"].slot_mapping.tolist() == [11]
    assert _metadata_signature(cloned) == signature


def test_diffusion_updates_persistent_cudagraph_input_buffer_in_place() -> None:
    source_path = (
        Path(__file__).parents[4]
        / "vllm_omni/model_executor/models/vibevoice/modeling_vibevoice_native.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "inputs_embeds.index_copy_(0, row_tensor, next_embeddings)" in source
    assert "inputs_embeds = inputs_embeds.clone()" not in source


def test_native_tp1_logits_only_project_the_constrained_token_rows() -> None:
    source_path = (
        Path(__file__).parents[4]
        / "vllm_omni/model_executor/models/vibevoice/modeling_vibevoice_native.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "self.lm_head.weight.index_select(0, self._valid_token_ids)" in source
    assert "logits.index_copy_(-1, self._valid_token_ids, selected_logits)" in source


def test_native_backend_uses_kv_prefix_cache_without_tensor_payload_cache() -> None:
    from vllm_omni.model_executor.models.vibevoice.modeling_vibevoice_native import (
        VibeVoiceNativeForConditionalGeneration,
    )

    assert VibeVoiceNativeForConditionalGeneration.requires_full_prefix_cached_hidden_states is False
    assert VibeVoiceNativeForConditionalGeneration.disable_omni_tensor_prefix_cache is True
    assert VibeVoiceNativeForConditionalGeneration.use_async_omni_output is True
    assert VibeVoiceNativeForConditionalGeneration.eager_omni_postprocess_before_async_output is True
    assert callable(VibeVoiceNativeForConditionalGeneration.profile_side_modules)


def test_native_backend_publishes_scheduler_evidence_hooks() -> None:
    source_path = (
        Path(__file__).parents[4]
        / "vllm_omni/model_executor/models/vibevoice/modeling_vibevoice_native.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "def observe_scheduler_step" in source
    assert "self._metrics.prefix_cache(computed_tokens=computed)" in source
    assert "self._metrics.scheduler_step(" in source
    assert '"scheduler_evidence": self._metrics.scheduler_snapshot()' in source


def test_checkpoint_preflight_accepts_the_released_manifest(tmp_path: Path) -> None:
    config = {
        "architectures": ["VibeVoiceForConditionalGeneration"],
        "model_type": "vibevoice",
        "decoder_config": {
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "max_position_embeddings": 65536,
            "num_attention_heads": 12,
            "num_hidden_layers": 28,
            "num_key_value_heads": 2,
            "tie_word_embeddings": True,
            "vocab_size": 151936,
        },
        "diffusion_head_config": {"head_layers": 4},
    }
    names = (
        _language_model_keys(28)
        | _prediction_head_keys(4)
        | _connector_keys("acoustic_connector")
        | _connector_keys("semantic_connector")
        | {"model.speech_bias_factor", "model.speech_scaling_factor"}
        | {f"model.acoustic_tokenizer.synthetic.{index}" for index in range(552)}
        | {f"model.semantic_tokenizer.synthetic.{index}" for index in range(276)}
    )
    shard_name = "model-00001-of-00001.safetensors"
    index = {
        "metadata": {"total_size": 123},
        "weight_map": {name: shard_name for name in names},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    (tmp_path / shard_name).touch()

    report = validate_vibevoice_checkpoint(tmp_path, inspect_headers=False)

    assert report.tensor_count == 1204
    assert report.shard_count == 1
    assert report.headers_checked is False
