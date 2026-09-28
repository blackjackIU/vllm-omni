# SPDX-License-Identifier: Apache-2.0
"""Generic logical KV streams backed by vLLM paged-cache tensors.

An auxiliary stream reuses scheduler-owned block tables while maintaining a
different logical sequence length and write position.  Attention modules in a
separate cache namespace still receive distinct KV tensors; reusing physical
block *indices* therefore does not alias their contents.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

import torch


@dataclass(frozen=True)
class AuxiliarySequenceState:
    sequence_length: int
    write_position: int | None
    active: bool
    # Back logical block zero with a later scheduler-owned physical block.
    # Auxiliary streams can thereby avoid immutable/shared prefix blocks.
    block_table_start: int = 0


@dataclass(frozen=True)
class AuxiliaryAttentionStreamSpec:
    name: str
    layer_name_marker: str
    sequences: tuple[AuxiliarySequenceState, ...]
    compact_active: bool = False


def _clone_with_fields(value: Any, **updates: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "__dataclass_fields__"):
        accepted = {key: item for key, item in updates.items() if key in value.__dataclass_fields__}
        if accepted:
            try:
                return replace(value, **accepted)
            except (TypeError, ValueError):
                pass
    cloned = copy.copy(value)
    for key, item in updates.items():
        if hasattr(cloned, key):
            try:
                setattr(cloned, key, item)
            except (AttributeError, TypeError):
                pass
    return cloned


def _metadata_common(metadata: Any) -> tuple[Any, str | None]:
    if isinstance(metadata, dict):
        for key in ("common_attn_metadata", "common_metadata", "common"):
            if key in metadata:
                return metadata[key], key
        return metadata, None
    for key in ("common_attn_metadata", "common_metadata", "common"):
        common = getattr(metadata, key, None)
        if common is not None:
            return common, key
    return metadata, None


def _replace_metadata_common(metadata: Any, common: Any, common_key: str | None) -> Any:
    if common_key is None:
        return common
    if isinstance(metadata, dict):
        cloned = dict(metadata)
        cloned[common_key] = common
        return cloned
    return _clone_with_fields(metadata, **{common_key: common})


def _build_logical_tensors(
    *,
    common: Any,
    sequences: tuple[AuxiliarySequenceState, ...],
    request_token_spans: list[tuple[int, int]],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    standard_seq_lens = getattr(common, "seq_lens", None)
    standard_slots = getattr(common, "slot_mapping", None)
    standard_block_table = getattr(common, "block_table_tensor", None)
    if standard_block_table is None:
        # Backend-specific metadata (FlashAttention, FlashInfer, Triton,
        # Torch-SDPA) is already materialized by the runner and conventionally
        # exposes the same tensor as ``block_table``.
        standard_block_table = getattr(common, "block_table", None)
    if standard_seq_lens is None or standard_slots is None or standard_block_table is None:
        raise RuntimeError("Attention backend metadata does not expose paged-cache common fields")

    active_indices = tuple(index for index, state in enumerate(sequences) if state.active)
    seq_lens = torch.ones_like(standard_seq_lens)
    slots = torch.full_like(standard_slots, -1)
    block_table = torch.full_like(standard_block_table, -1)
    num_reqs = min(len(sequences), len(request_token_spans), int(seq_lens.shape[0]))
    for request_index in range(num_reqs):
        state = sequences[request_index]
        logical_length = max(1, int(state.sequence_length))
        seq_lens[request_index] = logical_length
        logical_blocks = (logical_length + block_size - 1) // block_size
        source_start = max(0, int(state.block_table_start))
        source_end = source_start + logical_blocks
        if source_end > int(standard_block_table.shape[1]):
            raise RuntimeError(
                f"Auxiliary KV stream needs source block columns [{source_start}:{source_end}], "
                f"but the scheduler table width is {standard_block_table.shape[1]}"
            )
        block_table[request_index, :logical_blocks] = standard_block_table[
            request_index, source_start:source_end
        ]
        start, end = request_token_spans[request_index]
        if not state.active or state.write_position is None or end - start != 1:
            continue
        position = int(state.write_position)
        block_column = position // block_size
        if block_column >= int(block_table.shape[1]):
            raise RuntimeError(
                f"Auxiliary KV stream position {position} exceeds allocated block table width"
            )
        block_number = block_table[request_index, block_column]
        slots[start] = block_number * block_size + (position % block_size)
    return seq_lens, slots, block_table, active_indices


def apply_auxiliary_attention_streams(
    *,
    attn_metadata: Any,
    slot_mappings: dict[str, torch.Tensor] | None,
    specs: tuple[AuxiliaryAttentionStreamSpec, ...],
    request_token_spans: list[tuple[int, int]],
    block_size: int,
) -> dict[str, torch.Tensor]:
    """Clone backend metadata only for auxiliary attention layer names."""
    if not specs or attn_metadata is None:
        return {}
    if isinstance(attn_metadata, list):
        # DBO microbatches need their own request-span partitioning.  Native
        # VibeVoice profiles disable DBO until vLLM exposes that partition in
        # the model-runner hook rather than silently installing wrong slots.
        raise RuntimeError("Auxiliary attention streams do not support vLLM microbatch metadata")
    if not isinstance(attn_metadata, dict):
        raise TypeError(f"Unsupported attention metadata container: {type(attn_metadata).__name__}")

    replacements: dict[str, torch.Tensor] = {}
    for spec in specs:
        matching = [name for name in attn_metadata if spec.layer_name_marker in name]
        if not matching:
            raise RuntimeError(
                f"Auxiliary attention stream {spec.name!r} found no layers matching "
                f"{spec.layer_name_marker!r}"
            )
        template_common, _ = _metadata_common(attn_metadata[matching[0]])
        seq_lens, logical_slots, block_table, active_indices = _build_logical_tensors(
            common=template_common,
            sequences=spec.sequences,
            request_token_spans=request_token_spans,
            block_size=block_size,
        )
        if spec.compact_active:
            if not active_indices:
                raise RuntimeError(f"Auxiliary attention stream {spec.name!r} has no active rows")
            active_index_tensor = torch.tensor(
                active_indices,
                device=seq_lens.device,
                dtype=torch.long,
            )
            seq_lens = seq_lens.index_select(0, active_index_tensor)
            block_table = block_table.index_select(0, active_index_tensor)
            compact_slots = []
            for request_index in active_indices:
                start, end = request_token_spans[request_index]
                compact_slots.append(logical_slots[start:end])
            logical_slots = torch.cat(compact_slots, dim=0)
        logical_states = (
            [spec.sequences[index] for index in active_indices]
            if spec.compact_active
            else list(spec.sequences)
        )
        exact_max_seq_len = max(
            (max(1, int(state.sequence_length)) for state in logical_states),
            default=1,
        )
        # Decode CUDA graphs need a stable Python upper bound while the actual
        # logical lengths continue to advance in GPU tensors. Power-of-two
        # buckets avoid capturing a new graph for every generated speech token.
        max_seq_len = max(
            block_size,
            1 << (max(1, exact_max_seq_len) - 1).bit_length(),
        )
        seq_lens_cpu = torch.tensor(
            [max(1, int(state.sequence_length)) for state in logical_states],
            dtype=seq_lens.dtype,
            device="cpu",
        )
        seq_lens_cpu_upper_bound = torch.full_like(seq_lens_cpu, max_seq_len)
        num_queries = int(logical_slots.numel())
        query_start_loc = torch.arange(
            0,
            num_queries + 1,
            device=logical_slots.device,
            dtype=torch.int32,
        )
        query_start_loc_cpu = query_start_loc.to(device="cpu")
        num_computed_tokens_cpu = seq_lens_cpu - 1
        for layer_name in matching:
            metadata = attn_metadata[layer_name]
            common, common_key = _metadata_common(metadata)
            common = _clone_with_fields(
                common,
                seq_lens=seq_lens,
                slot_mapping=logical_slots,
                block_table_tensor=block_table,
                block_table=block_table,
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                max_seq_len=max_seq_len,
                max_query_len=1,
                num_actual_tokens=num_queries,
                num_reqs=int(seq_lens.numel()),
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=num_queries,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                _seq_lens_cpu=seq_lens_cpu,
                _num_computed_tokens_cpu=num_computed_tokens_cpu,
                causal=True,
                common_prefix_len=0,
                use_cascade=False,
                # Backend AOT scheduling tensors are derived from the positive
                # batch's query/sequence lengths.  Reusing them after compacting
                # an auxiliary stream can launch a kernel with stale dimensions.
                # All supported vLLM paged-attention backends accept ``None`` and
                # compute their normal runtime schedule for this small sub-batch.
                scheduler_metadata=None,
                prefix_scheduler_metadata=None,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
                num_decode_reqs=int(seq_lens.numel()),
                num_prefill_reqs=0,
            )
            attn_metadata[layer_name] = _replace_metadata_common(metadata, common, common_key)
            replacements[layer_name] = logical_slots
            if isinstance(slot_mappings, dict):
                slot_mappings[layer_name] = logical_slots
    return replacements


__all__ = [
    "AuxiliaryAttentionStreamSpec",
    "AuxiliarySequenceState",
    "apply_auxiliary_attention_streams",
]
