# SPDX-License-Identifier: Apache-2.0
"""Safe, model-owned CUDA graphs for VibeVoice's auxiliary CFG Qwen pass.

The positive Qwen invocation is owned by vLLM's model runner and therefore
uses vLLM's normal compile/CUDA-Graph path.  The negative Qwen pass executes
from ``prepare_omni_forward_inputs`` and must not enter that same wrapper: doing
so attempts a nested capture/replay.  This module gives the auxiliary pass its
own compile and graph lifecycle while retaining the active vLLM paged-attention
``ForwardContext``.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import torch
import torch.nn as nn
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "off", "no"}


def _is_metadata_container(value: Any) -> bool:
    return (
        isinstance(value, (torch.Tensor, dict, list, tuple))
        or is_dataclass(value)
        or (hasattr(value, "__dict__") and not isinstance(value, nn.Module))
    )


def _clone_metadata(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """Clone attention metadata and own every tensor storage it references."""
    if memo is None:
        memo = {}
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, torch.Tensor):
        cloned = value.detach().clone()
        memo[identity] = cloned
        return cloned
    if isinstance(value, dict):
        cloned_dict: dict[Any, Any] = {}
        memo[identity] = cloned_dict
        cloned_dict.update({key: _clone_metadata(item, memo) for key, item in value.items()})
        return cloned_dict
    if isinstance(value, list):
        cloned_list: list[Any] = []
        memo[identity] = cloned_list
        cloned_list.extend(_clone_metadata(item, memo) for item in value)
        return cloned_list
    if isinstance(value, tuple):
        cloned_tuple = tuple(_clone_metadata(item, memo) for item in value)
        memo[identity] = cloned_tuple
        return cloned_tuple
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _clone_metadata(getattr(value, field.name), memo)
            for field in fields(value)
        }
        try:
            # ``replace`` also works for frozen/slots dataclasses used by some
            # vLLM attention backends.
            cloned_object = replace(value, **updates)
        except (TypeError, ValueError):
            cloned_object = copy.copy(value)
            for name, item in updates.items():
                try:
                    setattr(cloned_object, name, item)
                except (AttributeError, TypeError):
                    pass
        memo[identity] = cloned_object
        return cloned_object
    if hasattr(value, "__dict__") and not isinstance(value, nn.Module):
        try:
            cloned_object = copy.copy(value)
        except (TypeError, RuntimeError):
            return value
        memo[identity] = cloned_object
        for name, item in vars(value).items():
            if _is_metadata_container(item):
                try:
                    setattr(cloned_object, name, _clone_metadata(item, memo))
                except (AttributeError, TypeError):
                    pass
        return cloned_object
    return value


def _copy_metadata_tensors(target: Any, source: Any, seen: set[tuple[int, int]] | None = None) -> bool:
    """Refresh captured metadata tensors; return false on structural mismatch."""
    if seen is None:
        seen = set()
    pair = (id(target), id(source))
    if pair in seen:
        return True
    seen.add(pair)
    if isinstance(target, torch.Tensor):
        if not isinstance(source, torch.Tensor) or target.shape != source.shape:
            return False
        if target.dtype != source.dtype or target.device != source.device:
            return False
        target.copy_(source, non_blocking=target.is_cuda)
        return True
    if isinstance(target, dict):
        return isinstance(source, dict) and target.keys() == source.keys() and all(
            _copy_metadata_tensors(target[key], source[key], seen) for key in target
        )
    if isinstance(target, (list, tuple)):
        return isinstance(source, type(target)) and len(target) == len(source) and all(
            _copy_metadata_tensors(dst, src, seen) for dst, src in zip(target, source, strict=True)
        )
    if is_dataclass(target) and is_dataclass(source):
        return all(
            _copy_metadata_tensors(getattr(target, field.name), getattr(source, field.name), seen)
            for field in fields(target)
        )
    if hasattr(target, "__dict__") and hasattr(source, "__dict__"):
        for name, dst in vars(target).items():
            if _is_metadata_container(dst):
                if not hasattr(source, name) or not _copy_metadata_tensors(dst, getattr(source, name), seen):
                    return False
    return True


_STATIC_FIELDS = (
    "max_seq_len",
    "max_query_len",
    "num_actual_tokens",
    "num_reqs",
    "num_prefills",
    "num_prefill_tokens",
    "num_decode_tokens",
    "num_decode_reqs",
    "num_prefill_reqs",
    "common_prefix_len",
    "causal",
    "use_cascade",
)


def _metadata_signature(value: Any, seen: set[int] | None = None) -> tuple[Any, ...]:
    """Describe tensor shapes and Python values baked into a CUDA graph."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return ("alias",)
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), str(value.device))
    if isinstance(value, dict):
        seen.add(identity)
        return (
            "dict",
            tuple((str(key), _metadata_signature(item, seen)) for key, item in value.items()),
        )
    if isinstance(value, (list, tuple)):
        seen.add(identity)
        return (type(value).__name__, tuple(_metadata_signature(item, seen) for item in value))
    if is_dataclass(value) or hasattr(value, "__dict__"):
        seen.add(identity)
        tensor_fields: list[tuple[str, tuple[Any, ...]]] = []
        values = vars(value) if hasattr(value, "__dict__") else {
            field.name: getattr(value, field.name) for field in fields(value)
        }
        for name, item in values.items():
            if _is_metadata_container(item):
                tensor_fields.append((name, _metadata_signature(item, seen)))
        static_fields = tuple(
            (name, getattr(value, name))
            for name in _STATIC_FIELDS
            if hasattr(value, name)
        )
        return (type(value).__name__, tuple(tensor_fields), static_fields)
    return (type(value).__name__,)


class _NegativeQwenCompileAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # Call forward explicitly to bypass Qwen2Model's runner-owned vLLM
        # compile wrapper. This adapter owns the auxiliary compilation instead.
        return self.model.forward(input_ids, positions, None, inputs_embeds)


@dataclass
class _CapturedNegativeGraph:
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor
    positions: torch.Tensor
    inputs_embeds: torch.Tensor
    output: torch.Tensor
    attention_metadata: dict[str, Any]


class NegativeCFGGraphRunner:
    """Compile and optionally graph the compact negative-CFG Qwen batch."""

    def __init__(
        self,
        model: nn.Module,
        *,
        layer_marker: str,
        tensor_parallel_size: int = 1,
    ) -> None:
        self.model = model
        self.layer_marker = layer_marker
        self.tensor_parallel_size = max(1, int(tensor_parallel_size))
        raw_sizes = os.getenv("VLLM_OMNI_VIBEVOICE_NEGATIVE_GRAPH_BATCH_SIZES", "1,2,4,8")
        self.capture_batch_sizes = tuple(
            sorted({int(item) for item in raw_sizes.split(",") if item.strip() and int(item) > 0})
        )
        self.compile_enabled = _enabled("VLLM_OMNI_VIBEVOICE_COMPILE_NEGATIVE_QWEN")
        self.graph_requested = _enabled(
            "VLLM_OMNI_VIBEVOICE_GRAPH_NEGATIVE_QWEN", default="0"
        )
        # The auxiliary branch owns a manual torch.cuda.CUDAGraph.  Under TP,
        # Qwen's row-parallel layers also enqueue NCCL collectives. Capturing
        # one independent graph per rank is not safe unless capture/replay is
        # coordinated by vLLM's distributed graph runner. The previous manual
        # path corrupted CUDA state on the second replay. Keep torch.compile
        # enabled, but fail closed for manual graphs whenever TP > 1.
        self.graph_enabled = self.graph_requested and self.tensor_parallel_size == 1
        self.graph_disabled_reason: str | None = None
        if self.graph_requested and not self.graph_enabled:
            self.graph_disabled_reason = "manual_negative_graph_unsupported_with_tensor_parallel"
            logger.warning(
                "VibeVoice negative CFG CUDA Graph disabled: tensor_parallel_size=%d; "
                "using compiled/eager auxiliary Qwen path",
                self.tensor_parallel_size,
            )
        self._compiled: nn.Module | None = None
        self._compile_verified = False
        self._graphs: dict[tuple[Any, ...], _CapturedNegativeGraph] = {}
        self._failed_signatures: set[tuple[Any, ...]] = set()
        self.graph_hits = 0
        self.graph_misses = 0
        self.last_execution = "uninitialized"

    @property
    def compile_verified(self) -> bool:
        return self._compile_verified

    @property
    def captured_batch_sizes(self) -> list[int]:
        return sorted({int(key[0]) for key in self._graphs})

    @property
    def graph_ready(self) -> bool:
        return bool(self._graphs)

    def configure_compile(self) -> None:
        if self._compiled is not None or not self.compile_enabled or not torch.cuda.is_available():
            return
        adapter = _NegativeQwenCompileAdapter(self.model)
        try:
            # This compiled callable is subsequently captured by our graph.
            # Disable Inductor graph trees to prevent nested graph ownership.
            self._compiled = torch.compile(
                adapter,
                fullgraph=False,
                dynamic=True,
                options={
                    "triton.cudagraphs": False,
                    "triton.cudagraph_trees": False,
                },
            )
            logger.info("VibeVoice negative CFG Qwen compile configured (Inductor CUDA Graphs disabled)")
        except Exception as exc:  # pragma: no cover - compiler/driver dependent
            self._compiled = None
            logger.warning("VibeVoice negative CFG Qwen compile unavailable: %s", exc)

    def _call(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if self._compiled is None:
            output = self.model.forward(input_ids, positions, None, inputs_embeds)
            self.last_execution = "eager"
        else:
            try:
                output = self._compiled(input_ids, positions, inputs_embeds)
                self._compile_verified = True
                self.last_execution = "compiled"
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:  # pragma: no cover - compiler/driver dependent
                logger.warning(
                    "VibeVoice negative CFG Qwen compile failed at runtime; using eager: %s",
                    exc,
                )
                self._compiled = None
                self._compile_verified = False
                output = self.model.forward(input_ids, positions, None, inputs_embeds)
                self.last_execution = "eager_fallback"
        return output

    def _negative_metadata(self) -> tuple[Any, dict[str, Any]] | None:
        context = get_forward_context()
        metadata = getattr(context, "attn_metadata", None)
        if not isinstance(metadata, dict):
            return None
        selected = {name: item for name, item in metadata.items() if self.layer_marker in name}
        if not selected:
            return None
        return context, selected

    @torch.no_grad()
    def _capture(
        self,
        *,
        key: tuple[Any, ...],
        context: Any,
        metadata: dict[str, Any],
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> _CapturedNegativeGraph:
        static_metadata = _clone_metadata(metadata)
        static_context = copy.copy(context)
        static_context.attn_metadata = static_metadata
        static_ids = torch.empty_like(input_ids)
        static_positions = torch.empty_like(positions)
        static_embeds = torch.empty_like(inputs_embeds)
        static_ids.copy_(input_ids)
        static_positions.copy_(positions)
        static_embeds.copy_(inputs_embeds)

        with override_forward_context(static_context):
            for _ in range(2):
                _ = self._call(static_ids, static_positions, static_embeds)
            torch.cuda.synchronize(input_ids.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph,
                pool=current_platform.get_global_graph_pool(),
                capture_error_mode="thread_local",
            ):
                output = self._call(static_ids, static_positions, static_embeds)

        entry = _CapturedNegativeGraph(
            graph=graph,
            input_ids=static_ids,
            positions=static_positions,
            inputs_embeds=static_embeds,
            output=output,
            attention_metadata=static_metadata,
        )
        self._graphs[key] = entry
        logger.info(
            "VibeVoice negative CFG CUDA Graph captured: batch_size=%d captured_batch_sizes=%s",
            int(input_ids.shape[0]),
            self.captured_batch_sizes,
        )
        return entry

    @torch.no_grad()
    def run(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        self.configure_compile()
        batch_size = int(input_ids.shape[0])
        if (
            not self.graph_enabled
            or not input_ids.is_cuda
            or batch_size not in self.capture_batch_sizes
            or torch.cuda.is_current_stream_capturing()
        ):
            self.graph_misses += 1
            return self._call(input_ids, positions, inputs_embeds)

        context_and_metadata = self._negative_metadata()
        if context_and_metadata is None:
            self.graph_misses += 1
            return self._call(input_ids, positions, inputs_embeds)
        context, metadata = context_and_metadata
        key = (batch_size, _metadata_signature(metadata))
        if key in self._failed_signatures:
            self.graph_misses += 1
            return self._call(input_ids, positions, inputs_embeds)

        entry = self._graphs.get(key)
        if entry is None:
            try:
                entry = self._capture(
                    key=key,
                    context=context,
                    metadata=metadata,
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                )
                # Capture executes the graph once and leaves a valid output.
                self.last_execution = "graph_capture"
                return entry.output.clone()
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:  # pragma: no cover - compiler/driver dependent
                self._graphs.pop(key, None)
                self._failed_signatures.add(key)
                self.graph_misses += 1
                logger.warning(
                    "VibeVoice negative CFG CUDA Graph capture failed; using compiled/eager path: %s",
                    exc,
                )
                return self._call(input_ids, positions, inputs_embeds)

        if not _copy_metadata_tensors(entry.attention_metadata, metadata):
            self.graph_misses += 1
            return self._call(input_ids, positions, inputs_embeds)
        entry.input_ids.copy_(input_ids, non_blocking=True)
        entry.positions.copy_(positions, non_blocking=True)
        entry.inputs_embeds.copy_(inputs_embeds, non_blocking=True)
        entry.graph.replay()
        self.graph_hits += 1
        self.last_execution = "graph_hit"
        return entry.output.clone()


__all__ = ["NegativeCFGGraphRunner"]
