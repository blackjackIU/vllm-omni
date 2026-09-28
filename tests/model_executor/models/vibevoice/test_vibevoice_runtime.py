# SPDX-License-Identifier: Apache-2.0

from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.model_executor.models.vibevoice.modeling_vibevoice import (
    _generation_group_key,
    _group_generation_infos,
)
from vllm_omni.model_executor.models.vibevoice.pipeline import (
    VIBEVOICE_LEGACY_PIPELINE,
    VIBEVOICE_PIPELINE,
    VIBEVOICE_STREAMING_PIPELINE,
)


def test_generation_groups_only_compatible_requests() -> None:
    infos = [
        {"text": "a", "seed": 42, "ddpm_steps": 10},
        {"text": "b", "seed": 42, "ddpm_steps": 10},
        {"text": "c", "seed": 7, "ddpm_steps": 10},
        {"text": "d", "seed": 42, "ddpm_steps": 5},
    ]

    groups = _group_generation_infos(infos)

    assert [indices for indices, _ in groups] == [[0, 1], [2], [3]]
    assert _generation_group_key(infos[0]) == _generation_group_key(infos[1])
    assert _generation_group_key(infos[0]) != _generation_group_key(infos[2])


def test_vibevoice_profiles_use_the_expected_worker_types() -> None:
    throughput = VIBEVOICE_PIPELINE.stages[0]
    streaming = VIBEVOICE_STREAMING_PIPELINE.stages[0]
    legacy = VIBEVOICE_LEGACY_PIPELINE.stages[0]

    assert throughput.execution_type == StageExecutionType.LLM_AR
    assert throughput.model_arch is None
    assert throughput.sampling_constraints["max_tokens"] == 65536
    assert streaming.execution_type == StageExecutionType.LLM_AR
    assert streaming.model_arch == "VibeVoiceStreamingForConditionalGeneration"
    assert streaming.sampling_constraints["max_tokens"] == 65536
    assert legacy.execution_type == StageExecutionType.LLM_GENERATION
    assert legacy.model_arch == "LegacyVibeVoiceForConditionalGeneration"
    assert legacy.sampling_constraints["max_tokens"] == 1
