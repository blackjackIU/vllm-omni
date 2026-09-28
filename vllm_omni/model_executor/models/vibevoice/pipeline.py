# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VibeVoice 1.5B native and legacy pipeline topologies."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

VIBEVOICE_PIPELINE = PipelineConfig(
    model_type="vibevoice",
    default_deploy_config_name="vibevoice.yaml",
    model_arch="VibeVoiceForConditionalGeneration",
    hf_architectures=("VibeVoiceForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="vibevoice",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            sampling_constraints={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 65536,
                "detokenize": False,
                "stop_token_ids": [151643],
            },
        ),
    ),
)

VIBEVOICE_STREAMING_PIPELINE = PipelineConfig(
    model_type="vibevoice_streaming",
    default_deploy_config_name="vibevoice_streaming.yaml",
    model_arch="VibeVoiceStreamingForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="vibevoice_streaming",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            sampling_constraints={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 65536,
                "detokenize": False,
                "stop_token_ids": [151643],
            },
        ),
    ),
)

VIBEVOICE_LEGACY_PIPELINE = PipelineConfig(
    model_type="vibevoice_legacy",
    default_deploy_config_name="vibevoice_legacy_reference.yaml",
    model_arch="LegacyVibeVoiceForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="vibevoice_legacy",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            sampling_constraints={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 1,
                "detokenize": False,
            },
        ),
    ),
)

__all__ = [
    "VIBEVOICE_LEGACY_PIPELINE",
    "VIBEVOICE_PIPELINE",
    "VIBEVOICE_STREAMING_PIPELINE",
]
