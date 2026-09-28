#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate a downloaded microsoft/VibeVoice-1.5B checkpoint before serve."""

from __future__ import annotations

import argparse
import json

from vllm_omni.model_executor.models.vibevoice.checkpoint import (
    validate_vibevoice_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Local microsoft/VibeVoice-1.5B directory")
    parser.add_argument(
        "--skip-headers",
        action="store_true",
        help="Only validate JSON manifests and shard presence",
    )
    args = parser.parse_args()
    report = validate_vibevoice_checkpoint(
        args.model_dir,
        inspect_headers=not args.skip_headers,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
