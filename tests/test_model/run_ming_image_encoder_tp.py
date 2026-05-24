#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate Ming-Omni image encoder output consistency across TP configurations.

Launches the full Ming-Omni pipeline with configurable image encoder TP,
sends image+text prompts through the OpenAI-compatible API, and compares
outputs between TP=1 (baseline) and TP=2.

Usage:

    # Run with image encoder on 1 GPU (baseline):
    python tests/test_model/run_ming_image_encoder_tp.py run \
        --image-encoder-tp 1 --gpu-image-encoder 0 --output tp1.json

    # Run with image encoder sharded across 2 GPUs:
    python tests/test_model/run_ming_image_encoder_tp.py run \
        --image-encoder-tp 2 --gpu-image-encoder 2 3 --output tp2.json

    # Compare outputs:
    python tests/test_model/run_ming_image_encoder_tp.py compare tp1.json tp2.json

Requires:
    - Ming-flash-omni-2.0 model weights downloaded
    - Multiple GPUs (at least 2 for TP=2 image encoder + thinker GPU)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import os
import sys

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TEST_PROMPTS = [
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://picsum.photos/id/237/300/200"},
                    },
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ],
    },
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://picsum.photos/id/10/300/200"},
                    },
                    {"type": "text", "text": "What colors are present in this image?"},
                ],
            }
        ],
    },
    {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
        ],
    },
    {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://picsum.photos/id/1/300/200"},
                    },
                    {"type": "text", "text": "How many objects are in this image?"},
                ],
            }
        ],
    },
]


async def run_pipeline(
    image_encoder_tp: int,
    gpu_image_encoder: list[int],
    gpu_thinker: int,
    cpu_offload_gb: int,
    mem_fraction: float,
    output_file: str,
):
    from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.proto import OmniRequest

    config = MingOmniPipelineConfig(model_path="inclusionAI/Ming-flash-omni-2.0")

    stages = [stage.model_copy(deep=True) for stage in config.stages]
    for stage in stages:
        if stage.name == "image_encoder":
            stage.tp_size = image_encoder_tp
            stage.parallelism = stage.parallelism.model_copy(
                update={"tp": image_encoder_tp}
            )
            if image_encoder_tp > 1:
                stage.gpu = gpu_image_encoder
            else:
                stage.gpu = gpu_image_encoder[0]
        if stage.name == "thinker":
            stage.gpu = gpu_thinker
            stage.factory_args = {
                **stage.factory_args,
                "server_args_overrides": {
                    "cpu_offload_gb": cpu_offload_gb,
                    "mem_fraction_static": mem_fraction,
                },
            }

    config = MingOmniPipelineConfig(
        model_path="inclusionAI/Ming-flash-omni-2.0",
        stages=stages,
    )

    runner = MultiProcessPipelineRunner(config)
    logger.info(
        "Starting pipeline with image_encoder_tp=%d, gpu_image_encoder=%s, "
        "gpu_thinker=%d ...",
        image_encoder_tp,
        gpu_image_encoder,
        gpu_thinker,
    )
    await runner.start(timeout=600)

    results = []
    try:
        for i, prompt in enumerate(TEST_PROMPTS):
            has_image = any(
                isinstance(m.get("content"), list)
                and any(c.get("type") == "image_url" for c in m["content"])
                for m in prompt["messages"]
            )
            label = f"[{i+1}/{len(TEST_PROMPTS)}] {'image+text' if has_image else 'text-only'}"
            logger.info("%s", label)

            result = await asyncio.wait_for(
                runner.coordinator.submit(
                    f"img-tp-test-{i}",
                    OmniRequest(
                        inputs=prompt,
                        params={"max_new_tokens": 64, "temperature": 0.0},
                    ),
                ),
                timeout=120,
            )

            text = ""
            if isinstance(result, dict):
                for stage_name, payload in result.items():
                    data = (
                        payload
                        if isinstance(payload, dict)
                        else getattr(payload, "data", {})
                    )
                    if isinstance(data, dict) and "text" in data:
                        text = data["text"]
                        break
            assert text, f"Empty output for prompt {i}"
            results.append(
                {
                    "prompt_idx": i,
                    "has_image": has_image,
                    "output": text,
                }
            )
            logger.info("  Output: %s", text[:200])
    finally:
        await runner.stop()

    with open(output_file, "w") as f:
        json.dump(
            {
                "image_encoder_tp": image_encoder_tp,
                "gpu_image_encoder": gpu_image_encoder,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Results saved to %s", output_file)


def compare_outputs(file1: str, file2: str) -> bool:
    with open(file1) as f:
        data1 = json.load(f)
    with open(file2) as f:
        data2 = json.load(f)

    tp1 = data1["image_encoder_tp"]
    tp2 = data2["image_encoder_tp"]
    print(f"\n{'='*60}")
    print(f"Comparing image_encoder_tp={tp1} vs image_encoder_tp={tp2}")
    print(f"{'='*60}")

    all_match = True
    for r1, r2 in zip(data1["results"], data2["results"]):
        match = r1["output"].strip() == r2["output"].strip()
        kind = "image+text" if r1["has_image"] else "text-only"
        status = "MATCH" if match else "MISMATCH"
        if not match:
            all_match = False
        print(f"\n[{status}] prompt {r1['prompt_idx']} ({kind})")
        print(f"  TP={tp1}: {r1['output'][:120]}")
        print(f"  TP={tp2}: {r2['output'][:120]}")

    print(f"\n{'='*60}")
    if all_match:
        print("ALL OUTPUTS MATCH — image encoder TP validation PASSED")
    else:
        print(
            "OUTPUTS DIFFER — expected for image prompts if TP introduces "
            "minor floating-point divergence. Check text-only prompts match."
        )
    print(f"{'='*60}")
    return all_match


def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run")
    run_p.add_argument(
        "--image-encoder-tp", type=int, required=True, help="TP size for image encoder"
    )
    run_p.add_argument(
        "--gpu-image-encoder",
        type=int,
        nargs="+",
        default=[0],
        help="GPU(s) for image encoder",
    )
    run_p.add_argument("--gpu-thinker", type=int, default=0, help="GPU for thinker")
    run_p.add_argument("--cpu-offload-gb", type=int, default=80)
    run_p.add_argument("--mem-fraction", type=float, default=0.80)
    run_p.add_argument("--output", type=str, default=None)

    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("file1")
    cmp_p.add_argument("file2")

    args = parser.parse_args()

    if args.cmd == "run":
        output = args.output or f"image_encoder_tp{args.image_encoder_tp}_results.json"
        asyncio.run(
            run_pipeline(
                args.image_encoder_tp,
                args.gpu_image_encoder,
                args.gpu_thinker,
                args.cpu_offload_gb,
                args.mem_fraction,
                output,
            )
        )
    elif args.cmd == "compare":
        sys.exit(0 if compare_outputs(args.file1, args.file2) else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
