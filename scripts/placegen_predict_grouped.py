"""Run target-free TAX-DPD predictions for the held-out PlaceGen test split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from non_rigid.utils.placegen_checkpoint import load_grouped_checkpoint
from non_rigid.utils.placegen_grouped import (
    MODEL_CAPABILITY,
    QUALITY_SCOPE,
    SPLITS,
    load_grouped_inference_manifest,
    load_grouped_observation,
    sha256_file,
    target_free_report_semantics,
)
from non_rigid.utils.placegen_prediction import predict_pose_candidates
from non_rigid.utils.placegen_request import MAX_CANDIDATE_COUNT

REPORT_SCHEMA = "placegen.taxdpd-grouped-prediction-report/0.1"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= args.candidates <= MAX_CANDIDATE_COUNT:
        raise ValueError(f"--candidates must be in [1, {MAX_CANDIDATE_COUNT}]")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    supplied_output = Path(args.output_json).expanduser()
    if supplied_output.exists() or supplied_output.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite prediction report: {supplied_output}"
        )
    output_path = supplied_output.resolve()
    inference = load_grouped_inference_manifest(args.inference_manifest)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA prediction but CUDA is unavailable")
    loaded = load_grouped_checkpoint(
        args.checkpoint,
        device,
        expected_split_assignment_sha256=inference.split_assignment_sha256,
    )
    checkpoint_dataset = loaded.payload["dataset"]
    if checkpoint_dataset["native_dataset_sha256"] != inference.native_dataset_sha256:
        raise ValueError("checkpoint and inference index use different native datasets")
    if checkpoint_dataset["point_counts"] != inference.point_counts:
        raise ValueError("checkpoint and inference index use different point counts")
    inference_sample_counts = {
        split: len(inference.samples_by_split[split]) for split in SPLITS
    }
    inference_group_counts = {
        split: len(
            {sample.split_group_id for sample in inference.samples_by_split[split]}
        )
        for split in SPLITS
    }
    if inference_sample_counts != checkpoint_dataset["sample_counts"]:
        raise ValueError("checkpoint and inference index sample partitions differ")
    if inference_group_counts != checkpoint_dataset["group_counts"]:
        raise ValueError(
            "checkpoint and inference index physical group partitions differ"
        )

    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    maximum_peak_vram = 0.0
    for sample_index, sample in enumerate(inference.samples_by_split["test"]):
        observation = load_grouped_observation(
            inference, sample.sample_id, required_split="test"
        )
        result = predict_pose_candidates(
            loaded.module,
            observation,
            device,
            candidate_count=args.candidates,
            seed=args.seed + sample_index * args.candidates,
        )
        if result.peak_vram_mib is not None:
            maximum_peak_vram = max(maximum_peak_vram, result.peak_vram_mib)
        records.append(
            {
                "sample_id": sample.sample_id,
                "split": "test",
                "split_group_id": sample.split_group_id,
                "descriptor_sha256": sample.descriptor_sha256,
                "input_npz_sha256": sample.input_npz_sha256,
                "model_input_sha256": observation.model_input_sha256,
                "seed": args.seed + sample_index * args.candidates,
                "candidate_count": len(result.candidates),
                "latency_seconds": result.latency_seconds,
                "peak_vram_mib": result.peak_vram_mib,
                "candidates": list(result.candidates),
            }
        )
    total_duration = time.perf_counter() - started
    if len(records) != inference_sample_counts["test"]:
        raise RuntimeError("held-out prediction did not cover the complete test split")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve(strict=True)
    report = {
        "schema": REPORT_SCHEMA,
        "quality_claim": False,
        "quality_assessment": "inconclusive-until-supervision-isolated-evaluation",
        "model": MODEL_CAPABILITY,
        "quality_scope": QUALITY_SCOPE,
        **target_free_report_semantics(),
        "training_manifest_read": False,
        "target_pose_or_goal_point_cloud_read": False,
        "inference_manifest": str(inference.manifest_path),
        "inference_manifest_sha256": inference.manifest_sha256,
        "split_assignment_sha256": inference.split_assignment_sha256,
        "native_dataset_sha256": inference.native_dataset_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_model_state_sha256": loaded.model_state_sha256,
        "checkpoint_optimizer_state_sha256": loaded.optimizer_state_sha256,
        "checkpoint_reload_verified": True,
        "checkpoint_selection_split": loaded.payload["training"]["selection_split"],
        "checkpoint_selection_candidate_index": loaded.payload["training"][
            "selection_candidate_index"
        ],
        "test_split_used_for_checkpoint_selection": loaded.payload["training"][
            "test_split_used_for_selection"
        ],
        "test_sample_count": len(records),
        "candidate_count_per_sample": args.candidates,
        "finite_prediction_gate_passed": True,
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "peak_vram_mib": maximum_peak_vram if device.type == "cuda" else None,
        "total_duration_seconds": total_duration,
        "predictions": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1701)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
