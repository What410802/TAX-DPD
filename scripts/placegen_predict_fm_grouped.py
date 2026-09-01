"""Run target-free TAX3Dv2 Euclidean FM on the held-out PlaceGen test split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from non_rigid.utils.placegen_fm_checkpoint import (
    FM_MODEL_CAPABILITY,
    load_fm_checkpoint,
)
from non_rigid.utils.placegen_fm_export import (
    EXPECTED_SPLIT_COUNTS,
    load_fm_observation,
    load_native_inference_index,
    sha256_file,
)
from non_rigid.utils.placegen_fm_prediction import predict_fm_pose_candidates

REPORT_SCHEMA = "placegen.tax3dv2-fm-grouped-prediction-report/0.1"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.candidates <= 0 or args.candidates > 32:
        raise ValueError("--candidates must be in [1, 32]")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    output = Path(args.output_json).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite prediction report: {output}")
    inference_manifest = Path(args.inference_manifest).expanduser().resolve(strict=True)
    inference = load_native_inference_index(inference_manifest)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA prediction but CUDA is unavailable")
    loaded = load_fm_checkpoint(
        args.checkpoint,
        device,
        expected_inference_manifest_sha256=inference.manifest_sha256,
    )
    checkpoint_dataset = loaded.payload["dataset"]
    if checkpoint_dataset["split_assignment_sha256"] != inference.split_assignment_sha256:
        raise ValueError("checkpoint and inference use different split assignments")
    if checkpoint_dataset["native_dataset_sha256"] != inference.native_dataset_sha256:
        raise ValueError("checkpoint and inference use different native datasets")
    if checkpoint_dataset["point_counts"] != inference.point_counts:
        raise ValueError("checkpoint and inference use different point counts")
    cfg = loaded.payload["config"]
    scale_factor = float(cfg["dataset"].get("pcd_scale_factor", 1.0))
    if scale_factor <= 0:
        raise ValueError("checkpoint dataset pcd_scale_factor must be positive")

    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    maximum_peak_vram = 0.0
    for sample_index, sample in enumerate(inference.samples_by_split["test"]):
        observation = load_fm_observation(sample, scale_factor=scale_factor)
        result = predict_fm_pose_candidates(
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
                "split": sample.split,
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
    duration = time.perf_counter() - started
    if len(records) != EXPECTED_SPLIT_COUNTS["test"]:
        raise RuntimeError("FM prediction did not cover the complete test split")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve(strict=True)
    report = {
        "schema": REPORT_SCHEMA,
        "quality_claim": False,
        "quality_assessment": "inconclusive-until-supervision-isolated-evaluation",
        "model": FM_MODEL_CAPABILITY,
        "architecture": {
            "point_encoder": cfg["model"]["point_encoder"],
            "fm_solver": cfg["model"]["fm_solver"],
            "fm_steps": int(cfg["model"]["fm_steps"]),
            "fm_noise_scale": float(cfg["model"]["fm_noise_scale"]),
            "pcd_scale_factor": scale_factor,
            "rotation_corruption": False,
        },
        "ground_truth_free": True,
        "training_manifest_read": False,
        "target_pose_or_goal_point_cloud_read": False,
        "training_manifest_sha256": checkpoint_dataset["training_manifest_sha256"],
        "inference_manifest": str(inference.manifest_path),
        "inference_manifest_sha256": inference.manifest_sha256,
        "split_assignment_sha256": inference.split_assignment_sha256,
        "native_dataset_sha256": inference.native_dataset_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_model_state_sha256": loaded.model_state_sha256,
        "checkpoint_reload_verified": True,
        "checkpoint_selection_split": loaded.payload["training"]["selection_split"],
        "test_split_used_for_checkpoint_selection": loaded.payload["training"][
            "test_split_used_for_selection"
        ],
        "test_sample_count": len(records),
        "candidate_count_per_sample": args.candidates,
        "finite_prediction_gate_passed": True,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "peak_vram_mib": maximum_peak_vram if device.type == "cuda" else None,
        "total_duration_seconds": duration,
        "planner_called": False,
        "simulator_called": False,
        "predictions": records,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
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
