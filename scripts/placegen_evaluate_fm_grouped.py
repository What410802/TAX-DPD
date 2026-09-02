"""Score a frozen target-free prediction report against PlaceGen supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from non_rigid.utils.placegen_fm_export import (
    EXPECTED_SPLIT_COUNTS,
    load_native_export_identity,
    sha256_file,
)

PREDICTION_SCHEMAS = {
    "placegen.tax3dv2-fm-grouped-prediction-report/0.1":
        "placegen.tax3dv2-fm-test-evaluation/0.1",
    "placegen.tax3dv2-ddpm-grouped-prediction-report/0.1":
        "placegen.tax3dv2-ddpm-test-evaluation/0.1",
}


def _pose_errors(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    translation_mm = float(np.linalg.norm(predicted[:3, 3] - target[:3, 3]) * 1000.0)
    cosine = float(
        np.clip((np.trace(predicted[:3, :3] @ target[:3, :3].T) - 1.0) / 2.0, -1.0, 1.0)
    )
    return translation_mm, float(np.degrees(np.arccos(cosine)))


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _ordered_point_rmse_mm(
    predicted_world: np.ndarray,
    target_world: np.ndarray,
) -> float:
    """RMS Euclidean point error, matching PlaceGen's common evaluator."""

    if predicted_world.shape != target_world.shape or predicted_world.ndim != 2:
        raise ValueError("ordered point arrays must have matching [N, D] shapes")
    if not np.isfinite(predicted_world).all() or not np.isfinite(target_world).all():
        raise ValueError("ordered point arrays must be finite")
    return float(
        np.sqrt(np.mean(np.sum((predicted_world - target_world) ** 2, axis=1)))
        * 1000.0
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_json).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evaluation report: {output}")
    identity = load_native_export_identity(
        args.training_manifest, args.inference_manifest
    )
    prediction_path = Path(args.prediction_report).expanduser().resolve(strict=True)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction_schema = prediction.get("schema")
    if prediction_schema not in PREDICTION_SCHEMAS:
        raise ValueError("unexpected FM prediction report schema")
    if prediction.get("ground_truth_free") is not True:
        raise ValueError("prediction report is not target-free")
    if prediction.get("inference_manifest_sha256") != identity.inference_manifest_sha256:
        raise ValueError("prediction and evaluator use different inference manifests")
    if prediction.get("split_assignment_sha256") != identity.split_assignment_sha256:
        raise ValueError("prediction and evaluator use different split assignments")
    split = prediction.get("prediction_split", args.split)
    if split != args.split:
        raise ValueError(
            f"prediction report split {split!r} does not match evaluator split {args.split!r}"
        )
    records = prediction.get("predictions")
    if not isinstance(records, list) or len(records) != EXPECTED_SPLIT_COUNTS[split]:
        raise ValueError(f"prediction report does not cover the complete {split} split")

    training_manifest = json.loads(identity.training_manifest_path.read_text(encoding="utf-8"))
    samples = {
        sample["sample_id"]: sample
        for sample in training_manifest["samples"]
        if sample["split"] == split
    }
    rows: list[dict[str, float | str]] = []
    for record in records:
        sample_id = record.get("sample_id")
        if sample_id not in samples or record.get("split") != split:
            raise ValueError(f"unknown/non-{split} prediction sample: {sample_id!r}")
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"prediction sample {sample_id} has no candidates")
        if not 0 <= args.candidate_index < len(candidates):
            raise ValueError(f"candidate index is unavailable for {sample_id}")
        candidate = candidates[args.candidate_index]
        if candidate.get("candidate_index") != args.candidate_index:
            raise ValueError("prediction candidate ordering is inconsistent")
        predicted_pose = np.asarray(candidate.get("world_from_object"), dtype=np.float64)
        if predicted_pose.shape != (4, 4) or not np.isfinite(predicted_pose).all():
            raise ValueError("predicted world_from_object must be finite [4, 4]")
        training_record = samples[sample_id]["training_npz"]
        training_path = identity.training_manifest_path.parent / training_record["path"]
        if training_path.is_symlink() or not training_path.is_file():
            raise FileNotFoundError(f"training supervision is missing: {training_path}")
        training_path = training_path.resolve(strict=True)
        if sha256_file(training_path) != training_record["sha256"]:
            raise ValueError("training supervision SHA-256 differs from its manifest")
        with np.load(training_path, allow_pickle=False) as data:
            source_pose = np.asarray(data["source_world_from_object"])
            target_pose = np.asarray(data["target_world_from_object"])
            source_world = np.asarray(data["action_points_source_world"], dtype=np.float64)
            target_world = np.asarray(data["action_points_target_world"], dtype=np.float64)
        translation_mm, rotation_deg = _pose_errors(predicted_pose, target_pose)
        object_points = (
            np.concatenate((source_world, np.ones((len(source_world), 1))), axis=1)
            @ np.linalg.inv(source_pose).T
        )
        predicted_world = (object_points @ predicted_pose.T)[:, :3]
        ordered_rmse_mm = _ordered_point_rmse_mm(predicted_world, target_world)
        pred_to_target = cKDTree(target_world).query(predicted_world, k=1)[0]
        target_to_pred = cKDTree(predicted_world).query(target_world, k=1)[0]
        symmetric_chamfer_mm = float(
            (pred_to_target.mean() + target_to_pred.mean()) * 500.0
        )
        rows.append(
            {
                "sample_id": sample_id,
                "translation_error_mm": translation_mm,
                "rotation_error_deg": rotation_deg,
                "ordered_point_rmse_mm": ordered_rmse_mm,
                "symmetric_chamfer_mm": symmetric_chamfer_mm,
            }
        )
    metrics = {
        name: _summary([float(row[name]) for row in rows])
        for name in (
            "translation_error_mm",
            "rotation_error_deg",
            "ordered_point_rmse_mm",
            "symmetric_chamfer_mm",
        )
    }
    metrics.update(
        {
            "pass_translation_lt20mm": sum(
                float(row["translation_error_mm"]) < 20.0 for row in rows
            ),
            "pass_rotation_lt10deg": sum(
                float(row["rotation_error_deg"]) < 10.0 for row in rows
            ),
            "pass_point_rmse_lt20mm": sum(
                float(row["ordered_point_rmse_mm"]) < 20.0 for row in rows
            ),
        }
    )
    evaluation_schema = PREDICTION_SCHEMAS[prediction_schema].replace(
        "-test-evaluation/", f"-{split}-evaluation/"
    )
    report = {
        "schema": evaluation_schema,
        "model": prediction.get("model"),
        "quality_claim": False,
        "prediction_report": str(prediction_path),
        "prediction_report_sha256": sha256_file(prediction_path),
        "checkpoint_sha256": prediction["checkpoint_sha256"],
        "training_manifest_sha256": identity.training_manifest_sha256,
        "inference_manifest_sha256": identity.inference_manifest_sha256,
        "prediction_split": split,
        "prediction_sample_count": len(rows),
        "test_sample_count": len(rows) if split == "test" else None,
        "candidate_index": args.candidate_index,
        "candidate_selection": "fixed-index-no-oracle",
        "metrics": metrics,
        "samples": rows,
        "ground_truth_used_by_evaluator": True,
        "planner_called": False,
        "simulator_called": False,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--prediction-report", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
