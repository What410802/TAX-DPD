"""Fail-closed contracts for grouped PlaceGen/TAX-DPD experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from non_rigid.utils.placegen_grouped import (
    MODEL_CAPABILITY,
    canonical_json_sha256,
    load_grouped_inference_manifest,
    load_grouped_observation,
    load_grouped_training_manifest,
    select_validation_checkpoint,
    sha256_file,
)
from non_rigid.utils.placegen_request import (
    PREDICT_REQUEST_PROFILE,
    canonical_request_id,
    load_prediction_request,
)


def _observation_arrays(offset: float) -> dict[str, np.ndarray]:
    child = np.asarray(
        [
            [offset + 0.00, 0.00, 0.10],
            [offset + 0.10, 0.00, 0.10],
            [offset + 0.00, 0.10, 0.10],
            [offset + 0.10, 0.10, 0.10],
        ],
        dtype=np.float32,
    )
    parent = np.asarray(
        [
            [-0.20, -0.20, 0.00],
            [0.20, -0.20, 0.00],
            [-0.20, 0.20, 0.00],
            [0.20, 0.20, 0.00],
        ],
        dtype=np.float32,
    )
    center = (
        np.concatenate((child.astype(np.float64), parent.astype(np.float64)))
        .mean(axis=0)
        .astype(np.float32)
    )
    return {
        "action_indices": np.arange(4, dtype=np.int64),
        "anchor_indices": np.arange(4, dtype=np.int64),
        "child_points_world": child,
        "parent_points_world": parent,
        "scene_center_world": center.astype(np.float64),
        "source_world_from_object": np.eye(4, dtype=np.float64),
    }


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_grouped_export(
    root: Path,
    *,
    group_counts: dict[str, int] | None = None,
) -> tuple[Path, Path]:
    counts = group_counts or {"train": 72, "validation": 12, "test": 12}
    task_name = "placegen_plate_generalization"
    task_type = "task_name_plate_on_rack"
    dataset_root = root / task_name / task_type
    inference_root = root / "inference"
    dataset_root.mkdir(parents=True)
    inference_root.mkdir()
    samples: list[dict[str, object]] = []
    inference_samples: list[dict[str, object]] = []
    split_names: dict[str, list[str]] = {name: [] for name in counts}
    sample_index = 0
    for split, count in counts.items():
        (inference_root / split).mkdir()
        for group_index in range(count):
            sample_id = f"{split}-{group_index:03d}"
            filename = f"{sample_id}.npz"
            split_group_id = f"group-{split}-{group_index:03d}"
            observation = _observation_arrays(sample_index * 0.001)
            child = observation["child_points_world"]
            parent = observation["parent_points_world"]
            center = observation["scene_center_world"].astype(np.float32)
            goal = child + np.asarray([0.01, 0.02, 0.03], dtype=np.float32)
            training_path = dataset_root / filename
            np.savez(
                training_path,
                multi_obj_final_obj_pose=np.asarray(
                    {"parent": np.zeros(7), "child": np.zeros(7)}, dtype=object
                ),
                multi_obj_final_pcd=np.asarray(
                    {"parent": parent, "child": goal}, dtype=object
                ),
                multi_obj_names=np.asarray(
                    {"parent": "rack", "child": "plate"}, dtype=object
                ),
                multi_obj_start_obj_pose=np.asarray(
                    {"parent": np.zeros(7), "child": np.zeros(7)}, dtype=object
                ),
                multi_obj_start_pcd=np.asarray(
                    {"parent": parent, "child": child}, dtype=object
                ),
                placegen_scene_center_world=center,
            )
            inference_path = inference_root / split / filename
            np.savez(inference_path, **observation)
            tensors = {
                "pc_action_centered": _array_sha256(child - center),
                "pc_anchor_centered": _array_sha256(parent - center),
                "pc_goal_centered": _array_sha256(goal - center),
            }
            record = {
                "sample_id": sample_id,
                "split": split,
                "split_group_id": split_group_id,
                "descriptor": {"path": f"native/{sample_id}.json", "sha256": "a" * 64},
                "physical_setup": {
                    "setup_id": split_group_id,
                    "setup_sha256": "b" * 64,
                    "physical_state_sha256": "c" * 64,
                },
                "training_npz": {
                    "path": training_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(training_path),
                },
                "inference_npz": {
                    "path": inference_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(inference_path),
                },
                "tensor_sha256": tensors,
                "oracle_kabsch_round_trip": {
                    "translation_error_mm": 0.0,
                    "rotation_error_deg": 0.0,
                },
            }
            samples.append(record)
            inference_samples.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "split_group_id": split_group_id,
                    "descriptor_sha256": "a" * 64,
                    "input_npz": {
                        "path": inference_path.relative_to(inference_root).as_posix(),
                        "sha256": sha256_file(inference_path),
                    },
                    "input_tensor_sha256": {
                        key: tensors[key]
                        for key in ("pc_action_centered", "pc_anchor_centered")
                    },
                }
            )
            split_names[split].append(filename)
            sample_index += 1
    split_dir = dataset_root / "split_info"
    split_dir.mkdir()
    filenames = {
        "train": "train_split.txt",
        "validation": "train_val_split.txt",
        "test": "test_split.txt",
    }
    for split, values in split_names.items():
        (split_dir / filenames[split]).write_text(
            "\n".join(values) + "\n", encoding="utf-8"
        )

    ledger = {"object_id": "plate", "support_id": "rack", "slot_id": "slot-left-0"}
    assignment_sha = "d" * 64
    manifest_payload = {
        "profile": "placegen.taxdpd-rpdiff-grouped/0.1",
        "task_name": task_name,
        "task_type": task_type,
        "sample_count": len(samples),
        "group_count": sum(counts.values()),
        "split_counts": counts,
        "point_counts": {"action": 4, "anchor": 4},
        "evaluation_valid": True,
        "correlated_split": False,
        "quality_claim": False,
        "quality_scope": "fixed-asset-fixed-slot-heldout-pose",
        "split_assignment": {
            "profile": "placegen.pose-supervision-split-assignment/0.1",
            "manifest_sha256": assignment_sha,
            "native_dataset_sha256": "e" * 64,
            "identity_ledger": ledger,
            "identity_ledger_sha256": canonical_json_sha256(ledger),
            "assignment_algorithm": "sha256-seeded-shuffle-v1",
            "seed": 23,
        },
        "oracle_kabsch_round_trip": {
            "translation_limit_mm": 0.1,
            "rotation_limit_deg": 0.1,
            "maximum_translation_error_mm": 0.0,
            "maximum_rotation_error_deg": 0.0,
            "all_samples_passed": True,
        },
        "samples": samples,
    }
    training_manifest = root / "manifest.json"
    training_manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inference_payload = {
        "profile": "placegen.taxdpd-inference-index/0.1",
        "quality_scope": "fixed-asset-fixed-slot-heldout-pose",
        "evaluation_valid": True,
        "correlated_split": False,
        "ground_truth_free": True,
        "sample_count": len(inference_samples),
        "point_counts": {"action": 4, "anchor": 4},
        "split_assignment": {
            "profile": "placegen.pose-supervision-split-assignment/0.1",
            "manifest_sha256": assignment_sha,
            "native_dataset_sha256": "e" * 64,
        },
        "samples": inference_samples,
    }
    inference_manifest = inference_root / "manifest.json"
    inference_manifest.write_text(
        json.dumps(inference_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return training_manifest, inference_manifest


def test_grouped_loaders_enforce_72_12_12_and_keep_test_out_of_selection(
    tmp_path: Path,
) -> None:
    training_path, inference_path = _write_grouped_export(tmp_path / "export")
    training = load_grouped_training_manifest(training_path)
    inference = load_grouped_inference_manifest(inference_path)

    assert training.group_counts == {"train": 72, "validation": 12, "test": 12}
    assert len(training.samples_by_split["train"]) == 72
    assert len(training.samples_by_split["validation"]) == 12
    assert len(training.samples_by_split["test"]) == 12
    assert training.split_assignment_sha256 == "d" * 64
    assert inference.split_assignment_sha256 == training.split_assignment_sha256
    decision = select_validation_checkpoint(
        previous=None,
        epoch=1,
        validation_candidate0_ordered_rmse_m=0.04,
        validation_sample_ids=tuple(
            sample.sample_id for sample in training.samples_by_split["validation"]
        ),
    )
    assert decision.epoch == 1
    with pytest.raises(ValueError, match="validation sample IDs"):
        select_validation_checkpoint(
            previous=decision,
            epoch=2,
            validation_candidate0_ordered_rmse_m=0.01,
            validation_sample_ids=(training.samples_by_split["test"][0].sample_id,),
            expected_validation_sample_ids=tuple(
                sample.sample_id for sample in training.samples_by_split["validation"]
            ),
        )


def test_grouped_training_loader_rejects_npz_tamper_and_extra_manifest_fields(
    tmp_path: Path,
) -> None:
    training_path, _ = _write_grouped_export(
        tmp_path / "export", group_counts={"train": 2, "validation": 1, "test": 1}
    )
    payload = json.loads(training_path.read_text(encoding="utf-8"))
    payload["target_leak"] = True
    training_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        load_grouped_training_manifest(
            training_path,
            expected_group_counts={"train": 2, "validation": 1, "test": 1},
        )

    payload.pop("target_leak")
    training_path.write_text(json.dumps(payload), encoding="utf-8")
    first = tmp_path / "export" / payload["samples"][0]["training_npz"]["path"]
    with first.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        load_grouped_training_manifest(
            training_path,
            expected_group_counts={"train": 2, "validation": 1, "test": 1},
        )


def test_inference_loader_rejects_training_manifest_and_goal_fields(
    tmp_path: Path,
) -> None:
    training_path, inference_path = _write_grouped_export(
        tmp_path / "export", group_counts={"train": 2, "validation": 1, "test": 1}
    )
    with pytest.raises(ValueError, match="inference/manifest.json"):
        load_grouped_inference_manifest(training_path)

    index = load_grouped_inference_manifest(inference_path)
    sample = index.samples_by_split["test"][0]
    arrays = _observation_arrays(0.0)
    arrays["target_pose"] = np.eye(4)
    np.savez(sample.input_npz, **arrays)
    payload = json.loads(inference_path.read_text(encoding="utf-8"))
    record = next(
        item for item in payload["samples"] if item["sample_id"] == sample.sample_id
    )
    record["input_npz"]["sha256"] = sha256_file(sample.input_npz)
    inference_path.write_text(json.dumps(payload), encoding="utf-8")
    index = load_grouped_inference_manifest(inference_path)
    with pytest.raises(ValueError, match="exactly observation fields"):
        load_grouped_observation(index, sample.sample_id, required_split="test")


def test_request_loader_is_target_free_and_binds_canonical_identity(
    tmp_path: Path,
) -> None:
    request_root = tmp_path / "request"
    request_root.mkdir()
    input_path = request_root / "input.npz"
    np.savez(input_path, **_observation_arrays(0.0))
    identity_fields = {
        "model": MODEL_CAPABILITY,
        "checkpoint_sha256": "f" * 64,
        "input_npz_sha256": sha256_file(input_path),
        "seed": 7,
        "candidate_count": 3,
    }
    request = {
        "profile": PREDICT_REQUEST_PROFILE,
        "request_id": canonical_request_id(**identity_fields),
        "model": MODEL_CAPABILITY,
        "checkpoint_sha256": "f" * 64,
        "input_npz": {"path": "input.npz", "sha256": sha256_file(input_path)},
        "seed": 7,
        "candidate_count": 3,
    }
    request_path = request_root / "request.json"
    request_path.write_text(
        json.dumps(request, sort_keys=True) + "\n", encoding="utf-8"
    )
    loaded = load_prediction_request(request_path)
    assert loaded.request_id == request["request_id"]
    assert loaded.input_npz == input_path

    request["target_pose"] = np.eye(4).tolist()
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        load_prediction_request(request_path)
