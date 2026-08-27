"""Tests for the observation-only PlaceGen/TAX-DPD interchange artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from non_rigid.utils.placegen_artifact import (
    TAXDPD_EXPORT_PROFILE,
    load_placegen_observation,
    model_input_sha256,
    sha256_array,
    sha256_file,
)


def _write_artifact(root: Path, *, extra_field: bool = False) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    child = np.asarray(
        [[0.0, 0.0, 0.1], [0.1, 0.0, 0.1], [0.0, 0.1, 0.1], [0.1, 0.1, 0.1]],
        dtype=np.float32,
    )
    parent = np.asarray(
        [[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0], [-0.2, 0.2, 0.0], [0.2, 0.2, 0.0]],
        dtype=np.float32,
    )
    center = (
        np.concatenate(
            (child.astype(np.float64), parent.astype(np.float64)),
            axis=0,
        )
        .mean(axis=0)
        .astype(np.float32)
    )
    source_pose = np.eye(4, dtype=np.float64)
    inference = root / "inference.npz"
    arrays: dict[str, np.ndarray] = {
        "action_indices": np.arange(len(child), dtype=np.int64),
        "anchor_indices": np.arange(len(parent), dtype=np.int64),
        "child_points_world": child,
        "parent_points_world": parent,
        "scene_center_world": np.asarray(center, dtype=np.float64),
        "source_world_from_object": source_pose,
    }
    if extra_field:
        arrays["child_goal_world"] = child.copy()
    np.savez(inference, **arrays)
    manifest = root / "manifest.json"
    manifest_payload = {
        "profile": TAXDPD_EXPORT_PROFILE,
        "sample_id": "sample-0",
        "point_counts": {"action": len(child), "anchor": len(parent)},
        "native": {"descriptor_sha256": "a" * 64},
        "files": {
            "inference_npz": {
                "path": inference.name,
                "sha256": sha256_file(inference),
            }
        },
        "tensor_sha256": {
            "pc_action_centered": sha256_array(child - center.astype(np.float32)),
            "pc_anchor_centered": sha256_array(parent - center.astype(np.float32)),
        },
    }
    manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    return inference, manifest


def test_observation_loader_validates_manifest_and_input_digest(tmp_path: Path) -> None:
    inference, manifest = _write_artifact(tmp_path)
    artifact = load_placegen_observation(inference, manifest)

    assert artifact.sample_id == "sample-0"
    assert artifact.descriptor_sha256 == "a" * 64
    assert artifact.model_input_sha256 == model_input_sha256(
        artifact.pc_action_centered,
        artifact.pc_anchor_centered,
    )
    assert artifact.pc_action_centered.dtype == np.float32
    assert artifact.pc_anchor_centered.dtype == np.float32


def test_observation_loader_rejects_goal_fields(tmp_path: Path) -> None:
    inference, manifest = _write_artifact(tmp_path, extra_field=True)
    with pytest.raises(ValueError, match="exactly observation fields"):
        load_placegen_observation(inference, manifest)


def test_observation_loader_rejects_tampered_input(tmp_path: Path) -> None:
    inference, manifest = _write_artifact(tmp_path)
    with inference.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_placegen_observation(inference, manifest)
