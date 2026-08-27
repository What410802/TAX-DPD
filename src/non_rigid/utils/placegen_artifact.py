"""Validated observation-only artifacts exchanged with PlaceGen.

The TAX-DPD process deliberately receives no goal point cloud or target pose.
Training supervision stays in the RPDiff-compatible dataset, while this module
defines the smaller deployment seam used for pose prediction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


TAXDPD_EXPORT_PROFILE = "placegen.taxdpd-rpdiff-overfit/0.1"
MODEL_INPUT_PROFILE = "placegen.taxdpd-model-input/0.1"
OBSERVATION_FIELDS = frozenset(
    {
        "action_indices",
        "anchor_indices",
        "child_points_world",
        "parent_points_world",
        "scene_center_world",
        "source_world_from_object",
    }
)


def sha256_file(path: Path) -> str:
    """Hash one regular file without following an ambiguous caller path."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """Hash an array including dtype and shape, not only its raw bytes."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def model_input_sha256(pc_action: np.ndarray, pc_anchor: np.ndarray) -> str:
    """Canonical combined identity for the two tensors seen by TAX-DPD."""

    digest = hashlib.sha256()
    digest.update(MODEL_INPUT_PROFILE.encode())
    digest.update(b"\0")
    for name, value in (("pc_action", pc_action), ("pc_anchor", pc_anchor)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must use float32, got {array.dtype}")
    if array.ndim != 2 or array.shape[1:] != (3,) or len(array) < 4:
        raise ValueError(f"{name} must have shape [N, 3] with N >= 4")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=np.float32, copy=True, order="C")


def _indices(value: np.ndarray, expected_count: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if len(array) != expected_count:
        raise ValueError(f"{name} length does not match its point cloud")
    if np.any(array < 0) or len(np.unique(array)) != len(array):
        raise ValueError(f"{name} must contain unique non-negative indices")
    return np.array(array, dtype=np.int64, copy=True, order="C")


def _se3(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must use float64, got {matrix.dtype}")
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite [4, 4] matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9, rtol=0.0):
        raise ValueError(f"{name} must have the homogeneous SE(3) bottom row")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6, rtol=0.0):
        raise ValueError(f"{name} rotation must have determinant +1")
    return np.array(matrix, dtype=np.float64, copy=True, order="C")


@dataclass(frozen=True)
class PlaceGenObservationArtifact:
    """One validated request and the provenance needed for a smoke report."""

    sample_id: str
    descriptor_sha256: str
    input_npz_sha256: str
    child_points_world: np.ndarray
    parent_points_world: np.ndarray
    scene_center_world: np.ndarray
    source_world_from_object: np.ndarray
    action_indices: np.ndarray
    anchor_indices: np.ndarray
    pc_action_centered: np.ndarray
    pc_anchor_centered: np.ndarray
    tensor_sha256: dict[str, str]
    supervision_tensor_sha256: str | None
    model_input_sha256: str

    def model_input_metadata(self) -> dict[str, Any]:
        return {
            "profile": MODEL_INPUT_PROFILE,
            "frame": "scene-centered-world",
            "dtype": "float32",
            "action_point_count": int(len(self.pc_action_centered)),
            "anchor_point_count": int(len(self.pc_anchor_centered)),
            "tensor_sha256": dict(self.tensor_sha256),
        }


def load_placegen_observation(
    input_npz: Path | str,
    manifest_path: Path | str,
) -> PlaceGenObservationArtifact:
    """Load an observation-only request and validate it against its manifest."""

    supplied_input = Path(input_npz).expanduser()
    supplied_manifest = Path(manifest_path).expanduser()
    for path, name in ((supplied_input, "input NPZ"), (supplied_manifest, "manifest")):
        if path.is_symlink():
            raise ValueError(f"{name} cannot be a symlink: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    input_path = supplied_input.resolve()
    manifest = _mapping(
        json.loads(supplied_manifest.read_text(encoding="utf-8")),
        "manifest",
    )
    if manifest.get("profile") != TAXDPD_EXPORT_PROFILE:
        raise ValueError(
            f"unexpected TAX-DPD export profile: {manifest.get('profile')!r}"
        )
    sample_id = manifest.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("manifest sample_id must be a non-empty string")
    native = _mapping(manifest.get("native"), "manifest.native")
    descriptor_sha256 = _sha256(
        native.get("descriptor_sha256"),
        "manifest.native.descriptor_sha256",
    )
    files = _mapping(manifest.get("files"), "manifest.files")
    inference = _mapping(files.get("inference_npz"), "manifest.files.inference_npz")
    declared_path = inference.get("path")
    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError("manifest inference NPZ path must be non-empty")
    manifest_root = supplied_manifest.resolve().parent
    expected_path = (manifest_root / declared_path).resolve()
    try:
        expected_path.relative_to(manifest_root)
    except ValueError as error:
        raise ValueError(
            "manifest inference NPZ path escapes the manifest directory"
        ) from error
    if expected_path != input_path:
        raise ValueError(
            f"input NPZ {input_path} does not match manifest-declared path {expected_path}"
        )
    expected_npz_sha256 = _sha256(
        inference.get("sha256"),
        "manifest.files.inference_npz.sha256",
    )
    actual_npz_sha256 = sha256_file(input_path)
    if actual_npz_sha256 != expected_npz_sha256:
        raise ValueError("input NPZ SHA-256 does not match the export manifest")

    with np.load(input_path, allow_pickle=False) as data:
        fields = frozenset(data.files)
        if fields != OBSERVATION_FIELDS:
            missing = sorted(OBSERVATION_FIELDS.difference(fields))
            unexpected = sorted(fields.difference(OBSERVATION_FIELDS))
            raise ValueError(
                "TAX-DPD inference NPZ must contain exactly observation fields; "
                f"missing={missing}, unexpected={unexpected}"
            )
        child_world = _points(data["child_points_world"], "child_points_world")
        parent_world = _points(data["parent_points_world"], "parent_points_world")
        action_indices = _indices(
            data["action_indices"], len(child_world), "action_indices"
        )
        anchor_indices = _indices(
            data["anchor_indices"], len(parent_world), "anchor_indices"
        )
        center = np.asarray(data["scene_center_world"])
        if center.dtype != np.dtype(np.float64):
            raise ValueError(f"scene_center_world must use float64, got {center.dtype}")
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("scene_center_world must be a finite [3] vector")
        scene_center = np.array(center, dtype=np.float64, copy=True, order="C")
        source_pose = _se3(data["source_world_from_object"], "source_world_from_object")

    point_counts = _mapping(manifest.get("point_counts"), "manifest.point_counts")
    if point_counts.get("action") != len(child_world):
        raise ValueError("manifest action point count does not match the inference NPZ")
    if point_counts.get("anchor") != len(parent_world):
        raise ValueError("manifest anchor point count does not match the inference NPZ")
    recomputed_center = (
        np.concatenate(
            (child_world.astype(np.float64), parent_world.astype(np.float64)),
            axis=0,
        )
        .mean(axis=0)
        .astype(np.float32)
    )
    if not np.array_equal(scene_center, recomputed_center.astype(np.float64)):
        raise ValueError(
            "scene_center_world does not exactly match the exported point clouds"
        )

    center_float32 = scene_center.astype(np.float32)
    pc_action = np.ascontiguousarray(child_world - center_float32, dtype=np.float32)
    pc_anchor = np.ascontiguousarray(parent_world - center_float32, dtype=np.float32)
    tensor_hashes = {
        "pc_action_centered": sha256_array(pc_action),
        "pc_anchor_centered": sha256_array(pc_anchor),
    }
    declared_tensor_hashes = _mapping(
        manifest.get("tensor_sha256"),
        "manifest.tensor_sha256",
    )
    for name, actual in tensor_hashes.items():
        expected = _sha256(
            declared_tensor_hashes.get(name),
            f"manifest.tensor_sha256.{name}",
        )
        if actual != expected:
            raise ValueError(f"{name} SHA-256 does not match the export manifest")
    declared_goal_hash = declared_tensor_hashes.get("pc_goal_centered")
    if declared_goal_hash is not None:
        declared_goal_hash = _sha256(
            declared_goal_hash,
            "manifest.tensor_sha256.pc_goal_centered",
        )

    for array in (
        child_world,
        parent_world,
        scene_center,
        source_pose,
        action_indices,
        anchor_indices,
        pc_action,
        pc_anchor,
    ):
        array.setflags(write=False)
    return PlaceGenObservationArtifact(
        sample_id=sample_id,
        descriptor_sha256=descriptor_sha256,
        input_npz_sha256=actual_npz_sha256,
        child_points_world=child_world,
        parent_points_world=parent_world,
        scene_center_world=scene_center,
        source_world_from_object=source_pose,
        action_indices=action_indices,
        anchor_indices=anchor_indices,
        pc_action_centered=pc_action,
        pc_anchor_centered=pc_anchor,
        tensor_sha256=tensor_hashes,
        supervision_tensor_sha256=declared_goal_hash,
        model_input_sha256=model_input_sha256(pc_action, pc_anchor),
    )


__all__ = [
    "MODEL_INPUT_PROFILE",
    "OBSERVATION_FIELDS",
    "TAXDPD_EXPORT_PROFILE",
    "PlaceGenObservationArtifact",
    "load_placegen_observation",
    "model_input_sha256",
    "sha256_array",
    "sha256_file",
]
