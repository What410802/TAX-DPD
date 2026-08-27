"""Strict online PlaceGen-to-TAX-DPD prediction request contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from non_rigid.utils.placegen_artifact import (
    OBSERVATION_FIELDS,
    PlaceGenObservationArtifact,
    model_input_sha256,
    sha256_array,
    sha256_file,
)
from non_rigid.utils.placegen_grouped import MODEL_CAPABILITY, canonical_json_sha256

PREDICT_REQUEST_PROFILE = "placegen.taxdpd-predict-request/0.1"
PREDICT_RESPONSE_PROFILE = "placegen.taxdpd-predict-response/0.1"
PREDICT_SCORE_KIND = "negative_ordered_rigid_fit_rmse_m"
MAX_CANDIDATE_COUNT = 32
_REQUEST_FIELDS = frozenset(
    {
        "profile",
        "request_id",
        "model",
        "checkpoint_sha256",
        "input_npz",
        "seed",
        "candidate_count",
    }
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    fields = frozenset(value)
    if fields != expected:
        missing = sorted(expected.difference(fields))
        unexpected = sorted(fields.difference(expected))
        raise ValueError(
            f"{name} fields must match schema exactly; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def canonical_request_id(
    *,
    model: str,
    checkpoint_sha256: str,
    input_npz_sha256: str,
    seed: int,
    candidate_count: int,
) -> str:
    """Return the path-independent identity agreed with the PlaceGen adapter."""

    return canonical_json_sha256(
        {
            "model": model,
            "checkpoint_sha256": checkpoint_sha256,
            "input_npz_sha256": input_npz_sha256,
            "seed": seed,
            "candidate_count": candidate_count,
        }
    )


@dataclass(frozen=True)
class PredictionRequest:
    """A validated target-free request and its byte identity."""

    request_path: Path
    request_sha256: str
    request_id: str
    checkpoint_sha256: str
    input_npz: Path
    input_npz_sha256: str
    seed: int
    candidate_count: int


def load_prediction_request(path: Path | str) -> PredictionRequest:
    """Load an exact request, reject path escapes, symlinks and identity drift."""

    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise ValueError(f"prediction request cannot be a symlink: {supplied}")
    if not supplied.is_file():
        raise FileNotFoundError(f"prediction request does not exist: {supplied}")
    request_path = supplied.resolve(strict=True)
    try:
        request = _mapping(
            json.loads(request_path.read_text(encoding="utf-8")),
            "prediction request",
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("prediction request is not valid UTF-8 JSON") from error
    _exact_fields(request, _REQUEST_FIELDS, "prediction request")
    if request.get("profile") != PREDICT_REQUEST_PROFILE:
        raise ValueError("unexpected prediction request profile")
    if request.get("model") != MODEL_CAPABILITY:
        raise ValueError("prediction request uses an unsupported model capability")
    checkpoint_sha = _sha256(request.get("checkpoint_sha256"), "checkpoint_sha256")
    request_id = _sha256(request.get("request_id"), "request_id")
    input_record = _mapping(request.get("input_npz"), "input_npz")
    _exact_fields(input_record, frozenset({"path", "sha256"}), "input_npz")
    declared_path = input_record.get("path")
    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError("input_npz.path must be a non-empty relative path")
    input_path = request_path.parent / declared_path
    if input_path.is_symlink():
        raise ValueError(f"prediction input cannot be a symlink: {input_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"prediction input does not exist: {input_path}")
    input_path = input_path.resolve(strict=True)
    try:
        input_path.relative_to(request_path.parent)
    except ValueError as error:
        raise ValueError("input_npz.path escapes the request directory") from error
    input_sha = _sha256(input_record.get("sha256"), "input_npz.sha256")
    if sha256_file(input_path) != input_sha:
        raise ValueError("prediction input SHA-256 does not match the request")
    seed = request.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    candidate_count = request.get("candidate_count")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or not 1 <= candidate_count <= MAX_CANDIDATE_COUNT
    ):
        raise ValueError(f"candidate_count must be in [1, {MAX_CANDIDATE_COUNT}]")
    expected_request_id = canonical_request_id(
        model=MODEL_CAPABILITY,
        checkpoint_sha256=checkpoint_sha,
        input_npz_sha256=input_sha,
        seed=seed,
        candidate_count=candidate_count,
    )
    if request_id != expected_request_id:
        raise ValueError("request_id does not match the canonical request identity")
    return PredictionRequest(
        request_path=request_path,
        request_sha256=sha256_file(request_path),
        request_id=request_id,
        checkpoint_sha256=checkpoint_sha,
        input_npz=input_path,
        input_npz_sha256=input_sha,
        seed=seed,
        candidate_count=candidate_count,
    )


def _points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.dtype(np.float32)
        or array.ndim != 2
        or array.shape[1:] != (3,)
        or len(array) < 4
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be finite float32 [N, 3] with N >= 4")
    return np.array(array, dtype=np.float32, copy=True, order="C")


def _indices(value: np.ndarray, count: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.int64) or array.shape != (count,):
        raise ValueError(f"{name} must be int64 [{count}]")
    if np.any(array < 0) or len(np.unique(array)) != count:
        raise ValueError(f"{name} must contain unique non-negative indices")
    return np.array(array, dtype=np.int64, copy=True, order="C")


def load_request_observation(request: PredictionRequest) -> PlaceGenObservationArtifact:
    """Parse the six-field request NPZ without permitting any goal information."""

    if sha256_file(request.input_npz) != request.input_npz_sha256:
        raise ValueError("prediction input changed after request validation")
    with np.load(request.input_npz, allow_pickle=False) as data:
        fields = frozenset(data.files)
        if fields != OBSERVATION_FIELDS:
            missing = sorted(OBSERVATION_FIELDS.difference(fields))
            unexpected = sorted(fields.difference(OBSERVATION_FIELDS))
            raise ValueError(
                "prediction NPZ must contain exactly observation fields; "
                f"missing={missing}, unexpected={unexpected}"
            )
        child = _points(data["child_points_world"], "child_points_world")
        parent = _points(data["parent_points_world"], "parent_points_world")
        action_indices = _indices(data["action_indices"], len(child), "action_indices")
        anchor_indices = _indices(data["anchor_indices"], len(parent), "anchor_indices")
        center_value = np.asarray(data["scene_center_world"])
        if (
            center_value.dtype != np.dtype(np.float64)
            or center_value.shape != (3,)
            or not np.isfinite(center_value).all()
        ):
            raise ValueError("scene_center_world must be finite float64 [3]")
        center = np.array(center_value, dtype=np.float64, copy=True, order="C")
        source_value = np.asarray(data["source_world_from_object"])
        if (
            source_value.dtype != np.dtype(np.float64)
            or source_value.shape != (4, 4)
            or not np.isfinite(source_value).all()
        ):
            raise ValueError("source_world_from_object must be finite float64 [4, 4]")
        rotation = source_value[:3, :3]
        if (
            not np.array_equal(source_value[3], np.asarray([0.0, 0.0, 0.0, 1.0]))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6, rtol=0.0)
        ):
            raise ValueError("source_world_from_object must be a proper SE(3) matrix")
        source_pose = np.array(source_value, dtype=np.float64, copy=True, order="C")
    recomputed_center = (
        np.concatenate((child.astype(np.float64), parent.astype(np.float64)))
        .mean(axis=0)
        .astype(np.float32)
        .astype(np.float64)
    )
    if not np.array_equal(center, recomputed_center):
        raise ValueError(
            "scene_center_world does not exactly match observed point clouds"
        )
    center32 = center.astype(np.float32)
    action = np.ascontiguousarray(child - center32, dtype=np.float32)
    anchor = np.ascontiguousarray(parent - center32, dtype=np.float32)
    hashes = {
        "pc_action_centered": sha256_array(action),
        "pc_anchor_centered": sha256_array(anchor),
    }
    for array in (
        child,
        parent,
        center,
        source_pose,
        action_indices,
        anchor_indices,
        action,
        anchor,
    ):
        array.setflags(write=False)
    return PlaceGenObservationArtifact(
        sample_id=request.request_id,
        descriptor_sha256=request.input_npz_sha256,
        input_npz_sha256=request.input_npz_sha256,
        child_points_world=child,
        parent_points_world=parent,
        scene_center_world=center,
        source_world_from_object=source_pose,
        action_indices=action_indices,
        anchor_indices=anchor_indices,
        pc_action_centered=action,
        pc_anchor_centered=anchor,
        tensor_sha256=hashes,
        supervision_tensor_sha256=None,
        model_input_sha256=model_input_sha256(action, anchor),
    )


__all__ = [
    "MAX_CANDIDATE_COUNT",
    "PREDICT_REQUEST_PROFILE",
    "PREDICT_RESPONSE_PROFILE",
    "PREDICT_SCORE_KIND",
    "PredictionRequest",
    "canonical_request_id",
    "load_prediction_request",
    "load_request_observation",
]
