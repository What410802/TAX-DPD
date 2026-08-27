"""Ground-truth-free pose extraction from reconstructed TAX-DPD candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from non_rigid.models.tax3d_v2 import TAX3Dv2FixedFrameModule
from non_rigid.utils.placegen_artifact import PlaceGenObservationArtifact
from non_rigid.utils.placegen_request import PREDICT_SCORE_KIND
from non_rigid.utils.rigid_transform import estimate_ordered_rigid_transform


def _translation_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(value, dtype=np.float64)
    return matrix


def _quaternion_wxyz(rotation: np.ndarray) -> list[float]:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    value = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    value /= np.linalg.norm(value)
    if value[0] < 0.0:
        value *= -1.0
    if not np.isfinite(value).all() or not np.isclose(
        np.linalg.norm(value), 1.0, atol=1e-9
    ):
        raise RuntimeError("pose conversion produced an invalid unit quaternion")
    return [float(component) for component in value]


@dataclass(frozen=True)
class CandidatePredictionResult:
    """One prediction call with ordered pose candidates and runtime evidence."""

    candidates: tuple[dict[str, Any], ...]
    latency_seconds: float
    peak_vram_mib: float | None


def predict_pose_candidates(
    module: TAX3Dv2FixedFrameModule,
    artifact: PlaceGenObservationArtifact,
    device: torch.device,
    *,
    candidate_count: int,
    seed: int,
) -> CandidatePredictionResult:
    """Sample target-free goal points and fit one proper world-frame pose each."""

    action = (
        torch.from_numpy(np.array(artifact.pc_action_centered, copy=True, order="C"))
        .unsqueeze(0)
        .to(device)
    )
    anchor = (
        torch.from_numpy(np.array(artifact.pc_anchor_centered, copy=True, order="C"))
        .unsqueeze(0)
        .to(device)
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        predicted = module.sample_candidates(
            action,
            anchor,
            num_trials=candidate_count,
            seed=seed,
        )[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency = time.perf_counter() - started
    predicted_numpy = predicted.detach().cpu().numpy().astype(np.float64)
    center_to_world = _translation_matrix(artifact.scene_center_world)
    world_to_center = _translation_matrix(-artifact.scene_center_world)
    records: list[dict[str, Any]] = []
    for index, target_centered in enumerate(predicted_numpy):
        fit = estimate_ordered_rigid_transform(
            artifact.pc_action_centered.astype(np.float64, copy=False),
            target_centered,
        )
        relative_world = center_to_world @ fit.matrix @ world_to_center
        absolute_world = relative_world @ artifact.source_world_from_object
        position = absolute_world[:3, 3]
        if not np.isfinite(position).all():
            raise RuntimeError("pose prediction produced a non-finite world position")
        records.append(
            {
                "candidate_index": index,
                "position_world_m": [float(value) for value in position],
                "quaternion_wxyz": _quaternion_wxyz(absolute_world[:3, :3]),
                "score": -float(fit.rmse),
                "score_kind": PREDICT_SCORE_KIND,
            }
        )
    if len(records) != candidate_count:
        raise RuntimeError("TAX-DPD returned an unexpected number of candidates")
    return CandidatePredictionResult(
        candidates=tuple(records),
        latency_seconds=latency,
        peak_vram_mib=(
            float(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
            if device.type == "cuda"
            else None
        ),
    )


__all__ = ["CandidatePredictionResult", "predict_pose_candidates"]
