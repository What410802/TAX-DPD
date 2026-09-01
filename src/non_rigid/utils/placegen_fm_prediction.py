"""Target-free FM goal sampling and world-frame pose extraction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from non_rigid.models.tax3d_v2 import TAX3Dv2FixedFrameFlowMatchingModule
from non_rigid.utils.placegen_fm_export import FMObservation
from non_rigid.utils.rigid_transform import estimate_ordered_rigid_transform


def _translation_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(value, dtype=np.float64)
    return matrix


def _quaternion_wxyz(rotation: np.ndarray) -> list[float]:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    quaternion = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return [float(value) for value in quaternion]


@dataclass(frozen=True)
class FMPredictionResult:
    candidates: tuple[dict[str, Any], ...]
    latency_seconds: float
    peak_vram_mib: float | None


def predict_fm_pose_candidates(
    module: TAX3Dv2FixedFrameFlowMatchingModule,
    observation: FMObservation,
    device: torch.device,
    *,
    candidate_count: int,
    seed: int,
) -> FMPredictionResult:
    """Sample ordered goal points and compose absolute world poses."""

    action = torch.from_numpy(observation.action_model.copy()).unsqueeze(0).to(device)
    anchor = torch.from_numpy(observation.anchor_model.copy()).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        predictions = module.sample_candidates(
            action,
            anchor,
            num_trials=candidate_count,
            seed=seed,
        )[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency = time.perf_counter() - started
    predictions_model = predictions.detach().cpu().numpy().astype(np.float64)
    action_centered_m = observation.action_model.astype(np.float64) / observation.scale_factor
    center_to_world = _translation_matrix(observation.model_center_world)
    world_to_center = _translation_matrix(-observation.model_center_world)
    records: list[dict[str, Any]] = []
    for candidate_index, prediction_model in enumerate(predictions_model):
        prediction_centered_m = prediction_model / observation.scale_factor
        fit = estimate_ordered_rigid_transform(action_centered_m, prediction_centered_m)
        relative_world = center_to_world @ fit.matrix @ world_to_center
        absolute_world = relative_world @ observation.source_world_from_object
        if not np.isfinite(absolute_world).all():
            raise RuntimeError("FM pose composition produced non-finite values")
        records.append(
            {
                "candidate_index": candidate_index,
                "candidate_seed": seed + candidate_index,
                "world_from_object": absolute_world.tolist(),
                "relative_world_from_source": relative_world.tolist(),
                "position_world_m": [float(value) for value in absolute_world[:3, 3]],
                "quaternion_wxyz": _quaternion_wxyz(absolute_world[:3, :3]),
                "ordered_rigid_fit_rmse_m": float(fit.rmse),
                "score": -float(fit.rmse),
                "score_kind": "negative_ordered_rigid_fit_rmse_m",
            }
        )
    if len(records) != candidate_count:
        raise RuntimeError("FM sampler returned an unexpected number of candidates")
    return FMPredictionResult(
        candidates=tuple(records),
        latency_seconds=latency,
        peak_vram_mib=(
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
    )


__all__ = ["FMPredictionResult", "predict_fm_pose_candidates"]
