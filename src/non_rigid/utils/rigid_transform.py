"""Small, auditable rigid-transform utilities for ordered point predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RigidTransformFit:
    """A proper SE(3) fit and diagnostics for ordered source/target points."""

    matrix: np.ndarray
    rmse: float
    singular_values: np.ndarray


def estimate_ordered_rigid_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    degeneracy_tolerance: float = 1.0e-10,
) -> RigidTransformFit:
    """Fit ``target ~= R @ source + t`` with reflection-corrected Kabsch.

    Planar point sets are accepted because two non-zero singular directions are
    sufficient for the plate-placement smoke.  Collinear and coincident inputs
    fail closed instead of returning an arbitrary rotation.
    """

    source_points = np.asarray(source, dtype=np.float64)
    target_points = np.asarray(target, dtype=np.float64)
    if source_points.ndim != 2 or source_points.shape[1:] != (3,):
        raise ValueError("source must have shape [N, 3]")
    if target_points.shape != source_points.shape:
        raise ValueError("target must have the same [N, 3] shape as source")
    if len(source_points) < 4:
        raise ValueError("at least four ordered point pairs are required")
    if not np.isfinite(source_points).all() or not np.isfinite(target_points).all():
        raise ValueError("ordered point pairs must be finite")
    if not np.isfinite(degeneracy_tolerance) or degeneracy_tolerance <= 0.0:
        raise ValueError("degeneracy_tolerance must be a positive finite number")

    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    covariance = source_centered.T @ target_centered
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    if singular_values[1] <= degeneracy_tolerance:
        raise ValueError("ordered source points are collinear or coincident")

    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_transpose[-1] *= -1.0
        rotation = right_transpose.T @ left.T
    translation = target_center - rotation @ source_center

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    fitted = source_points @ rotation.T + translation
    rmse = float(np.sqrt(np.mean(np.sum((fitted - target_points) ** 2, axis=1))))
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(rmse)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6, rtol=0.0)
    ):
        raise RuntimeError("Kabsch fit did not produce a finite proper SE(3) transform")
    return RigidTransformFit(
        matrix=matrix,
        rmse=rmse,
        singular_values=np.asarray(singular_values, dtype=np.float64),
    )


__all__ = ["RigidTransformFit", "estimate_ordered_rigid_transform"]
