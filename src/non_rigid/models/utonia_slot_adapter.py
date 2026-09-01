"""Slot-preserving utilities for an optional Utonia/PTv3 encoder.

Utonia is intentionally an optional dependency.  This module contains the
tensor seam only: callers may provide a Utonia ``Point``-like mapping returned
by a separately managed Python environment.  No Utonia import occurs here, so
the TAX-DPD/FM baseline remains runnable without ``spconv``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def build_utonia_input(coord: np.ndarray) -> dict[str, np.ndarray]:
    """Build the official zero-color/zero-normal input for one role cloud."""

    coordinates = np.asarray(coord)
    if (
        coordinates.dtype != np.float32
        or coordinates.ndim != 2
        or coordinates.shape[1:] != (3,)
        or len(coordinates) < 4
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("coord must be finite float32 [N, 3] with N >= 4")
    coordinates = np.ascontiguousarray(coordinates)
    return {
        "coord": coordinates,
        # Utonia's documented custom-data contract requires these keys; absent
        # modalities are represented by zeros, never by TAX-DPD flow features.
        "color": np.zeros_like(coordinates),
        "normal": np.zeros_like(coordinates),
    }


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite [N, C] tensor")
    return value


def _inverse(value: Any, count: int, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    inverse = value
    if inverse.ndim == 2 and inverse.shape[1] == 1:
        inverse = inverse[:, 0]
    elif inverse.ndim != 1:
        raise ValueError(f"{name} must have shape [N] or [N, 1]")
    if inverse.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError(f"{name} must use an integer dtype")
    inverse = inverse.long()
    if inverse.shape[0] == 0 or torch.any(inverse < 0) or torch.any(inverse >= count):
        raise ValueError(f"{name} contains an out-of-range index")
    return inverse


def upcast_slot_features(
    point: Mapping[str, Any],
    *,
    concatenate_levels: int = 0,
) -> torch.Tensor:
    """Map hierarchical Utonia features back to the original ordered slots.

    The mapping mirrors Utonia's documented ``pooling_inverse`` chain followed
    by the GridSample ``inverse``.  ``concatenate_levels`` controls how many
    skip levels are concatenated; zero keeps only the final feature width.
    The input mapping is never mutated.
    """

    if isinstance(concatenate_levels, bool) or concatenate_levels < 0:
        raise ValueError("concatenate_levels must be a non-negative integer")
    current: Mapping[str, Any] = point
    feature = _tensor(current.get("feat"), "point.feat")
    level = 0
    while "pooling_parent" in current:
        parent = current["pooling_parent"]
        if not isinstance(parent, Mapping):
            raise TypeError("pooling_parent must be a Point-like mapping")
        parent_feature = _tensor(parent.get("feat"), "pooling_parent.feat")
        inverse = _inverse(
            current.get("pooling_inverse"), len(feature), "pooling_inverse"
        )
        if inverse.shape[0] != len(parent_feature):
            raise ValueError(
                "pooling_inverse length must match the parent feature count"
            )
        lifted = feature[inverse]
        feature = (
            torch.cat((parent_feature, lifted), dim=1)
            if level < concatenate_levels
            else lifted
        )
        current = parent
        level += 1
    if "inverse" in current:
        original_inverse = _inverse(current["inverse"], len(feature), "inverse")
        feature = feature[original_inverse]
    if not torch.isfinite(feature).all():
        raise ValueError("upcast slot features contain non-finite values")
    return feature


def validate_role_slots(
    features: torch.Tensor,
    coordinates: np.ndarray,
    *,
    role: str,
) -> torch.Tensor:
    """Assert that a role feature tensor still has the input slot count."""

    if not isinstance(role, str) or not role:
        raise ValueError("role must be a non-empty string")
    coords = np.asarray(coordinates)
    _tensor(features, f"{role}.features")
    if features.shape[0] != len(coords):
        raise ValueError(
            f"{role} feature count {features.shape[0]} does not match coordinates {len(coords)}"
        )
    return features


__all__ = [
    "build_utonia_input",
    "upcast_slot_features",
    "validate_role_slots",
]
