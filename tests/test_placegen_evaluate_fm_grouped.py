"""Metric contracts shared by FM/DDPM grouped evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.placegen_evaluate_fm_grouped import _ordered_point_rmse_mm, _success_flags


def test_ordered_rmse_is_rms_euclidean_not_coordinate_rmse() -> None:
    target = np.zeros((2, 3), dtype=np.float64)
    predicted = np.asarray([[0.003, 0.004, 0.0], [0.0, 0.0, 0.012]])

    assert _ordered_point_rmse_mm(predicted, target) == pytest.approx(
        np.sqrt((0.005**2 + 0.012**2) / 2.0) * 1000.0
    )


def test_ordered_rmse_rejects_shape_mismatch_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="matching"):
        _ordered_point_rmse_mm(np.zeros((2, 3)), np.zeros((3, 3)))
    invalid = np.zeros((2, 3))
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _ordered_point_rmse_mm(invalid, np.zeros((2, 3)))


def test_success_flags_use_strict_combined_task_thresholds() -> None:
    assert _success_flags(
        translation_error_mm=5.9,
        rotation_error_deg=2.8,
        ordered_point_rmse_mm=5.5,
        translation_threshold_mm=6.0,
        rotation_threshold_deg=float(np.degrees(0.05)),
        ordered_rmse_threshold_mm=6.0,
    ) == {
        "translation": True,
        "rotation": True,
        "ordered_rmse": True,
        "pose": True,
        "pose_and_ordered_rmse": True,
    }
    failed = _success_flags(
        translation_error_mm=6.0,
        rotation_error_deg=2.8,
        ordered_point_rmse_mm=5.5,
        translation_threshold_mm=6.0,
        rotation_threshold_deg=float(np.degrees(0.05)),
        ordered_rmse_threshold_mm=6.0,
    )
    assert failed["translation"] is False
    assert failed["pose"] is False
    assert failed["pose_and_ordered_rmse"] is False
