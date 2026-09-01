"""Metric contracts shared by FM/DDPM grouped evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.placegen_evaluate_fm_grouped import _ordered_point_rmse_mm


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
