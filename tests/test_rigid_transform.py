"""Unit contracts for ordered-point rigid decoding."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from non_rigid.utils.rigid_transform import estimate_ordered_rigid_transform


def test_kabsch_recovers_nontrivial_rigid_transform_for_planar_points() -> None:
    x, y = np.meshgrid(np.linspace(-0.2, 0.2, 8), np.linspace(-0.1, 0.1, 6))
    source = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    rotation = Rotation.from_euler("xyz", [0.37, -0.21, 0.52]).as_matrix()
    translation = np.array([0.18, -0.07, 0.31])
    target = source @ rotation.T + translation

    fit = estimate_ordered_rigid_transform(source, target)

    np.testing.assert_allclose(fit.matrix[:3, :3], rotation, atol=1.0e-10)
    np.testing.assert_allclose(fit.matrix[:3, 3], translation, atol=1.0e-10)
    assert fit.rmse < 1.0e-12
    assert np.linalg.det(fit.matrix[:3, :3]) == pytest.approx(1.0)


def test_kabsch_rejects_collinear_points() -> None:
    source = np.column_stack((np.linspace(0.0, 1.0, 8), np.zeros((8, 2))))

    with pytest.raises(ValueError, match="collinear or coincident"):
        estimate_ordered_rigid_transform(source, source + 0.1)


def test_kabsch_rejects_non_finite_points() -> None:
    source = np.eye(4, 3)
    target = source.copy()
    target[0, 0] = np.nan

    with pytest.raises(ValueError, match="must be finite"):
        estimate_ordered_rigid_transform(source, target)
