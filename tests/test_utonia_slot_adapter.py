"""Utonia slot/inverse contracts without importing CUDA extensions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from non_rigid.models.utonia_slot_adapter import (
    build_utonia_input,
    upcast_slot_features,
    validate_role_slots,
)


def test_build_utonia_input_uses_zero_missing_modalities() -> None:
    coord = np.arange(12, dtype=np.float32).reshape(4, 3)
    point = build_utonia_input(coord)
    assert set(point) == {"coord", "color", "normal"}
    np.testing.assert_array_equal(point["coord"], coord)
    np.testing.assert_array_equal(point["color"], np.zeros((4, 3), np.float32))
    np.testing.assert_array_equal(point["normal"], np.zeros((4, 3), np.float32))


def test_upcast_pooling_and_grid_inverse_preserve_original_slot_order() -> None:
    # Four original slots -> three voxel slots -> two pooled slots.
    root = {
        "feat": torch.tensor([[10.0], [20.0], [30.0]]),
        "inverse": torch.tensor([[2], [0], [2], [1]]),
    }
    pooled = {
        "feat": torch.tensor([[100.0], [200.0]]),
        "pooling_parent": root,
        "pooling_inverse": torch.tensor([[1], [0], [1]]),
    }
    # The three grid slots receive pooled features [200, 100, 200], then grid
    # inverse restores the requested original order [slot2, slot0, slot2, slot1].
    output = upcast_slot_features(pooled)
    torch.testing.assert_close(output[:, 0], torch.tensor([200.0, 200.0, 200.0, 100.0]))
    assert pooled["feat"].shape == (2, 1)  # input mapping was not mutated


def test_upcast_can_concatenate_one_skip_level() -> None:
    parent = {"feat": torch.tensor([[1.0], [2.0]])}
    point = {
        "feat": torch.tensor([[10.0], [20.0], [30.0]]),
        "pooling_parent": parent,
        "pooling_inverse": torch.tensor([[2], [0]]),
    }
    output = upcast_slot_features(point, concatenate_levels=1)
    torch.testing.assert_close(output, torch.tensor([[1.0, 30.0], [2.0, 10.0]]))


def test_slot_adapter_rejects_mismatched_or_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="float32"):
        build_utonia_input(np.zeros((4, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="feature count"):
        validate_role_slots(torch.zeros(3, 2), np.zeros((4, 3), np.float32), role="action")
    with pytest.raises(ValueError, match="out-of-range"):
        upcast_slot_features(
            {
                "feat": torch.ones(2, 1),
                "inverse": torch.tensor([[0], [3]]),
            }
        )
