"""Tests for the Utonia/TAX3Dv2 feature seam without CUDA extensions."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from non_rigid.models.encoders import (
    UtoniaJointFeatureEncoder,
    UtoniaSlotFeatureEncoder,
)


class FakeSlotEncoder(nn.Module):
    def __init__(self, width: int, output_width: int = 8):
        super().__init__()
        self.projection = nn.Conv1d(3, output_width, 1)
        self.width = width
        self.call_slot_counts: list[int] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.call_slot_counts.append(value.shape[-1])
        return self.projection(value[:, :3, :])


def _cfg(point_type: str = "flow") -> SimpleNamespace:
    return SimpleNamespace(type=point_type, point_encoder="utonia")


def test_joint_utonia_encoder_keeps_role_slots_and_flow_mlp() -> None:
    slots = FakeSlotEncoder(4)
    encoder = UtoniaJointFeatureEncoder(
        3,
        8,
        _cfg(),
        slot_encoder=slots,
    )
    x0 = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 5)
    x = torch.randn(2, 3, 4)
    action, anchor = encoder(x, y, x0)
    assert action.shape == (2, 4, 8)
    assert anchor.shape == (2, 5, 8)
    assert torch.isfinite(action).all()
    assert torch.isfinite(anchor).all()
    # Static action+anchor geometry is encoded jointly once, preserving
    # cross-role neighborhoods without rerunning the frozen backbone.
    assert slots.call_slot_counts == [9]


def test_joint_utonia_encoder_does_not_mutate_inputs() -> None:
    encoder = UtoniaJointFeatureEncoder(
        3,
        8,
        _cfg(),
        slot_encoder=FakeSlotEncoder(4),
    )
    x0 = torch.randn(1, 3, 4)
    y = torch.randn(1, 3, 4)
    x = torch.randn(1, 3, 4)
    before = (x.clone(), y.clone(), x0.clone())
    encoder(x, y, x0)
    for actual, expected in zip((x, y, x0), before):
        torch.testing.assert_close(actual, expected)


def test_static_context_is_reused_without_backbone_calls() -> None:
    slots = FakeSlotEncoder(4)
    encoder = UtoniaJointFeatureEncoder(3, 8, _cfg(), slot_encoder=slots)
    x0 = torch.randn(1, 3, 4)
    y = torch.randn(1, 3, 5)
    static = encoder.prepare_static_context(x0, y)
    assert slots.call_slot_counts == [9]
    encoder(torch.randn_like(x0), y, x0, static_geometry=static)
    encoder(torch.randn_like(x0), y, x0, static_geometry=static)
    assert slots.call_slot_counts == [9]


def test_frozen_backbone_stays_eval_when_parent_trains() -> None:
    backbone = nn.Linear(3, 3)
    cfg = SimpleNamespace(
        utonia_input_scale_factor=15.0,
        utonia_transform_scale=0.5,
        utonia_transform_seed=1701,
    )
    encoder = UtoniaSlotFeatureEncoder(
        8,
        cfg,
        backbone=backbone,
        transform=lambda point: point,
    )
    encoder.train()
    assert encoder.training is True
    assert backbone.training is False


def test_model_cfg_without_runtime_checkpoint_fails_closed() -> None:
    try:
        UtoniaJointFeatureEncoder(3, 8, _cfg())
    except ValueError as error:
        assert "utonia_checkpoint" in str(error)
    else:
        raise AssertionError("missing Utonia checkpoint must fail closed")
