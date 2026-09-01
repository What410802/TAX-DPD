"""Tests for the Utonia/TAX3Dv2 feature seam without CUDA extensions."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from non_rigid.models.encoders import (
    CachedUtoniaSlotFeatureEncoder,
    UtoniaJointFeatureEncoder,
    UtoniaSlotFeatureEncoder,
    utonia_geometry_sha256,
)
from non_rigid.models.tax3d_v2 import TAX3Dv2FixedFrameFlowMatchingModule


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
    assert slots.call_slot_counts == [9, 9]


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


def test_fm_velocity_model_prepares_utonia_context_once_for_many_nfes() -> None:
    class FeatureEncoder:
        def __init__(self):
            self.calls = 0

        def prepare_static_context(self, x0, y):
            self.calls += 1
            return (x0[:, :1], y[:, :1])

    class Network:
        def __init__(self):
            self.dit = SimpleNamespace(feature_encoder=FeatureEncoder())
            self.static_ids: list[int] = []

        def __call__(self, frame, shape, time, *, static_geometry, **kwargs):
            del time, kwargs
            self.static_ids.append(id(static_geometry))
            return torch.zeros_like(frame), torch.zeros_like(shape)

    network = Network()
    module = SimpleNamespace(
        network=network,
        model_cfg=SimpleNamespace(point_encoder="utonia"),
        fm_time_scale=100.0,
    )
    kwargs = {"x0": torch.randn(1, 3, 4), "y": torch.randn(1, 3, 5)}
    velocity = TAX3Dv2FixedFrameFlowMatchingModule._velocity_model(module, kwargs)
    state = torch.randn(1, 3, 5)
    velocity(state, torch.tensor([0.25]))
    velocity(state, torch.tensor([0.75]))
    assert network.dit.feature_encoder.calls == 1
    assert len(set(network.static_ids)) == 1


def test_model_cfg_without_runtime_checkpoint_fails_closed() -> None:
    try:
        UtoniaJointFeatureEncoder(3, 8, _cfg())
    except ValueError as error:
        assert "utonia_checkpoint" in str(error)
    else:
        raise AssertionError("missing Utonia checkpoint must fail closed")


def test_cached_utonia_encoder_loads_exact_geometry(tmp_path) -> None:
    scene = torch.randn(1, 3, 9)
    key = utonia_geometry_sha256(scene)
    features = np.arange(9 * 576, dtype=np.float32).reshape(9, 576)
    feature_path = tmp_path / f"{key}.npz"
    np.savez(feature_path, features=features)
    digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "taxdpd.utonia-static-feature-cache/0.1",
        "feature_width": 576,
        "records": [
            {
                "sample_id": "sample-0",
                "split": "train",
                "geometry_sha256": key,
                "feature_npz": feature_path.name,
                "feature_npz_sha256": digest,
                "slot_count": 9,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    encoder = CachedUtoniaSlotFeatureEncoder(
        8, SimpleNamespace(utonia_feature_manifest=str(manifest_path))
    )
    actual = encoder.frozen_features(scene)
    torch.testing.assert_close(actual, torch.from_numpy(features).t().unsqueeze(0))


def test_geometry_cache_key_tolerates_float32_centering_roundoff() -> None:
    scene = torch.randn(1, 3, 9)
    perturbed = scene + torch.full_like(scene, 4.0e-7)
    assert utonia_geometry_sha256(scene) == utonia_geometry_sha256(perturbed)
