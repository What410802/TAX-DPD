"""Focused tests for the TAX3Dv2 fixed-frame FM adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from non_rigid.models.tax3d_v2 import TAX3Dv2FixedFrameFlowMatchingModule


def _cfg(*, learn_sigma: bool = False, rotation_noise: object = False):
    model = SimpleNamespace(
        learn_sigma=learn_sigma,
        diff_rotation_noise_scale=rotation_noise,
        fm_steps=4,
        fm_solver="heun",
        fm_time_scale=100.0,
        zero_shape=True,
        diff_train_steps=100,
        diff_inference_steps=100,
        diff_noise_scale=1.0,
    )
    training = SimpleNamespace(
        lr=1.0e-4,
        weight_decay=1.0e-5,
        num_training_steps=10,
        lr_warmup_steps=0,
        batch_size=1,
        val_batch_size=1,
        sample_size=8,
        sample_size_anchor=8,
        num_wta_trials=1,
    )
    return SimpleNamespace(model=model, training=training)


class _UnusedNetwork(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("not called by this contract test")


def test_fm_adapter_requires_three_channel_head_and_no_rotation_corruption() -> None:
    with pytest.raises(ValueError, match="learn_sigma=false"):
        TAX3Dv2FixedFrameFlowMatchingModule(_UnusedNetwork(), _cfg(learn_sigma=True))
    with pytest.raises(ValueError, match="rotation corruption"):
        TAX3Dv2FixedFrameFlowMatchingModule(
            _UnusedNetwork(), _cfg(rotation_noise=45.0)
        )


def test_frame_shape_decomposition_round_trips_and_preserves_zero_mean() -> None:
    target = torch.randn(2, 3, 16)
    frame, shape = TAX3Dv2FixedFrameFlowMatchingModule._split_target(target)
    assert torch.allclose(frame + shape, target)
    assert torch.allclose(shape.mean(dim=2), torch.zeros(2, 3), atol=1.0e-7)

    module = TAX3Dv2FixedFrameFlowMatchingModule(_UnusedNetwork(), _cfg())
    source_frame, source_shape = module._source(target)
    assert source_frame.shape == (2, 3, 1)
    assert source_shape.shape == target.shape
    assert torch.allclose(source_shape.mean(dim=2), torch.zeros(2, 3), atol=1.0e-7)


def test_inference_source_shape_can_be_built_from_observation_only() -> None:
    module = TAX3Dv2FixedFrameFlowMatchingModule(_UnusedNetwork(), _cfg())
    observed_action = torch.randn(2, 3, 16)
    source_frame, source_shape = module._source(observed_action)
    assert source_frame.shape == (2, 3, 1)
    assert source_shape.shape == observed_action.shape
