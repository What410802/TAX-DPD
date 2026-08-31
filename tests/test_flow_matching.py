"""Contract tests for the controlled Euclidean FM mechanism."""

from __future__ import annotations

import pytest
import torch

from non_rigid.models.flow_matching import (
    flow_matching_loss,
    integrate_velocity,
    sample_condot_path,
)


def test_condot_endpoints_and_velocity_match_finite_difference() -> None:
    source = torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])
    target = torch.tensor([[[4.0, 3.0], [2.0, 1.0]]])
    at_start = sample_condot_path(source, target, torch.tensor([0.0]))
    at_end = sample_condot_path(source, target, torch.tensor([1.0]))
    assert torch.equal(at_start.state, source)
    assert torch.equal(at_end.state, target)

    epsilon = 1.0e-3
    before = sample_condot_path(source, target, torch.tensor([0.4 - epsilon]))
    after = sample_condot_path(source, target, torch.tensor([0.4 + epsilon]))
    finite_difference = (after.state - before.state) / (2 * epsilon)
    assert torch.allclose(before.velocity, target - source)
    assert torch.allclose(finite_difference, before.velocity, atol=1.0e-4)


def test_flow_matching_loss_is_zero_for_exact_velocity() -> None:
    source = torch.randn(2, 3, 8)
    target = torch.randn(2, 3, 8)

    def exact(_state: torch.Tensor, _time: torch.Tensor, **_: object) -> torch.Tensor:
        return target - source

    losses, sample, prediction = flow_matching_loss(
        exact, target, source=source, time=torch.tensor([0.25, 0.75])
    )
    assert torch.equal(prediction, sample.velocity)
    assert torch.equal(losses, torch.zeros(2))


@pytest.mark.parametrize("method", ["euler", "heun"])
def test_solver_reaches_endpoint_for_constant_velocity(method: str) -> None:
    source = torch.randn(2, 3, 8)
    target = torch.randn(2, 3, 8)

    def constant(_state: torch.Tensor, _time: torch.Tensor, **_: object) -> torch.Tensor:
        return target - source

    actual = integrate_velocity(constant, source, steps=7, method=method)
    assert torch.allclose(actual, target, atol=2.0e-6)


def test_heun_converges_faster_on_time_varying_field() -> None:
    source = torch.zeros(1, 3, 4)

    def time_field(state: torch.Tensor, time: torch.Tensor, **_: object) -> torch.Tensor:
        return time.reshape(-1, 1, 1).expand_as(state)

    expected = torch.full_like(source, 0.5)
    euler = integrate_velocity(time_field, source, steps=4, method="euler")
    heun = integrate_velocity(time_field, source, steps=4, method="heun")
    assert torch.max(torch.abs(heun - expected)) < torch.max(torch.abs(euler - expected))
    assert torch.allclose(heun, expected, atol=1.0e-7)


def test_rotation_corruption_cannot_be_passed_as_an_implicit_option() -> None:
    source = torch.zeros(1, 3, 4)
    target = torch.ones_like(source)
    with pytest.raises(TypeError):
        flow_matching_loss(  # type: ignore[call-arg]
            lambda state, time: state,
            target,
            source=source,
            time=torch.tensor([0.5]),
            rotation_noise_scale=45.0,
        )
