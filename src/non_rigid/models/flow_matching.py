"""Minimal Euclidean conditional flow matching for TAX-DPD ablations.

This module deliberately implements only the generator mechanism shared by the
controlled DDPM-vs-FM experiment.  It has no dependency on TAX-DPD's dataset or
DiT wrappers: callers provide a velocity model with the same point tensor shape.
Rotation-aware paths are excluded until their velocity target is specified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

VelocityModel = Callable[..., Tensor]


@dataclass(frozen=True)
class CondOTSample:
    """One linear conditional optimal-transport path sample."""

    time: Tensor
    state: Tensor
    velocity: Tensor


def _batch_time(time: Tensor, reference: Tensor) -> Tensor:
    if time.ndim != 1 or time.shape[0] != reference.shape[0]:
        raise ValueError("time must have shape [batch]")
    if not torch.isfinite(time).all() or torch.any(time < 0) or torch.any(time > 1):
        raise ValueError("time must be finite and lie in [0, 1]")
    return time.reshape((time.shape[0],) + (1,) * (reference.ndim - 1))


def sample_condot_path(source: Tensor, target: Tensor, time: Tensor) -> CondOTSample:
    """Sample ``x_t=(1-t)x_0+t*x_1`` and its exact velocity target."""

    if source.shape != target.shape or source.ndim < 2:
        raise ValueError("source and target must have the same batched tensor shape")
    if not torch.is_floating_point(source) or source.dtype != target.dtype:
        raise ValueError("source and target must use the same floating dtype")
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("source and target must be finite")
    expanded_time = _batch_time(time.to(device=source.device, dtype=source.dtype), source)
    velocity = target - source
    state = source + expanded_time * velocity
    return CondOTSample(time=time, state=state, velocity=velocity)


def velocity_mse(prediction: Tensor, target: Tensor) -> Tensor:
    """Return the per-example FM velocity mean-squared error."""

    if prediction.shape != target.shape:
        raise ValueError("predicted and target velocities must have matching shapes")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("velocities must be finite")
    return (prediction - target).square().flatten(start_dim=1).mean(dim=1)


def flow_matching_loss(
    model: VelocityModel,
    target: Tensor,
    *,
    source: Optional[Tensor] = None,
    time: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Tensor, CondOTSample, Tensor]:
    """Train a velocity model on the standard Euclidean CondOT path."""

    if source is None:
        source = torch.randn(
            target.shape,
            dtype=target.dtype,
            device=target.device,
            generator=generator,
        )
    if time is None:
        time = torch.rand(
            (target.shape[0],),
            dtype=target.dtype,
            device=target.device,
            generator=generator,
        )
    sample = sample_condot_path(source, target, time)
    prediction = model(sample.state, sample.time, **(model_kwargs or {}))
    return velocity_mse(prediction, sample.velocity), sample, prediction


@torch.no_grad()
def integrate_velocity(
    model: VelocityModel,
    source: Tensor,
    *,
    steps: int,
    method: str = "heun",
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tensor:
    """Integrate a learned velocity field from time 0 to 1.

    ``euler`` costs one model evaluation per step; ``heun`` costs two and is
    the default for the first TAX-DPD comparison.  NFE, not only step count,
    must be reported when comparing this solver with DDPM sampling.
    """

    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if method not in {"euler", "heun"}:
        raise ValueError("method must be 'euler' or 'heun'")
    if not torch.is_floating_point(source) or not torch.isfinite(source).all():
        raise ValueError("source must be a finite floating tensor")
    state = source.clone()
    batch = source.shape[0]
    dt = 1.0 / steps
    kwargs = model_kwargs or {}
    for index in range(steps):
        t0 = torch.full(
            (batch,), index * dt, dtype=source.dtype, device=source.device
        )
        velocity0 = model(state, t0, **kwargs)
        if velocity0.shape != state.shape or not torch.isfinite(velocity0).all():
            raise ValueError("velocity model must return a finite tensor matching state")
        if method == "euler":
            state = state + dt * velocity0
            continue
        predictor = state + dt * velocity0
        t1 = torch.full(
            (batch,), (index + 1) * dt, dtype=source.dtype, device=source.device
        )
        velocity1 = model(predictor, t1, **kwargs)
        if velocity1.shape != state.shape or not torch.isfinite(velocity1).all():
            raise ValueError("velocity model must return a finite tensor matching state")
        state = state + 0.5 * dt * (velocity0 + velocity1)
    return state


__all__ = [
    "CondOTSample",
    "flow_matching_loss",
    "integrate_velocity",
    "sample_condot_path",
    "velocity_mse",
]
