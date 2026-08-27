"""Contracts for ground-truth-free fixed-frame TAX-DPD sampling."""

from types import SimpleNamespace

import lightning as L
import pytest
import torch
from torch import nn

pytest.importorskip("pytorch3d")
pytest.importorskip("torch_geometric")
pytest.importorskip("rpad")

from non_rigid.models.tax3d_v2 import TAX3Dv2FixedFrameModule


class _UnusedNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))


class _RecordingDiffusion:
    def __init__(self) -> None:
        self.initial_noise: list[tuple[torch.Tensor, torch.Tensor]] = []

    def p_sample_loop_progressive(
        self,
        model: nn.Module,
        shape_r: torch.Size,
        shape_s: torch.Size,
        *,
        noise_r: torch.Tensor,
        noise_s: torch.Tensor,
        **_: object,
    ):
        del model
        assert tuple(shape_r) == tuple(noise_r.shape)
        assert tuple(shape_s) == tuple(noise_s.shape)
        self.initial_noise.append((noise_r.clone(), noise_s.clone()))
        yield {"sample_r": noise_r, "sample_s": noise_s}


def _module() -> tuple[TAX3Dv2FixedFrameModule, _RecordingDiffusion]:
    module = object.__new__(TAX3Dv2FixedFrameModule)
    L.LightningModule.__init__(module)
    module.network = _UnusedNetwork()
    module.model_cfg = SimpleNamespace(zero_shape=True)
    module.noise_scale = 1.0
    diffusion = _RecordingDiffusion()
    module.diffusion = diffusion
    return module, diffusion


def test_sample_candidates_needs_no_ground_truth_and_is_seed_stable() -> None:
    module, diffusion = _module()
    action = torch.linspace(-1.0, 1.0, 24).reshape(2, 4, 3)
    anchor = torch.linspace(-2.0, 2.0, 30).reshape(2, 5, 3)

    first = module.sample_candidates(action, anchor, num_trials=2, seed=17)
    second = module.sample_candidates(action, anchor, num_trials=2, seed=17)
    extended = module.sample_candidates(action, anchor, num_trials=3, seed=17)

    assert first.shape == (2, 2, 4, 3)
    assert torch.equal(first, second)
    assert torch.equal(first, extended[:, :2])
    assert not torch.equal(first[:, 0], first[:, 1])
    assert len(diffusion.initial_noise) == 7
    for _, shape_noise in diffusion.initial_noise:
        torch.testing.assert_close(
            shape_noise.mean(dim=2),
            torch.zeros_like(shape_noise.mean(dim=2)),
            atol=1.0e-6,
            rtol=0.0,
        )


@pytest.mark.parametrize(
    ("action", "anchor", "message"),
    [
        (torch.zeros(4, 3), torch.zeros(1, 4, 3), "pc_action must have shape"),
        (torch.zeros(1, 3, 3), torch.zeros(1, 4, 3), "at least four points"),
        (torch.zeros(1, 4, 3), torch.zeros(2, 4, 3), "batch sizes must match"),
    ],
)
def test_sample_candidates_rejects_invalid_inputs(
    action: torch.Tensor,
    anchor: torch.Tensor,
    message: str,
) -> None:
    module, _ = _module()

    with pytest.raises(ValueError, match=message):
        module.sample_candidates(action, anchor, seed=0)


def test_sample_candidates_rejects_non_finite_points() -> None:
    module, _ = _module()
    action = torch.zeros(1, 4, 3)
    action[0, 0, 0] = torch.nan

    with pytest.raises(ValueError, match="non-finite"):
        module.sample_candidates(action, torch.zeros(1, 4, 3), seed=0)
