"""Tests for the batch seam of the PlaceGen TAX3Dv2 runner."""

from __future__ import annotations

import torch

from scripts.train_placegen_tax3dv2 import _batch, _batched_indices


class _Dataset:
    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "pc": torch.full((4, 3), float(index)),
            "sample_id": f"sample-{index}",
        }


def test_batched_indices_keeps_last_partial_batch() -> None:
    assert _batched_indices(10, 4) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_batch_collates_tensors_and_utonia_sample_ids() -> None:
    batch = _batch(_Dataset(), [2, 5], torch.device("cpu"))
    assert batch["pc"].shape == (2, 4, 3)
    assert batch["pc"].dtype == torch.float32
    assert batch["sample_id"] == ["sample-2", "sample-5"]
    assert torch.all(batch["pc"][0] == 2.0)
    assert torch.all(batch["pc"][1] == 5.0)
