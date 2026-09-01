"""Tests for the target-preserving PlaceGen TAX-DPD data seam."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from non_rigid.datasets.rigid import PlaceGenTaxDpdDataset


def _config():
    return SimpleNamespace(
        train_dataset_size=None,
        val_dataset_size=None,
        test_dataset_size=None,
        sample_size_action=4,
        sample_size_anchor=4,
        pcd_scale_factor=15.0,
    )


def _write(root, *, bad_object: bool = False):
    for split in ("train", "validation", "test"):
        directory = root / split
        directory.mkdir(parents=True)
        action = np.arange(12, dtype=np.float32).reshape(4, 3) / 100
        anchor = np.arange(12, 24, dtype=np.float32).reshape(4, 3) / 100
        goal = action + np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
        identity = np.eye(4, dtype=np.float64)
        values = {
            "action_points_source_world": action,
            "action_points_target_world": goal,
            "anchor_points_world": anchor,
            "source_world_from_object": identity,
            "target_world_from_object": identity,
            "shapenet_id": np.asarray("sample-0"),
        }
        if bad_object:
            values["shapenet_id"] = np.asarray({"unsafe": True}, dtype=object)
        np.savez(directory / "000000_final_obj_points.npz", **values)


def test_loader_preserves_ordered_flow_and_uses_joint_scene_center(tmp_path) -> None:
    _write(tmp_path)
    item = PlaceGenTaxDpdDataset(tmp_path, _config(), "train")[0]
    expected_flow = torch.tensor([1.5, -3.0, 4.5]).expand(4, 3)
    assert torch.allclose(item["flow"], expected_flow)
    assert torch.allclose(
        torch.cat((item["pc_action"], item["pc_anchor"]), dim=0).mean(dim=0),
        torch.zeros(3),
        atol=1.0e-7,
    )
    assert torch.allclose(item["pc"] - item["pc_action"], expected_flow)
    assert item["rpdiff_pcd_scale_factor"].item() == pytest.approx(15.0)


def test_loader_maps_validation_directory_and_rejects_pickle(tmp_path) -> None:
    _write(tmp_path)
    assert len(PlaceGenTaxDpdDataset(tmp_path, _config(), "val")) == 1

    bad_root = tmp_path / "bad"
    _write(bad_root, bad_object=True)
    with pytest.raises(ValueError, match="pickle-free"):
        PlaceGenTaxDpdDataset(bad_root, _config(), "train")[0]


def test_loader_requires_positive_scale_factor(tmp_path) -> None:
    _write(tmp_path)
    config = _config()
    config.pcd_scale_factor = 0.0
    with pytest.raises(ValueError, match="pcd_scale_factor"):
        PlaceGenTaxDpdDataset(tmp_path, config, "train")
