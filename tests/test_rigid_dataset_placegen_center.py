"""Compatibility contract for PlaceGen's canonical TAX-DPD scene center."""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("pytorch3d")

from non_rigid.datasets.rigid import RPDiffDataset


def test_rpdiff_dataset_uses_exported_placegen_center(tmp_path: Path) -> None:
    task_dir = tmp_path / "placegen_plate_smoke" / "task_name_plate_on_rack"
    split_dir = task_dir / "split_info"
    split_dir.mkdir(parents=True)
    filename = "sample-0.npz"
    rng = np.random.default_rng(31)
    child = rng.normal(size=(512, 3)).astype(np.float32)
    parent = rng.normal(size=(1024, 3)).astype(np.float32)
    goal = (child + np.asarray([0.2, -0.1, 0.3], dtype=np.float32)).astype(np.float32)
    canonical_center = (
        np.concatenate((child.astype(np.float64), parent.astype(np.float64)), axis=0)
        .mean(axis=0)
        .astype(np.float32)
    )
    np.savez(
        task_dir / filename,
        multi_obj_start_pcd=np.asarray(
            {"parent": parent, "child": child}, dtype=object
        ),
        multi_obj_final_pcd=np.asarray({"parent": parent, "child": goal}, dtype=object),
        placegen_scene_center_world=canonical_center,
    )
    for split_name in ("train_split.txt", "train_val_split.txt", "test_split.txt"):
        (split_dir / split_name).write_text(filename + "\n", encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "name": "rpdiff",
            "rpdiff_task_name": "placegen_plate_smoke",
            "rpdiff_task_type": "task_name_plate_on_rack",
            "num_demos": None,
            "train_dataset_size": 1,
            "val_dataset_size": 1,
            "test_dataset_size": 1,
            "val_use_defaults": False,
            "preprocess": False,
            "augment_gt": False,
            "sample_size_action": 512,
            "sample_size_anchor": 1024,
            "downsample_type": "fps",
            "pred_frame": "anchor_center",
            "noisy_goal_scale": 0.0,
            "scene_transform_type": "identity",
            "translation_variance": 0.0,
            "rotation_variance": 0.0,
            "action_plane_occlusion": 0.0,
            "action_plane_standoff": 0.04,
            "action_ball_occlusion": 0.0,
            "action_ball_radius": 0.1,
            "anchor_plane_occlusion": 0.0,
            "anchor_plane_standoff": 0.04,
            "anchor_ball_occlusion": 0.0,
            "anchor_ball_radius": 0.1,
        }
    )

    item = RPDiffDataset(tmp_path, cfg, "train")[0]

    np.testing.assert_array_equal(item["pc_action"].numpy(), child - canonical_center)
    np.testing.assert_array_equal(item["pc_anchor"].numpy(), parent - canonical_center)
    np.testing.assert_array_equal(item["pc"].numpy(), goal - canonical_center)
