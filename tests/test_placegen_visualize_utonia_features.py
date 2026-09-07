"""Identity and publication contracts for cached Utonia feature visualization."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from non_rigid.models.encoders import utonia_geometry_sha256
from scripts.placegen_visualize_utonia_features import (
    SUMMARY_SCHEMA,
    load_visualization_data,
    run,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    sample_id = "rack-plate-final-blind-v2-000004"
    inference_root = tmp_path / "inference"
    input_path = inference_root / "test" / f"{sample_id}.npz"
    input_path.parent.mkdir(parents=True)
    child = np.arange(1024 * 3, dtype=np.float32).reshape(1024, 3) / 10000.0
    parent = child + np.float32(0.25)
    np.savez(
        input_path,
        action_indices=np.arange(1024, dtype=np.int64),
        anchor_indices=np.arange(1024, dtype=np.int64),
        child_points_world=child,
        parent_points_world=parent,
        scene_center_world=np.zeros(3, dtype=np.float64),
        source_world_from_object=np.eye(4, dtype=np.float64),
    )
    input_sha = _sha256(input_path)
    inference_path = inference_root / "manifest.json"
    inference = {
        "profile": "placegen.taxpose-inference-index/0.1",
        "ground_truth_free": True,
        "samples": [
            {
                "sample_id": sample_id,
                "split": "test",
                "input_npz": {"path": f"test/{sample_id}.npz", "sha256": input_sha},
            }
        ],
    }
    inference_path.write_text(json.dumps(inference), encoding="utf-8")
    scene = np.concatenate((child * 15.0, parent * 15.0), axis=0)
    scene -= scene.mean(axis=0, dtype=np.float32)
    geometry_sha = utonia_geometry_sha256(torch.from_numpy(scene).t().unsqueeze(0))
    feature_root = tmp_path / "cache"
    feature_root.mkdir()
    feature_path = feature_root / f"{geometry_sha}.npz"
    features = np.arange(2048 * 576, dtype=np.float32).reshape(2048, 576) / 100.0
    np.savez_compressed(feature_path, features=features)
    cache_path = feature_root / "manifest.json"
    cache = {
        "schema": "taxdpd.utonia-static-feature-cache/0.1",
        "inference_manifest_sha256": _sha256(inference_path),
        "feature_width": 576,
        "scale_factor": 15.0,
        "transform_scale": 0.5,
        "center_shift": False,
        "transform_seed": 1701,
        "records": [
            {
                "sample_id": sample_id,
                "split": "test",
                "geometry_sha256": geometry_sha,
                "feature_npz": feature_path.name,
                "feature_npz_sha256": _sha256(feature_path),
                "slot_count": 2048,
            }
        ],
    }
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return inference_path, cache_path, sample_id


def test_loads_cached_features_only_when_all_identity_links_match(tmp_path: Path) -> None:
    inference, cache, sample_id = _write_fixture(tmp_path)

    data = load_visualization_data(
        inference_manifest=inference,
        cache_manifest=cache,
        sample_id=sample_id,
    )

    assert data.features.shape == (2048, 576)
    assert data.child_points_world.shape == (1024, 3)
    assert data.parent_points_world.shape == (1024, 3)
    assert data.split == "test"
    assert data.center_shift is False


def test_rejects_feature_archive_hash_mismatch(tmp_path: Path) -> None:
    inference, cache, sample_id = _write_fixture(tmp_path)
    document = json.loads(cache.read_text(encoding="utf-8"))
    feature_path = cache.parent / document["records"][0]["feature_npz"]
    np.savez_compressed(feature_path, features=np.zeros((2048, 576), dtype=np.float32))

    with pytest.raises(ValueError, match="feature NPZ SHA-256"):
        load_visualization_data(
            inference_manifest=inference,
            cache_manifest=cache,
            sample_id=sample_id,
        )


def test_run_publishes_hash_addressed_html_and_refuses_replacement(tmp_path: Path) -> None:
    inference, cache, sample_id = _write_fixture(tmp_path)
    output_html = tmp_path / "utonia.html"
    summary_path = tmp_path / "utonia.summary.json"
    args = Namespace(
        inference_manifest=inference,
        cache_manifest=cache,
        sample_id=sample_id,
        output_html=output_html,
        summary=summary_path,
        show=False,
    )

    summary = run(args, html_renderer=lambda data, show: "<html><body>utonia</body></html>")

    assert summary["schema"] == SUMMARY_SCHEMA
    assert summary["model_rerun"] is False
    assert summary["quality_claim"] is False
    assert summary["utonia_config"]["feature_width"] == 576
    assert summary["output_html_sha256"] == _sha256(output_html)
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    with pytest.raises(FileExistsError, match="overwrite HTML"):
        run(args, html_renderer=lambda data, show: "<html></html>")
