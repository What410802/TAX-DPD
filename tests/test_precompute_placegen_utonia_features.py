"""Contracts for joining Utonia feature precompute to PlaceGen training exports."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from scripts.precompute_placegen_utonia_features import _load_training_sample_paths


def test_training_manifest_maps_long_sample_id_to_numeric_archive(tmp_path) -> None:
    export_root = tmp_path / "taxpose"
    training_root = export_root / "training"
    archive = training_root / "train" / "000000_final_obj_points.npz"
    archive.parent.mkdir(parents=True)
    np.savez(archive, value=np.asarray([1], dtype=np.int64))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = {
        "profile": "placegen.taxpose-rpdiff/0.1",
        "samples": [
                {
                    "sample_id": "rack-plate-final-blind-v1-000000",
                    "split": "train",
                    "training_npz": {
                    "path": "training/train/000000_final_obj_points.npz",
                    "sha256": digest,
                },
            }
        ],
    }
    manifest_path = export_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    paths = _load_training_sample_paths(manifest_path, training_root)

    assert paths == {"rack-plate-final-blind-v1-000000": archive.resolve()}
