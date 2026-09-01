"""Precompute target-free Utonia slot features for selected PlaceGen splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import torch

from non_rigid.models.encoders import (
    UtoniaSlotFeatureEncoder,
    utonia_geometry_sha256,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument(
        "--training-root",
        type=Path,
        help="optional native training root; if supplied, source action/anchor are read from it",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "validation"), default=("train", "validation")
    )
    parser.add_argument("--max-train", type=int, default=72)
    parser.add_argument("--max-validation", type=int, default=12)
    parser.add_argument("--scale-factor", type=float, default=15.0)
    parser.add_argument("--transform-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.inference_manifest.expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("profile") != "placegen.taxpose-inference-index/0.1":
        raise ValueError("unexpected PlaceGen inference profile")
    if manifest.get("ground_truth_free") is not True:
        raise ValueError("feature precompute requires a target-free inference manifest")
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite feature cache: {output_root}")
    output_root.mkdir(parents=True)
    device = torch.device(args.device)
    cfg = SimpleNamespace(
        utonia_input_scale_factor=float(args.scale_factor),
        utonia_transform_scale=float(args.transform_scale),
        utonia_center_shift=False,
        utonia_transform_seed=int(args.seed),
        utonia_checkpoint=str(args.checkpoint.expanduser().resolve(strict=True)),
        utonia_enable_flash=False,
    )
    encoder = UtoniaSlotFeatureEncoder(128, cfg).to(device).eval()
    limits = {"train": args.max_train, "validation": args.max_validation}
    selected_counts = {split: 0 for split in args.splits}
    records = []
    for sample in manifest.get("samples", []):
        split = sample.get("split")
        if split not in selected_counts or selected_counts[split] >= limits[split]:
            continue
        relative = PurePosixPath(sample["input_npz"]["path"])
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("unsafe inference NPZ path")
        path = manifest_path.parent.joinpath(*relative.parts).resolve(strict=True)
        path.relative_to(manifest_path.parent)
        if sha256_file(path) != sample["input_npz"]["sha256"]:
            raise ValueError("inference NPZ SHA-256 mismatch")
        with np.load(path, allow_pickle=False) as archive:
            action = np.asarray(archive["child_points_world"], dtype=np.float32)
            anchor = np.asarray(archive["parent_points_world"], dtype=np.float32)
        if args.training_root is not None:
            numeric_id = str(sample["sample_id"]).removeprefix("rack-plate-")
            if len(numeric_id) != 6 or not numeric_id.isdigit():
                raise ValueError("sample_id does not map to a canonical training artifact")
            train_path = (
                args.training_root.expanduser().resolve(strict=True)
                / split
                # PlaceGenTaxDpdDataset indexes *_final_obj_points.npz.  The
                # init artifact has a similar schema but is a different
                # snapshot and would produce a valid yet unjoinable cache.
                / f"{numeric_id}_final_obj_points.npz"
            )
            if train_path.is_symlink() or not train_path.is_file():
                raise FileNotFoundError(f"training source NPZ is missing: {train_path}")
            with np.load(train_path, allow_pickle=False) as archive:
                action = np.asarray(archive["action_points_source_world"], dtype=np.float32)
                anchor = np.asarray(archive["anchor_points_world"], dtype=np.float32)
        if action.shape != (1024, 3) or anchor.shape != (1024, 3):
            raise ValueError("expected 1024 action and anchor points")
        scale = np.float32(args.scale_factor)
        scene = np.concatenate((action * scale, anchor * scale), axis=0)
        scene -= scene.mean(axis=0, dtype=np.float32)
        scene_tensor = torch.from_numpy(scene).t().unsqueeze(0).to(device)
        geometry_sha = utonia_geometry_sha256(scene_tensor)
        with torch.inference_mode():
            features = encoder.frozen_features(scene_tensor)[0].t().cpu().numpy()
        features = np.ascontiguousarray(features, dtype=np.float32)
        if features.shape != (2048, 576) or not np.isfinite(features).all():
            raise RuntimeError("Utonia returned invalid static features")
        feature_name = f"{geometry_sha}.npz"
        feature_path = output_root / feature_name
        np.savez_compressed(feature_path, features=features)
        records.append(
            {
                "sample_id": sample["sample_id"],
                "split": split,
                "geometry_sha256": geometry_sha,
                "feature_npz": feature_name,
                "feature_npz_sha256": sha256_file(feature_path),
                "slot_count": 2048,
            }
        )
        selected_counts[split] += 1
        print(json.dumps({"sample_id": sample["sample_id"], "split": split}))
    if any(selected_counts[split] != limits[split] for split in selected_counts):
        raise RuntimeError(f"feature cache did not reach requested limits: {selected_counts}")
    cache_manifest = {
        "schema": "taxdpd.utonia-static-feature-cache/0.1",
        "inference_manifest_sha256": sha256_file(manifest_path),
        "utonia_checkpoint_sha256": sha256_file(args.checkpoint),
        "feature_width": 576,
        "scale_factor": float(args.scale_factor),
        "transform_scale": float(args.transform_scale),
        "center_shift": False,
        "transform_seed": int(args.seed),
        "split_counts": selected_counts,
        "test_npz_read": False,
        "records": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_root": str(output_root), "split_counts": selected_counts}))


if __name__ == "__main__":
    main()
