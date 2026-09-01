"""Add native split/dataset identities to an early FM checkpoint envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from non_rigid.utils.placegen_fm_checkpoint import (
    FM_CHECKPOINT_SCHEMA,
    validate_fm_checkpoint_payload,
)
from non_rigid.utils.placegen_fm_export import load_native_export_identity
from non_rigid.utils.state_digest import state_dict_sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, str]:
    source = Path(args.input_checkpoint).expanduser().resolve(strict=True)
    output = Path(args.output_checkpoint).expanduser()
    report = Path(args.output_json).expanduser()
    if output.exists() or output.is_symlink() or report.exists() or report.is_symlink():
        raise FileExistsError("output checkpoint/report already exists")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != FM_CHECKPOINT_SCHEMA:
        raise ValueError("input is not an early PlaceGen FM checkpoint")
    dataset = payload.get("dataset")
    old_dataset_fields = {
        "training_manifest_sha256",
        "inference_manifest_sha256",
        "data_root",
        "point_counts",
        "split_counts",
    }
    if not isinstance(dataset, dict) or set(dataset) != old_dataset_fields:
        raise ValueError("input checkpoint is not the supported early envelope")
    identity = load_native_export_identity(
        args.training_manifest, args.inference_manifest
    )
    if dataset["training_manifest_sha256"] != identity.training_manifest_sha256:
        raise ValueError("checkpoint training manifest SHA-256 differs")
    if dataset["inference_manifest_sha256"] != identity.inference_manifest_sha256:
        raise ValueError("checkpoint inference manifest SHA-256 differs")
    if dataset["point_counts"] != identity.point_counts:
        raise ValueError("checkpoint point counts differ")
    if dataset["split_counts"] != identity.split_counts:
        raise ValueError("checkpoint split counts differ")
    actual_state_sha = state_dict_sha256(payload["model_state_dict"])
    if actual_state_sha != payload.get("model_state_sha256"):
        raise ValueError("checkpoint model state SHA-256 differs")
    upgraded = dict(payload)
    upgraded["dataset"] = {
        **dataset,
        "split_assignment_sha256": identity.split_assignment_sha256,
        "native_dataset_sha256": identity.native_dataset_sha256,
    }
    validate_fm_checkpoint_payload(upgraded)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(upgraded, output)
    result = {
        "profile": "placegen.tax3dv2-fm-checkpoint-upgrade/0.1",
        "input_checkpoint": str(source),
        "input_checkpoint_sha256": _sha256_file(source),
        "output_checkpoint": str(output),
        "output_checkpoint_sha256": _sha256_file(output),
        "model_state_sha256": actual_state_sha,
        "split_assignment_sha256": identity.split_assignment_sha256,
        "native_dataset_sha256": identity.native_dataset_sha256,
    }
    report = report.resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
