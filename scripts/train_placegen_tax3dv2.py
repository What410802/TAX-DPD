"""Train a PlaceGen-backed TAX3Dv2 generator without the unavailable RPDiff tree.

The first supported mode is ``fm`` and uses the controlled Euclidean CondOT
module.  This runner intentionally avoids Lightning/W&B and publishes a small,
auditable checkpoint plus a manifest.  It is a development/pilot runner, not a
claim that the incomplete public TAX-DPD release reproduces the paper pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from non_rigid.datasets.rigid import PlaceGenTaxDpdDataset
from non_rigid.models.flow_matching import flow_matching_loss
from non_rigid.utils.script_utils import create_model
from non_rigid.utils.placegen_fm_checkpoint import (
    FM_CHECKPOINT_SCHEMA,
    FM_MODEL_CAPABILITY,
)
from non_rigid.utils.state_digest import state_dict_sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_placegen_manifests(
    data_root: Path,
    *,
    training_manifest_path: Path | None,
    inference_manifest_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Validate the native PlaceGen export identity without reading targets.

    The current PlaceGen exporter predates TAX-DPD's grouped manifest schema,
    so this runner records and checks the native ``taxpose-rpdiff`` manifests
    directly.  Training reads supervision NPZs later through the dataset; the
    inference manifest is only an identity/provenance fence.
    """

    training_path = (
        training_manifest_path
        or data_root.parent / "manifest.json"
    ).expanduser().resolve(strict=True)
    inference_path = (
        inference_manifest_path
        or data_root.parent / "inference" / "manifest.json"
    ).expanduser().resolve(strict=True)
    training = json.loads(training_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    if training.get("profile") != "placegen.taxpose-rpdiff/0.1":
        raise ValueError("unexpected native PlaceGen training manifest profile")
    if inference.get("profile") != "placegen.taxpose-inference-index/0.1":
        raise ValueError("unexpected native PlaceGen inference manifest profile")
    if training.get("evaluation_valid") is not True or training.get("correlated_split") is not False:
        raise ValueError("training manifest is not evaluation-valid and disjoint")
    if inference.get("evaluation_valid") is not True or inference.get("correlated_split") is not False:
        raise ValueError("inference manifest is not evaluation-valid and disjoint")
    if inference.get("ground_truth_free") is not True:
        raise ValueError("inference manifest must be ground-truth-free")
    if training.get("point_counts") != {"action": 1024, "anchor": 1024}:
        raise ValueError("unexpected native PlaceGen point counts")
    if inference.get("point_counts") != training.get("point_counts"):
        raise ValueError("training/inference point counts differ")
    split_counts = training.get("split_counts")
    if split_counts != {"train": 72, "validation": 12, "test": 12}:
        raise ValueError(f"unexpected native PlaceGen split counts: {split_counts!r}")
    if inference.get("sample_count") != training.get("sample_count"):
        raise ValueError("training/inference sample counts differ")
    training_assignment = training.get("split_assignment", {})
    inference_assignment = inference.get("split_assignment", {})
    if inference_assignment.get("manifest_sha256") != training_assignment.get("manifest_sha256"):
        raise ValueError("native split assignment manifest identity differs")
    if inference_assignment.get("native_dataset_sha256") != training_assignment.get("native_dataset_sha256"):
        raise ValueError("native dataset identities differ")
    return (
        training,
        inference,
        _sha256_file(training_path),
        _sha256_file(inference_path),
    )


def _batch(dataset: PlaceGenTaxDpdDataset, index: int, device: torch.device) -> dict[str, Any]:
    item = dataset[index]
    return {
        key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
        for key, value in item.items()
    }


def _loss_for_item(module: Any, item: dict[str, Any], *, seed: int) -> torch.Tensor:
    target = module._get_x_start(item)
    target_frame, target_shape = module._split_target(target)
    endpoint = torch.cat((target_frame, target_shape), dim=2)
    source_frame, source_shape = module._source(target)
    source = torch.cat((source_frame, source_shape), dim=2)
    generator = torch.Generator(device=endpoint.device)
    generator.manual_seed(seed)
    # Keep time sampling deterministic per item/epoch while preserving a random
    # source path; source is generated by the module's configured prior.
    time = torch.rand((endpoint.shape[0],), device=endpoint.device, generator=generator)
    losses, _, _ = flow_matching_loss(
        module._velocity_model(module._model_kwargs(item)),
        endpoint,
        source=source,
        time=time,
    )
    return losses.mean()


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode != "fm":
        raise ValueError("only --mode fm is implemented; DDPM remains the comparison baseline")
    if args.epochs <= 0 or args.max_train <= 0 or args.max_validation <= 0:
        raise ValueError("epochs and sample limits must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_cfg = OmegaConf.load(Path(__file__).parents[1] / "configs/model/tax3dv2_fm.yaml")
    # The prior is part of the FM checkpoint contract.  Persist the CLI value
    # in the resolved config rather than relying on a runtime-only override.
    model_cfg.fm_noise_scale = float(args.noise_scale)
    training_cfg = OmegaConf.load(
        Path(__file__).parents[1] / "configs/training/placegen_taxdpd_tax3dv2_fm.yaml"
    )
    dataset_cfg = OmegaConf.load(Path(__file__).parents[1] / "configs/dataset/placegen_taxdpd.yaml")
    dataset_cfg.data_dir = str(Path(args.data_root).expanduser().resolve())
    dataset_cfg.train_dataset_size = None
    dataset_cfg.val_dataset_size = None
    dataset_cfg.test_dataset_size = None
    cfg = OmegaConf.create({"model": model_cfg, "training": training_cfg, "dataset": dataset_cfg})
    _, module = create_model(cfg)
    module = module.to(device)
    module.noise_scale = float(args.noise_scale)
    train_dataset = PlaceGenTaxDpdDataset(dataset_cfg.data_dir, dataset_cfg, "train")
    validation_dataset = PlaceGenTaxDpdDataset(dataset_cfg.data_dir, dataset_cfg, "val")
    train_count = min(len(train_dataset), args.max_train)
    validation_count = min(len(validation_dataset), args.max_validation)
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("PlaceGen train/validation splits must both be non-empty")

    training_manifest, inference_manifest, training_manifest_sha, inference_manifest_sha = _load_placegen_manifests(
        Path(args.data_root),
        training_manifest_path=args.training_manifest,
        inference_manifest_path=args.inference_manifest,
    )
    if training_manifest["point_counts"] != {
        "action": int(dataset_cfg.sample_size_action),
        "anchor": int(dataset_cfg.sample_size_anchor),
    }:
        raise ValueError("dataset point counts do not match the training manifest")
    if training_manifest["split_assignment"]["manifest_sha256"] != inference_manifest["split_assignment"]["manifest_sha256"]:
        raise ValueError("training and inference manifests use different split assignments")

    optimizer = torch.optim.Adam(module.parameters(), lr=args.learning_rate)
    best_validation = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        module.train()
        train_values: list[float] = []
        order = list(range(train_count))
        random.Random(args.seed + epoch).shuffle(order)
        for offset, index in enumerate(order):
            item = _batch(train_dataset, index, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_item(module, item, seed=args.seed + epoch * 100000 + offset)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite train loss at epoch={epoch}, index={index}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), args.grad_clip_norm)
            optimizer.step()
            train_values.append(float(loss.detach().cpu()))
        module.eval()
        validation_values: list[float] = []
        with torch.no_grad():
            for offset, index in enumerate(range(validation_count)):
                item = _batch(validation_dataset, index, device)
                value = _loss_for_item(module, item, seed=args.seed + 900000 + epoch * 1000 + offset)
                validation_values.append(float(value.detach().cpu()))
        train_mean = float(np.mean(train_values))
        validation_mean = float(np.mean(validation_values))
        history.append({"epoch": epoch, "train_loss": train_mean, "validation_loss": validation_mean})
        print(json.dumps(history[-1], sort_keys=True))
        if validation_mean < best_validation:
            best_validation = validation_mean
            best_state = {
                key: value.detach().cpu().clone() for key, value in module.network.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint was produced")
    checkpoint = Path(args.output_checkpoint).expanduser().resolve()
    manifest_path = Path(args.output_manifest or checkpoint.with_suffix(".json")).expanduser().resolve()
    if checkpoint.exists() or manifest_path.exists():
        raise FileExistsError("checkpoint or manifest output already exists; choose a new run directory")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state_sha256 = state_dict_sha256(best_state)
    checkpoint_payload = {
        "schema": FM_CHECKPOINT_SCHEMA,
        "model": FM_MODEL_CAPABILITY,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset": {
            "training_manifest_sha256": training_manifest_sha,
            "inference_manifest_sha256": inference_manifest_sha,
            "split_assignment_sha256": training_manifest["split_assignment"]["manifest_sha256"],
            "native_dataset_sha256": training_manifest["split_assignment"]["native_dataset_sha256"],
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "point_counts": dict(training_manifest["point_counts"]),
            "split_counts": dict(training_manifest["split_counts"]),
        },
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "train_count": train_count,
            "validation_count": validation_count,
            "best_validation_loss": best_validation,
            "history": history,
            "selection_split": "validation",
            "test_split_used_for_selection": False,
        },
        "model_state_dict": best_state,
        "model_state_sha256": state_sha256,
    }
    torch.save(checkpoint_payload, checkpoint)
    checkpoint_sha256 = _sha256_file(checkpoint)
    manifest = {
        "profile": "placegen.tax3dv2-training-run/0.1",
        "mode": args.mode,
        "model": "TAX3Dv2-fixed-frame-euclidean-condot-fm",
        "checkpoint_schema": FM_CHECKPOINT_SCHEMA,
        "model_capability": FM_MODEL_CAPABILITY,
        "architecture": {
            "point_encoder": "pn2",
            "joint_encode": True,
            "learn_sigma": False,
            "rotation_corruption": False,
            "fm_solver": model_cfg.fm_solver,
            "fm_steps": int(model_cfg.fm_steps),
        },
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "train_count": train_count,
        "validation_count": validation_count,
        "noise_scale": module.noise_scale,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "best_validation_loss": best_validation,
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": state_sha256,
        "training_manifest": str(
            (args.training_manifest or (Path(args.data_root).parent / "manifest.json")).expanduser().resolve()
        ),
        "training_manifest_sha256": training_manifest_sha,
        "inference_manifest": str(
            (args.inference_manifest or (Path(args.data_root).parent / "inference" / "manifest.json")).expanduser().resolve()
        ),
        "inference_manifest_sha256": inference_manifest_sha,
        "history": history,
        "planner_called": False,
        "simulator_called": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), "manifest": str(manifest_path)}, sort_keys=True))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--inference-manifest", type=Path)
    parser.add_argument("--mode", choices=("fm",), default="fm")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-train", type=int, default=72)
    parser.add_argument("--max-validation", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    # With the explicit TAX-DPD 15x coordinate frame, 0.08 is an empirically
    # calibrated development prior (roughly 5.3 mm in PlaceGen metres).  The
    # value is recorded in the run manifest and must not be tuned on final test.
    parser.add_argument("--noise-scale", type=float, default=0.08)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device")
    args = parser.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
