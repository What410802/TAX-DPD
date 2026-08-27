"""Train reconstructed fixed-frame TAX-DPD on PlaceGen's held-out pose split."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import omegaconf
import torch
from torch.utils.data import DataLoader

from non_rigid.datasets.rigid import RPDiffDataset
from non_rigid.models.tax3d_v2 import TAX3Dv2FixedFrameModule, TAX3Dv2Network
from non_rigid.utils.placegen_checkpoint import (
    GROUPED_CHECKPOINT_SCHEMA,
    load_grouped_checkpoint,
    optimizer_state_sha256,
)
from non_rigid.utils.placegen_grouped import (
    MODEL_CAPABILITY,
    QUALITY_SCOPE,
    GroupedTrainingManifest,
    ValidationCheckpointDecision,
    load_grouped_training_manifest,
    select_validation_checkpoint,
    sha256_file,
)
from non_rigid.utils.state_digest import clone_state_dict, state_dict_sha256

REPORT_SCHEMA = "placegen.taxdpd-grouped-training-report/0.1"


def _config(
    repo_root: Path,
    dataset: GroupedTrainingManifest,
    *,
    batch_size: int,
    validation_batch_size: int,
    epochs: int,
) -> omegaconf.DictConfig:
    model = omegaconf.OmegaConf.load(repo_root / "configs/model/tax3dv2.yaml")
    dataset_cfg = omegaconf.OmegaConf.load(repo_root / "configs/dataset/rpdiff.yaml")
    training = omegaconf.OmegaConf.load(
        repo_root / "configs/training/rpdiff_tax3dv2.yaml"
    )
    dataset_cfg.data_dir = str(dataset.root)
    dataset_cfg.rpdiff_task_name = dataset.task_name
    dataset_cfg.rpdiff_task_type = dataset.task_type
    dataset_cfg.preprocess = False
    dataset_cfg.augment_gt = False
    dataset_cfg.num_demos = None
    dataset_cfg.train_dataset_size = None
    dataset_cfg.val_dataset_size = None
    dataset_cfg.test_dataset_size = None
    dataset_cfg.sample_size_action = dataset.point_counts["action"]
    dataset_cfg.sample_size_anchor = dataset.point_counts["anchor"]
    dataset_cfg.downsample_type = "fps"
    dataset_cfg.val_use_defaults = False
    dataset_cfg.scene_transform_type = "identity"
    dataset_cfg.translation_variance = 0.0
    dataset_cfg.rotation_variance = 0.0
    dataset_cfg.action_plane_occlusion = 0.0
    dataset_cfg.action_ball_occlusion = 0.0
    dataset_cfg.anchor_plane_occlusion = 0.0
    dataset_cfg.anchor_ball_occlusion = 0.0
    training.epochs = epochs
    training.batch_size = batch_size
    training.val_batch_size = validation_batch_size
    training.num_training_steps = (
        math.ceil(dataset.split_counts["train"] / batch_size) * epochs
    )
    training.sample_size = dataset.point_counts["action"]
    training.sample_size_anchor = dataset.point_counts["anchor"]
    training.additional_train_logging_period = 10 ** 9
    training.prediction_error_type = "demo"
    return omegaconf.OmegaConf.create(
        {
            "mode": "train",
            "seed": 0,
            "model": model,
            "dataset": dataset_cfg,
            "training": training,
            "resources": {"num_workers": 0, "gpus": [0]},
        }
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def _dataset_paths(dataset: RPDiffDataset) -> tuple[Path, ...]:
    return tuple(Path(value).resolve(strict=True) for value in dataset.demo_files)


def _validate_loader_partition(
    dataset: RPDiffDataset,
    expected_paths: tuple[Path, ...],
    split: str,
) -> None:
    actual = _dataset_paths(dataset)
    if actual != expected_paths:
        raise RuntimeError(
            f"RPDiff {split} loader files differ from the grouped manifest"
        )
    if len(dataset) != len(expected_paths):
        raise RuntimeError(f"RPDiff {split} loader length differs from its manifest")


def _validate_candidate0(
    module: TAX3Dv2FixedFrameModule,
    loader: DataLoader,
    device: torch.device,
    *,
    validation_seed: int,
    expected_sample_ids: tuple[str, ...],
) -> tuple[float, tuple[float, ...]]:
    module.eval()
    sample_metrics: list[float] = []
    seen = 0
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            batch = _move_batch(raw_batch, device)
            predicted = module.sample_candidates(
                batch["pc_action"],
                batch["pc_anchor"],
                num_trials=1,
                seed=validation_seed + batch_index,
            )[:, 0]
            goal = batch["pc"]
            if predicted.shape != goal.shape:
                raise RuntimeError(
                    "validation candidate zero shape differs from ground truth"
                )
            rmse = torch.sqrt(
                torch.mean(torch.sum((predicted - goal) ** 2, dim=-1), dim=-1)
            )
            if not torch.isfinite(rmse).all():
                raise RuntimeError(
                    "validation candidate-zero ordered RMSE is non-finite"
                )
            values = [float(value) for value in rmse.detach().cpu()]
            sample_metrics.extend(values)
            seen += len(values)
    if seen != len(expected_sample_ids):
        raise RuntimeError("validation evaluated an unexpected number of samples")
    metric = float(np.mean(np.asarray(sample_metrics, dtype=np.float64)))
    if not math.isfinite(metric):
        raise RuntimeError("mean validation candidate-zero RMSE is non-finite")
    return metric, tuple(sample_metrics)


def _checkpoint_payload(
    *,
    cfg: omegaconf.DictConfig,
    dataset: GroupedTrainingManifest,
    module: TAX3Dv2FixedFrameModule,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    seed: int,
    validation_seed: int,
    train_loss_mean: float,
    gradient_norm_mean: float,
    validation_metric: float,
) -> dict[str, Any]:
    model_state = clone_state_dict(module.network.state_dict())
    optimizer_state = optimizer.state_dict()
    return {
        "schema": GROUPED_CHECKPOINT_SCHEMA,
        "model": MODEL_CAPABILITY,
        "config": omegaconf.OmegaConf.to_container(cfg, resolve=True),
        "dataset": {
            "manifest_sha256": dataset.manifest_sha256,
            "split_assignment_sha256": dataset.split_assignment_sha256,
            "native_dataset_sha256": dataset.native_dataset_sha256,
            "task_name": dataset.task_name,
            "task_type": dataset.task_type,
            "point_counts": dict(dataset.point_counts),
            "sample_counts": dict(dataset.split_counts),
            "group_counts": dict(dataset.group_counts),
            "quality_scope": QUALITY_SCOPE,
        },
        "training": {
            "epoch": epoch,
            "global_step": global_step,
            "seed": seed,
            "validation_seed": validation_seed,
            "train_loss_mean": train_loss_mean,
            "gradient_norm_mean": gradient_norm_mean,
            "validation_candidate0_ordered_rmse_m": validation_metric,
            "selection_split": "validation",
            "selection_candidate_index": 0,
            "validation_sample_ids": list(dataset.validation_sample_ids),
            "test_split_used_for_selection": False,
        },
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "model_state_sha256": state_dict_sha256(model_state),
        "optimizer_state_sha256": optimizer_state_sha256(optimizer_state),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs < 2:
        raise ValueError("--epochs must be at least 2 for grouped training")
    if args.batch_size <= 0 or args.validation_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.seed < 0 or args.validation_seed < 0:
        raise ValueError("seeds must be non-negative")
    expected_group_counts = {
        "train": args.expected_train_groups,
        "validation": args.expected_validation_groups,
        "test": args.expected_test_groups,
    }
    if any(value <= 0 for value in expected_group_counts.values()):
        raise ValueError("expected physical group counts must be positive")
    repo_root = Path(args.repo_root).expanduser().resolve(strict=True)
    supplied_checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    if supplied_checkpoint_dir.exists() or supplied_checkpoint_dir.is_symlink():
        raise FileExistsError(
            f"refusing to reuse checkpoint directory: {supplied_checkpoint_dir}"
        )
    checkpoint_dir = supplied_checkpoint_dir.resolve()
    supplied_report = Path(args.report).expanduser()
    if supplied_report.exists() or supplied_report.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite training report: {supplied_report}"
        )
    report_path = supplied_report.resolve()
    dataset = load_grouped_training_manifest(
        args.manifest,
        expected_group_counts=expected_group_counts,
    )
    cfg = _config(
        repo_root,
        dataset,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        epochs=args.epochs,
    )
    cfg.seed = args.seed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA training but CUDA is unavailable")

    train_dataset = RPDiffDataset(dataset.root, cfg.dataset, "train")
    validation_dataset = RPDiffDataset(dataset.root, cfg.dataset, "val")
    expected_train_paths = tuple(
        sample.training_npz for sample in dataset.samples_by_split["train"]
    )
    expected_validation_paths = tuple(
        sample.training_npz for sample in dataset.samples_by_split["validation"]
    )
    _validate_loader_partition(train_dataset, expected_train_paths, "train")
    _validate_loader_partition(
        validation_dataset, expected_validation_paths, "validation"
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.validation_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    if len(train_loader) < 2:
        raise RuntimeError(
            "grouped training requires at least two optimizer steps per epoch"
        )

    module = TAX3Dv2FixedFrameModule(network=TAX3Dv2Network(cfg.model), cfg=cfg).to(
        device
    )
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    initial_model_sha = state_dict_sha256(module.network.state_dict())
    checkpoint_dir.mkdir(parents=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()
    global_step = 0
    history: list[dict[str, Any]] = []
    best: ValidationCheckpointDecision | None = None
    checkpoint_by_epoch: dict[int, Path] = {}

    for epoch in range(1, args.epochs + 1):
        module.train()
        epoch_started = time.perf_counter()
        losses: list[float] = []
        gradient_norms: list[float] = []
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            x_start = module._get_x_start(batch)
            timestep = torch.randint(
                0,
                module.diff_train_steps,
                (x_start.shape[0],),
                device=device,
            ).long()
            loss_values = module.diffusion.training_losses(
                module.network,
                x_start,
                timestep,
                module._model_kwargs(batch),
            )
            loss = loss_values["loss"].mean()
            if not torch.isfinite(loss):
                raise RuntimeError("grouped training produced a non-finite loss")
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    module.parameters(), float(cfg.training.grad_clip_norm)
                )
            )
            if not math.isfinite(gradient_norm):
                raise RuntimeError(
                    "grouped training produced a non-finite gradient norm"
                )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(gradient_norm)
            global_step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_duration = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        validation_metric, validation_sample_metrics = _validate_candidate0(
            module,
            validation_loader,
            device,
            validation_seed=args.validation_seed,
            expected_sample_ids=dataset.validation_sample_ids,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_duration = time.perf_counter() - validation_started
        train_loss_mean = float(np.mean(np.asarray(losses, dtype=np.float64)))
        gradient_norm_mean = float(
            np.mean(np.asarray(gradient_norms, dtype=np.float64))
        )
        selection = select_validation_checkpoint(
            previous=best,
            epoch=epoch,
            validation_candidate0_ordered_rmse_m=validation_metric,
            validation_sample_ids=dataset.validation_sample_ids,
            expected_validation_sample_ids=dataset.validation_sample_ids,
        )
        improved = best is None or selection.epoch != best.epoch
        best = selection
        checkpoint_path = checkpoint_dir / f"epoch-{epoch:04d}.pt"
        payload = _checkpoint_payload(
            cfg=cfg,
            dataset=dataset,
            module=module,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            seed=args.seed,
            validation_seed=args.validation_seed,
            train_loss_mean=train_loss_mean,
            gradient_norm_mean=gradient_norm_mean,
            validation_metric=validation_metric,
        )
        with checkpoint_path.open("xb") as stream:
            torch.save(payload, stream)
        checkpoint_by_epoch[epoch] = checkpoint_path
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": len(losses),
                "global_step": global_step,
                "train_loss_mean": train_loss_mean,
                "train_loss_min": min(losses),
                "train_loss_max": max(losses),
                "gradient_norm_mean": gradient_norm_mean,
                "gradient_norm_max": max(gradient_norms),
                "train_duration_seconds": train_duration,
                "validation_candidate0_ordered_rmse_m": validation_metric,
                "validation_sample_rmse_m": list(validation_sample_metrics),
                "validation_duration_seconds": validation_duration,
                "selected_as_best": improved,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )

    if best is None:
        raise RuntimeError("training completed without a validation selection")
    best_checkpoint_path = checkpoint_by_epoch[best.epoch]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_vram_mib = float(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
    else:
        peak_vram_mib = None
    # Reload on CPU so verification does not temporarily double the live CUDA
    # model footprint or distort the measured training peak.
    loaded = load_grouped_checkpoint(
        best_checkpoint_path,
        torch.device("cpu"),
        expected_dataset_manifest_sha256=dataset.manifest_sha256,
        expected_split_assignment_sha256=dataset.split_assignment_sha256,
    )
    reloaded_cfg = omegaconf.OmegaConf.create(loaded.payload["config"])
    reloaded_optimizer = torch.optim.AdamW(
        loaded.module.parameters(),
        lr=float(reloaded_cfg.training.lr),
        weight_decay=float(reloaded_cfg.training.weight_decay),
    )
    reloaded_optimizer.load_state_dict(loaded.payload["optimizer_state_dict"])
    if (
        optimizer_state_sha256(reloaded_optimizer.state_dict())
        != loaded.optimizer_state_sha256
    ):
        raise RuntimeError("reloaded optimizer state differs from the best checkpoint")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_duration = time.perf_counter() - training_started
    final_model_sha = state_dict_sha256(module.network.state_dict())
    if final_model_sha == initial_model_sha:
        raise RuntimeError("multi-epoch training did not change model parameters")
    report = {
        "schema": REPORT_SCHEMA,
        "quality_claim": False,
        "quality_assessment": "inconclusive-until-held-out-pose-evaluation",
        "model": MODEL_CAPABILITY,
        "dataset_manifest": str(dataset.manifest_path),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "split_assignment_sha256": dataset.split_assignment_sha256,
        "native_dataset_sha256": dataset.native_dataset_sha256,
        "quality_scope": QUALITY_SCOPE,
        "group_counts": dict(dataset.group_counts),
        "sample_counts": dict(dataset.split_counts),
        "epochs": args.epochs,
        "global_optimizer_steps": global_step,
        "multi_epoch_verified": args.epochs >= 2,
        "multi_step_verified": global_step >= 2,
        "initial_model_state_sha256": initial_model_sha,
        "final_model_state_sha256": final_model_sha,
        "parameter_changed": initial_model_sha != final_model_sha,
        "selection_split": "validation",
        "selection_candidate_index": 0,
        "test_split_used_for_checkpoint_selection": False,
        "test_model_batches_evaluated": 0,
        "best_epoch": best.epoch,
        "best_validation_candidate0_ordered_rmse_m": (
            best.validation_candidate0_ordered_rmse_m
        ),
        "best_checkpoint": str(best_checkpoint_path),
        "best_checkpoint_sha256": sha256_file(best_checkpoint_path),
        "best_checkpoint_model_state_sha256": loaded.model_state_sha256,
        "best_checkpoint_optimizer_state_sha256": loaded.optimizer_state_sha256,
        "checkpoint_reload_verified": True,
        "model_reload_verified": True,
        "optimizer_reload_verified": True,
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "peak_vram_mib": peak_vram_mib,
        "total_duration_seconds": total_duration,
        "seed": args.seed,
        "validation_seed": args.validation_seed,
        "history": history,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validation-seed", type=int, default=1701)
    parser.add_argument("--expected-train-groups", type=int, default=72)
    parser.add_argument("--expected-validation-groups", type=int, default=12)
    parser.add_argument("--expected-test-groups", type=int, default=12)
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
