"""Overfit one real PlaceGen export as a bounded TAX-DPD feedback gate."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

REPORT_SCHEMA = "placegen.taxdpd-one-sample-overfit-report/0.1"
DEFAULT_STEPS = 200
LOSS_CYCLE_LENGTH = 100
LOSS_RATIO_LIMIT = 0.20
SAMPLED_RMSE_LIMIT_M = 0.02
SAMPLED_RMSE_REDUCTION_MIN = 0.80
TRANSLATION_LIMIT_MM = 20.0
ROTATION_LIMIT_DEG = 10.0
SEVERE_DISTANCE_M = 0.5
PROBE_TIMESTEPS = (0, 50, 99)
PROBE_SEED = 424242
BOUNDARY_TOLERANCE = 1.0e-3


def evaluate_overfit_gate(metrics: Mapping[str, Any], *, steps: int) -> dict[str, Any]:
    """Apply the pre-registered one-sample gate without importing ML packages."""

    if steps < 2 * LOSS_CYCLE_LENGTH or steps % LOSS_CYCLE_LENGTH != 0:
        raise ValueError("steps must be at least 200 and a multiple of 100")
    required = {
        "finite",
        "parameter_changed",
        "first_cycle_loss_median",
        "second_cycle_loss_median",
        "noop_ordered_rmse_m",
        "sampled_ordered_rmse_m",
        "world_translation_error_mm",
        "world_rotation_error_deg",
    }
    missing = sorted(required.difference(metrics))
    if missing:
        raise ValueError(f"gate metrics are missing fields: {missing}")
    numeric_names = sorted(required.difference({"finite", "parameter_changed"}))
    numeric = {name: float(metrics[name]) for name in numeric_names}
    values_finite = all(math.isfinite(value) for value in numeric.values())
    early = numeric["first_cycle_loss_median"]
    late = numeric["second_cycle_loss_median"]
    noop_rmse = numeric["noop_ordered_rmse_m"]
    sampled_rmse = numeric["sampled_ordered_rmse_m"]
    loss_ratio = late / early if early > 0.0 else math.inf
    sampled_reduction = 1.0 - sampled_rmse / noop_rmse if noop_rmse > 0.0 else -math.inf
    checks = {
        "finite": bool(metrics["finite"]) and values_finite,
        "parameter_changed": bool(metrics["parameter_changed"]),
        "second_vs_first_cycle_loss_ratio": loss_ratio <= LOSS_RATIO_LIMIT,
        "sampled_ordered_rmse_m": sampled_rmse <= SAMPLED_RMSE_LIMIT_M,
        "sampled_rmse_reduction": sampled_reduction >= SAMPLED_RMSE_REDUCTION_MIN,
        "world_translation_error_mm": (
            numeric["world_translation_error_mm"] <= TRANSLATION_LIMIT_MM
        ),
        "world_rotation_error_deg": (
            numeric["world_rotation_error_deg"] <= ROTATION_LIMIT_DEG
        ),
    }
    severe_red_flags = {
        "sampled_ordered_rmse_at_least_0_5m": sampled_rmse >= SEVERE_DISTANCE_M,
        "world_translation_error_at_least_0_5m": (
            numeric["world_translation_error_mm"] >= SEVERE_DISTANCE_M * 1000.0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "steps": steps,
            "matched_loss_cycle_length": LOSS_CYCLE_LENGTH,
            "second_vs_first_cycle_loss_ratio_max": LOSS_RATIO_LIMIT,
            "sampled_ordered_rmse_max_m": SAMPLED_RMSE_LIMIT_M,
            "sampled_rmse_reduction_min": SAMPLED_RMSE_REDUCTION_MIN,
            "world_translation_error_max_mm": TRANSLATION_LIMIT_MM,
            "world_rotation_error_max_deg": ROTATION_LIMIT_DEG,
            "severe_distance_m": SEVERE_DISTANCE_M,
        },
        "derived": {
            "second_vs_first_cycle_loss_ratio": loss_ratio,
            "sampled_rmse_reduction": sampled_reduction,
        },
        "severe_red_flags": severe_red_flags,
        "severe_red_flag": any(severe_red_flags.values()),
    }


def _ordered_rmse(source: np.ndarray, target: np.ndarray) -> float:
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if source_array.shape != target_array.shape or source_array.ndim != 2:
        raise ValueError("ordered point clouds must have the same [N, 3] shape")
    return float(np.sqrt(np.mean(np.sum((source_array - target_array) ** 2, axis=1))))


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = (
        np.asarray(first, dtype=np.float64) @ np.asarray(second, dtype=np.float64).T
    )
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _translation_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(value, dtype=np.float64)
    return matrix


def _array_range(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "maximum_absolute": float(np.max(np.abs(array))),
    }


def _boundary_counts(value: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(np.asarray(value, dtype=np.float64))
    total = int(absolute.size)
    result: dict[str, Any] = {
        "absolute_tolerance": BOUNDARY_TOLERANCE,
        "coordinate_count": total,
    }
    for boundary in (1.0, 2.0):
        count = int(np.count_nonzero(np.abs(absolute - boundary) <= BOUNDARY_TOLERANCE))
        result[f"near_abs_{int(boundary)}_count"] = count
        result[f"near_abs_{int(boundary)}_fraction"] = count / total
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import omegaconf
    import torch

    from non_rigid.datasets.rigid import RPDiffDataset
    from non_rigid.models.tax3d_v2 import TAX3Dv2FixedFrameModule, TAX3Dv2Network
    from non_rigid.utils.placegen_artifact import (
        load_placegen_observation,
        model_input_sha256,
        sha256_array,
        sha256_file,
    )
    from non_rigid.utils.rigid_transform import estimate_ordered_rigid_transform
    from non_rigid.utils.state_digest import state_dict_sha256

    if args.steps < 2 * LOSS_CYCLE_LENGTH or args.steps % LOSS_CYCLE_LENGTH != 0:
        raise ValueError("--steps must be at least 200 and a multiple of 100")
    if args.seed < 0 or args.sample_seed < 0:
        raise ValueError("seeds must be non-negative")
    supplied_report = Path(args.report).expanduser()
    if supplied_report.exists() or supplied_report.is_symlink():
        raise FileExistsError(f"refusing to overwrite report: {supplied_report}")
    report_path = supplied_report.resolve()
    repo_root = Path(args.repo_root).expanduser().resolve(strict=True)
    data_root = Path(args.data_root).expanduser().resolve(strict=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device but CUDA is unavailable")

    artifact = load_placegen_observation(args.input_npz, args.manifest)
    model_cfg = omegaconf.OmegaConf.load(repo_root / "configs/model/tax3dv2.yaml")
    dataset_cfg = omegaconf.OmegaConf.load(repo_root / "configs/dataset/rpdiff.yaml")
    training_cfg = omegaconf.OmegaConf.load(
        repo_root / "configs/training/rpdiff_tax3dv2.yaml"
    )
    dataset_cfg.data_dir = str(data_root)
    dataset_cfg.rpdiff_task_name = args.task_name
    dataset_cfg.rpdiff_task_type = args.task_type
    dataset_cfg.preprocess = False
    dataset_cfg.augment_gt = False
    dataset_cfg.num_demos = 1
    dataset_cfg.train_dataset_size = 1
    dataset_cfg.val_dataset_size = 1
    dataset_cfg.test_dataset_size = 1
    dataset_cfg.sample_size_action = len(artifact.pc_action_centered)
    dataset_cfg.sample_size_anchor = len(artifact.pc_anchor_centered)
    dataset_cfg.downsample_type = "fps"
    dataset_cfg.scene_transform_type = "identity"
    dataset_cfg.translation_variance = 0.0
    dataset_cfg.rotation_variance = 0.0
    dataset_cfg.action_plane_occlusion = 0.0
    dataset_cfg.action_ball_occlusion = 0.0
    dataset_cfg.anchor_plane_occlusion = 0.0
    dataset_cfg.anchor_ball_occlusion = 0.0
    training_cfg.batch_size = 1
    training_cfg.val_batch_size = 1
    training_cfg.num_training_steps = args.steps
    training_cfg.sample_size = len(artifact.pc_action_centered)
    training_cfg.sample_size_anchor = len(artifact.pc_anchor_centered)
    training_cfg.additional_train_logging_period = 10**9
    training_cfg.prediction_error_type = "demo"
    cfg = omegaconf.OmegaConf.create(
        {
            "mode": "train",
            "seed": args.seed,
            "model": model_cfg,
            "dataset": dataset_cfg,
            "training": training_cfg,
            "resources": {"num_workers": 0, "gpus": [0]},
        }
    )

    training_npz = (
        data_root / args.task_name / args.task_type / f"{artifact.sample_id}.npz"
    )
    if training_npz.is_symlink() or not training_npz.is_file():
        raise FileNotFoundError(
            f"training NPZ must be a regular file at the expected path: {training_npz}"
        )
    manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_training = manifest_payload.get("files", {}).get("training_npz")
    if not isinstance(declared_training, dict):
        raise TypeError("manifest is missing files.training_npz provenance")
    declared_path = declared_training.get("path")
    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError("manifest training NPZ path must be a non-empty string")
    manifest_root = manifest_path.parent
    expected_training = (manifest_root / declared_path).resolve()
    try:
        expected_training.relative_to(manifest_root)
    except ValueError as error:
        raise ValueError("manifest training NPZ path escapes its directory") from error
    if expected_training != training_npz:
        raise ValueError("training NPZ does not match the manifest-declared path")
    if declared_training.get("sha256") != sha256_file(training_npz):
        raise ValueError("training NPZ SHA-256 does not match the export manifest")

    dataset = RPDiffDataset(data_root, cfg.dataset, "train")
    if (
        len(dataset.demo_files) != 1
        or Path(dataset.demo_files[0]).resolve() != training_npz
    ):
        raise RuntimeError("RPDiff loader did not select exactly the exported sample")
    batch = dataset[0]
    training_action = batch["pc_action"].detach().cpu().numpy()
    training_anchor = batch["pc_anchor"].detach().cpu().numpy()
    training_goal = batch["pc"].detach().cpu().numpy()
    observed_hashes = {
        "pc_action_centered": sha256_array(training_action),
        "pc_anchor_centered": sha256_array(training_anchor),
    }
    if observed_hashes != artifact.tensor_sha256:
        raise RuntimeError("RPDiff loader tensors differ from observation tensors")
    if (
        model_input_sha256(training_action, training_anchor)
        != artifact.model_input_sha256
    ):
        raise RuntimeError("RPDiff loader model input differs from observation input")
    goal_hash = sha256_array(training_goal)
    if artifact.supervision_tensor_sha256 != goal_hash:
        raise RuntimeError(
            "RPDiff loader goal tensor differs from manifest supervision"
        )

    tensor_batch = {
        name: value.unsqueeze(0).to(device)
        if isinstance(value, torch.Tensor)
        else value
        for name, value in batch.items()
    }
    module = TAX3Dv2FixedFrameModule(TAX3Dv2Network(cfg.model), cfg).to(device)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    initial_model_sha = state_dict_sha256(module.network.state_dict())

    def fixed_loss_probes() -> dict[str, float]:
        was_training = module.training
        module.eval()
        cuda_devices: list[int] = []
        if device.type == "cuda":
            cuda_devices = [
                torch.cuda.current_device()
                if device.index is None
                else int(device.index)
            ]
        values: dict[str, float] = {}
        try:
            with torch.inference_mode():
                for timestep_value in PROBE_TIMESTEPS:
                    with torch.random.fork_rng(devices=cuda_devices):
                        torch.random.default_generator.manual_seed(PROBE_SEED)
                        if device.type == "cuda":
                            torch.cuda.default_generators[cuda_devices[0]].manual_seed(
                                PROBE_SEED
                            )
                        probe_timestep = torch.tensor(
                            [timestep_value], dtype=torch.long, device=device
                        )
                        probe_losses = module.diffusion.training_losses(
                            module.network,
                            module._get_x_start(tensor_batch),
                            probe_timestep,
                            module._model_kwargs(tensor_batch),
                        )
                        value = float(probe_losses["loss"].mean().detach().cpu())
                        if not math.isfinite(value):
                            raise RuntimeError(
                                "non-finite fixed loss probe at timestep "
                                f"{timestep_value}"
                            )
                        values[str(timestep_value)] = value
        finally:
            module.train(was_training)
        return values

    module.train()
    target_x_start = module._get_x_start(tensor_batch)
    target_reference = target_x_start.mean(dim=-1, keepdim=True)
    target_shape_residual = target_x_start - target_reference
    target_decomposition = {
        "x_start": _array_range(target_x_start.detach().cpu().numpy()),
        "reference_mean": _array_range(target_reference.detach().cpu().numpy()),
        "shape_residual": _array_range(target_shape_residual.detach().cpu().numpy()),
    }
    probe_loss_before = fixed_loss_probes()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    training_started = time.perf_counter()
    losses: list[float] = []
    gradient_norms: list[float] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        x_start = module._get_x_start(tensor_batch)
        timestep = torch.tensor(
            [step % module.diff_train_steps], dtype=torch.long, device=device
        )
        loss_values = module.diffusion.training_losses(
            module.network,
            x_start,
            timestep,
            module._model_kwargs(tensor_batch),
        )
        loss = loss_values["loss"].mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at optimizer step {step + 1}")
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                module.parameters(), float(cfg.training.grad_clip_norm)
            )
        )
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient norm at optimizer step {step + 1}")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(gradient_norm)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_duration = time.perf_counter() - training_started
    final_model_sha = state_dict_sha256(module.network.state_dict())
    probe_loss_after = fixed_loss_probes()
    probe_loss_ratio = {
        timestep: probe_loss_after[timestep] / probe_loss_before[timestep]
        if probe_loss_before[timestep] > 0.0
        else math.inf
        for timestep in probe_loss_before
    }

    sampling_started = time.perf_counter()
    module.eval()
    with torch.inference_mode():
        sampled = module.sample_candidates(
            tensor_batch["pc_action"],
            tensor_batch["pc_anchor"],
            num_trials=1,
            seed=args.sample_seed,
        )[0, 0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    sampling_duration = time.perf_counter() - sampling_started
    sampled_numpy = sampled.detach().cpu().numpy().astype(np.float64)
    action_numpy = training_action.astype(np.float64, copy=False)
    goal_numpy = training_goal.astype(np.float64, copy=False)
    noop_rmse = _ordered_rmse(action_numpy, goal_numpy)
    sampled_rmse = _ordered_rmse(sampled_numpy, goal_numpy)
    predicted_fit = estimate_ordered_rigid_transform(action_numpy, sampled_numpy)
    target_fit = estimate_ordered_rigid_transform(action_numpy, goal_numpy)
    center_to_world = _translation_matrix(artifact.scene_center_world)
    world_to_center = _translation_matrix(-artifact.scene_center_world)
    predicted_relative_world = center_to_world @ predicted_fit.matrix @ world_to_center
    target_relative_world = center_to_world @ target_fit.matrix @ world_to_center
    predicted_world_pose = predicted_relative_world @ artifact.source_world_from_object
    target_world_pose = target_relative_world @ artifact.source_world_from_object
    translation_error_mm = float(
        np.linalg.norm(predicted_world_pose[:3, 3] - target_world_pose[:3, 3]) * 1000.0
    )
    rotation_error_deg = _rotation_error_deg(
        predicted_world_pose[:3, :3], target_world_pose[:3, :3]
    )
    value_range = {
        "minimum_m": float(np.min(sampled_numpy)),
        "maximum_m": float(np.max(sampled_numpy)),
        "maximum_absolute_m": float(np.max(np.abs(sampled_numpy))),
    }
    finite = bool(
        np.isfinite(sampled_numpy).all()
        and np.isfinite(losses).all()
        and np.isfinite(gradient_norms).all()
        and np.isfinite(predicted_world_pose).all()
        and np.isfinite(target_world_pose).all()
    )
    first_cycle_loss_median = float(np.median(losses[:LOSS_CYCLE_LENGTH]))
    second_cycle_loss_median = float(np.median(losses[-LOSS_CYCLE_LENGTH:]))
    metrics = {
        "finite": finite,
        "parameter_changed": initial_model_sha != final_model_sha,
        "first_cycle_loss_median": first_cycle_loss_median,
        "second_cycle_loss_median": second_cycle_loss_median,
        "noop_ordered_rmse_m": noop_rmse,
        "sampled_ordered_rmse_m": sampled_rmse,
        "world_translation_error_mm": translation_error_mm,
        "world_rotation_error_deg": rotation_error_deg,
    }
    gate = evaluate_overfit_gate(metrics, steps=args.steps)
    report = {
        "schema": REPORT_SCHEMA,
        "quality_claim": False,
        "diagnostic_scope": "single-export-overfit-not-generalization",
        "model": "TAX-DPD-reconstructed-fixed-frame-w/o-GMM",
        "sample_id": artifact.sample_id,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "input_npz": str(Path(args.input_npz).expanduser().resolve(strict=True)),
        "input_npz_sha256": artifact.input_npz_sha256,
        "training_npz": str(training_npz),
        "training_npz_sha256": sha256_file(training_npz),
        "model_input_sha256": artifact.model_input_sha256,
        "supervision_tensor_sha256": goal_hash,
        "steps": args.steps,
        "timestep_schedule": "step-modulo-100",
        "seed": args.seed,
        "sample_seed": args.sample_seed,
        "candidate_index": 0,
        "candidate_count": 1,
        "sampling_only_after_final_step": True,
        "target_decomposition": target_decomposition,
        "fixed_loss_probes": {
            "timesteps": list(PROBE_TIMESTEPS),
            "seed": PROBE_SEED,
            "module_mode_before": "eval",
            "module_mode_after": "eval",
            "loss_before": probe_loss_before,
            "loss_after": probe_loss_after,
            "after_to_before_ratio": probe_loss_ratio,
            "fork_rng_does_not_advance_training_rng": True,
        },
        "loss": {
            "first_matched_timestep_cycle": losses[:LOSS_CYCLE_LENGTH],
            "second_matched_timestep_cycle": losses[-LOSS_CYCLE_LENGTH:],
            "first_cycle_median": first_cycle_loss_median,
            "second_cycle_median": second_cycle_loss_median,
            "minimum": min(losses),
            "maximum": max(losses),
        },
        "gradient_norm": {
            "mean": float(np.mean(gradient_norms)),
            "maximum": max(gradient_norms),
        },
        "noop_ordered_rmse_m": noop_rmse,
        "final_sampled_ordered_rmse_m": sampled_rmse,
        "predicted_kabsch_fit_rmse_m": predicted_fit.rmse,
        "target_kabsch_fit_rmse_m": target_fit.rmse,
        "world_translation_error_mm": translation_error_mm,
        "world_rotation_error_deg": rotation_error_deg,
        "predicted_world_from_object": predicted_world_pose.tolist(),
        "target_world_from_object": target_world_pose.tolist(),
        "sampled_point_range": value_range,
        "sampled_coordinate_boundaries": _boundary_counts(sampled_numpy),
        "finite": finite,
        "initial_model_state_sha256": initial_model_sha,
        "final_model_state_sha256": final_model_sha,
        "parameter_changed": initial_model_sha != final_model_sha,
        "gate": gate,
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "peak_vram_mib": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda"
            else None
        ),
        "training_duration_seconds": training_duration,
        "sampling_duration_seconds": sampling_duration,
        "total_duration_seconds": time.perf_counter() - started,
    }
    _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-name", default="placegen_plate_smoke")
    parser.add_argument("--task-type", default="task_name_plate_on_rack")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample-seed", type=int, default=1701)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, sort_keys=True))
    if args.require_pass and not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
