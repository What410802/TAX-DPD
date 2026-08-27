"""Run exactly one real TAX-DPD fixed-frame optimizer update on a PlaceGen export.

This is deliberately a small, W&B-free smoke runner.  It exercises the real
RPDiff loader, reconstructed TAX3Dv2 network, diffusion loss, backward pass,
optimizer update, and checkpoint serialization without claiming convergence or
full TAX-DPD reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from non_rigid.utils.placegen_artifact import (
    load_placegen_observation,
    model_input_sha256,
    sha256_array,
    sha256_file,
)
from non_rigid.utils.state_digest import clone_state_dict, state_dict_sha256


def _sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _config(
    root: Path, data_root: Path, task_name: str, task_type: str
) -> omegaconf.DictConfig:
    model = omegaconf.OmegaConf.load(root / "configs/model/tax3dv2.yaml")
    dataset = omegaconf.OmegaConf.load(root / "configs/dataset/rpdiff.yaml")
    training = omegaconf.OmegaConf.load(root / "configs/training/rpdiff_tax3dv2.yaml")
    dataset.data_dir = str(data_root)
    dataset.rpdiff_task_name = task_name
    dataset.rpdiff_task_type = task_type
    dataset.preprocess = False
    dataset.augment_gt = False
    dataset.train_dataset_size = 1
    dataset.val_dataset_size = 1
    dataset.test_dataset_size = 1
    dataset.sample_size_action = 512
    dataset.sample_size_anchor = 1024
    dataset.downsample_type = "fps"
    dataset.scene_transform_type = "identity"
    dataset.translation_variance = 0.0
    dataset.rotation_variance = 0.0
    dataset.action_plane_occlusion = 0.0
    dataset.action_ball_occlusion = 0.0
    dataset.anchor_plane_occlusion = 0.0
    dataset.anchor_ball_occlusion = 0.0
    training.batch_size = 1
    training.val_batch_size = 1
    training.num_training_steps = 1
    training.sample_size = 1536
    training.sample_size_anchor = 1024
    training.additional_train_logging_period = 10 ** 9
    training.prediction_error_type = "demo"
    return omegaconf.OmegaConf.create(
        {
            "mode": "train",
            "seed": 7,
            "model": model,
            "dataset": dataset,
            "training": training,
            "resources": {"num_workers": 0, "gpus": [0]},
        }
    )


def _batch(dataset: RPDiffDataset) -> dict[str, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    value = next(iter(loader))
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping batch, got {type(value).__name__}")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.repo_root).resolve()
    data_root = Path(args.data_root).resolve()
    supplied_checkpoint = Path(args.checkpoint).expanduser()
    checkpoint_path = supplied_checkpoint.resolve()
    supplied_report = Path(args.report).expanduser()
    report_path = supplied_report.resolve()
    if supplied_checkpoint.is_symlink():
        raise ValueError(f"checkpoint path cannot be a symlink: {supplied_checkpoint}")
    if supplied_report.exists() or supplied_report.is_symlink():
        raise FileExistsError(f"refusing to overwrite report: {supplied_report}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device but CUDA is unavailable")

    artifact = load_placegen_observation(args.input_npz, args.manifest)
    cfg = _config(root, data_root, args.task_name, args.task_type)
    if len(artifact.pc_action_centered) != int(cfg.dataset.sample_size_action):
        raise ValueError(
            "observation action point count differs from the training config"
        )
    if len(artifact.pc_anchor_centered) != int(cfg.dataset.sample_size_anchor):
        raise ValueError(
            "observation anchor point count differs from the training config"
        )
    training_npz = (
        data_root / args.task_name / args.task_type / f"{artifact.sample_id}.npz"
    )
    if not training_npz.is_file() or training_npz.is_symlink():
        raise FileNotFoundError(
            f"training NPZ must be a regular file at the expected path: {training_npz}"
        )
    manifest_payload = json.loads(
        Path(args.manifest).expanduser().resolve().read_text(encoding="utf-8")
    )
    declared_training = (
        manifest_payload.get("files", {}).get("training_npz")
        if isinstance(manifest_payload, dict)
        else None
    )
    if not isinstance(declared_training, dict):
        raise ValueError("manifest is missing files.training_npz provenance")
    declared_training_path = declared_training.get("path")
    if not isinstance(declared_training_path, str) or not declared_training_path:
        raise ValueError("manifest training NPZ path must be a non-empty string")
    manifest_root = Path(args.manifest).expanduser().resolve().parent
    expected_training_path = (manifest_root / declared_training_path).resolve()
    try:
        expected_training_path.relative_to(manifest_root)
    except ValueError as error:
        raise ValueError("manifest training NPZ path escapes its directory") from error
    if expected_training_path != training_npz:
        raise ValueError("training NPZ does not match the manifest-declared path")
    if declared_training.get("sha256") != sha256_file(training_npz):
        raise ValueError("training NPZ SHA-256 does not match the export manifest")
    with np.load(training_npz, allow_pickle=True) as training_export:
        if "placegen_scene_center_world" not in training_export.files:
            raise ValueError(
                "training NPZ is missing the PlaceGen canonical scene center extension"
            )
        exported_center = np.asarray(training_export["placegen_scene_center_world"])
        if exported_center.shape != (3,) or not np.array_equal(
            exported_center.astype(np.float64), artifact.scene_center_world
        ):
            raise ValueError(
                "training and inference NPZ files do not share the canonical scene center"
            )
    dataset = RPDiffDataset(
        cfg.dataset.data_dir and Path(cfg.dataset.data_dir), cfg.dataset, "train"
    )
    batch = _batch(dataset)
    training_action = batch["pc_action"][0].detach().cpu().numpy()
    training_anchor = batch["pc_anchor"][0].detach().cpu().numpy()
    training_input_hashes = {
        "pc_action_centered": sha256_array(training_action),
        "pc_anchor_centered": sha256_array(training_anchor),
    }
    if training_input_hashes != artifact.tensor_sha256:
        raise RuntimeError(
            "RPDiff loader tensors differ from the observation-only inference tensors"
        )
    training_model_input_hash = model_input_sha256(training_action, training_anchor)
    if training_model_input_hash != artifact.model_input_sha256:
        raise RuntimeError(
            "RPDiff loader model-input SHA differs from the inference artifact"
        )
    supervision_hashes = {
        "pc_goal_centered": sha256_array(batch["pc"][0].detach().cpu().numpy())
    }
    if artifact.supervision_tensor_sha256 is None:
        raise ValueError("manifest is missing pc_goal_centered supervision SHA-256")
    if supervision_hashes["pc_goal_centered"] != artifact.supervision_tensor_sha256:
        raise RuntimeError("RPDiff loader goal tensor differs from the manifest SHA")

    network = TAX3Dv2Network(cfg.model)
    module = TAX3Dv2FixedFrameModule(network=network, cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    batch = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }
    state_before = clone_state_dict(module.network.state_dict())
    parameters_before = {
        name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in module.network.named_parameters()
    }
    before_hash = state_dict_sha256(state_before)
    module.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    optimizer_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    x_start = module._get_x_start(batch)
    timestep = torch.tensor([0], dtype=torch.long, device=device)
    loss_dict = module.diffusion.training_losses(
        module.network,
        x_start,
        timestep,
        module._model_kwargs(batch),
    )
    loss = loss_dict["loss"].mean()
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite TAX-DPD smoke loss: {loss.item()}")
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0))
    if not np.isfinite(gradient_norm):
        raise RuntimeError(f"non-finite TAX-DPD smoke gradient norm: {gradient_norm}")
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    optimizer_step_latency_seconds = time.perf_counter() - optimizer_started
    state_after = clone_state_dict(module.network.state_dict())
    parameters_after = {
        name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in module.network.named_parameters()
    }
    non_finite_state = [
        name
        for name, value in state_after.items()
        if (value.is_floating_point() or value.is_complex())
        and not torch.isfinite(value).all()
    ]
    if non_finite_state:
        raise RuntimeError(
            f"optimizer produced non-finite model state: {non_finite_state[:5]}"
        )
    after_hash = state_dict_sha256(state_after)
    if before_hash == after_hash:
        raise RuntimeError(
            "TAX-DPD smoke optimizer step did not change network parameters"
        )
    changed_parameter_names = [
        name
        for name in sorted(parameters_before)
        if not torch.equal(parameters_before[name], parameters_after[name])
    ]
    if not changed_parameter_names:
        raise RuntimeError(
            "TAX-DPD optimizer state hash changed without a changed tensor"
        )
    maximum_parameter_update = max(
        float((parameters_after[name] - parameters_before[name]).abs().max())
        for name in changed_parameter_names
    )
    if not np.isfinite(maximum_parameter_update) or maximum_parameter_update <= 0.0:
        raise RuntimeError(
            "optimizer did not produce a finite positive parameter update"
        )

    checkpoint = {
        "schema": "placegen.taxdpd-one-step/0.1",
        "args": omegaconf.OmegaConf.to_container(cfg, resolve=True),
        "model_state_dict": state_after,
        "optimizer_state_dict": optimizer.state_dict(),
        "smoke": {
            "seed": args.seed,
            "timestep": 0,
            "sample_id": artifact.sample_id,
            "descriptor_sha256": artifact.descriptor_sha256,
            "input_npz_sha256": artifact.input_npz_sha256,
            "training_npz_sha256": sha256_file(training_npz),
            "model_input_sha256": artifact.model_input_sha256,
            "observation_input_tensor_sha256": training_input_hashes,
            "supervision_tensor_sha256": supervision_hashes,
            "model_state_sha256": after_hash,
            "parameter_hash_before": before_hash,
            "parameter_hash_after": after_hash,
            "loss": float(loss.detach().cpu()),
            "gradient_norm": gradient_norm,
            "changed_parameter_tensor_count": len(changed_parameter_names),
            "maximum_parameter_update": maximum_parameter_update,
        },
    }
    with checkpoint_path.open("xb") as stream:
        torch.save(checkpoint, stream)
    # Verify the serialized state in a fresh module before reporting success.
    reloaded_payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(reloaded_payload, dict):
        raise RuntimeError("saved TAX-DPD checkpoint is not a mapping")
    reloaded_cfg = omegaconf.OmegaConf.create(reloaded_payload["args"])
    reloaded_network = TAX3Dv2Network(reloaded_cfg.model)
    reloaded_network.load_state_dict(reloaded_payload["model_state_dict"])
    reloaded_optimizer = torch.optim.AdamW(
        reloaded_network.parameters(),
        lr=float(reloaded_cfg.training.lr),
        weight_decay=float(reloaded_cfg.training.weight_decay),
    )
    reloaded_optimizer.load_state_dict(reloaded_payload["optimizer_state_dict"])
    if not reloaded_optimizer.state:
        raise RuntimeError("reloaded TAX-DPD optimizer has no one-step state")
    reloaded_state_hash = state_dict_sha256(reloaded_network.state_dict())
    if reloaded_state_hash != after_hash:
        raise RuntimeError(
            "reloaded TAX-DPD checkpoint state hash differs from saved state"
        )
    reloaded_smoke = reloaded_payload.get("smoke", {})
    if reloaded_smoke.get("observation_input_tensor_sha256") != training_input_hashes:
        raise RuntimeError("reloaded checkpoint observation input hashes differ")
    if reloaded_smoke.get("supervision_tensor_sha256") != supervision_hashes:
        raise RuntimeError("reloaded checkpoint supervision hashes differ")
    report = {
        "schema": "placegen.taxdpd-one-step-report/0.1",
        "quality_claim": False,
        "model": "TAX-DPD-reconstructed-fixed-frame-w/o-GMM",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "sample_id": artifact.sample_id,
        "descriptor_sha256": artifact.descriptor_sha256,
        "input_npz_sha256": artifact.input_npz_sha256,
        "training_npz": str(training_npz),
        "training_npz_sha256": sha256_file(training_npz),
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256_file(Path(args.manifest).resolve()),
        "model_input_sha256": artifact.model_input_sha256,
        "model_input": artifact.model_input_metadata(),
        "observation_input_tensor_sha256": training_input_hashes,
        "supervision_tensor_sha256": supervision_hashes,
        "model_state_sha256_before": before_hash,
        "model_state_sha256_after": after_hash,
        "checkpoint_reload_verified": True,
        "optimizer_reload_verified": True,
        "loss": float(loss.detach().cpu()),
        "gradient_norm": gradient_norm,
        "parameter_changed": before_hash != after_hash,
        "changed_parameter_tensor_count": len(changed_parameter_names),
        "maximum_parameter_update": maximum_parameter_update,
        "optimizer_step_latency_seconds": optimizer_step_latency_seconds,
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "peak_vram_mib": (
            float(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
            if device.type == "cuda"
            else None
        ),
        "seed": args.seed,
        "timestep": 0,
        "task_name": args.task_name,
        "task_type": args.task_type,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
