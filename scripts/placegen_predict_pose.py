"""Ground-truth-free TAX-DPD pose prediction for a PlaceGen smoke export."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import omegaconf
import numpy as np
import torch

from non_rigid.models.tax3d_v2 import TAX3Dv2FixedFrameModule, TAX3Dv2Network
from non_rigid.utils.placegen_artifact import (
    MODEL_INPUT_PROFILE,
    load_placegen_observation,
    sha256_file,
)
from non_rigid.utils.rigid_transform import estimate_ordered_rigid_transform
from non_rigid.utils.state_digest import state_dict_sha256


def _translation_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(value, dtype=np.float64)
    return matrix


def _load_checkpoint(
    path: Path, device: torch.device
) -> tuple[TAX3Dv2FixedFrameModule, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "placegen.taxdpd-one-step/0.1"
    ):
        raise ValueError("checkpoint is not a PlaceGen TAX-DPD one-step checkpoint")
    cfg = omegaconf.OmegaConf.create(payload["args"])
    network = TAX3Dv2Network(cfg.model)
    module = TAX3Dv2FixedFrameModule(network=network, cfg=cfg).to(device)
    module.network.load_state_dict(payload["model_state_dict"])
    state_hash = state_dict_sha256(module.network.state_dict())
    smoke = payload.get("smoke")
    if not isinstance(smoke, dict):
        raise ValueError("checkpoint is missing its smoke provenance")
    declared_hash = smoke.get("model_state_sha256")
    if not isinstance(declared_hash, str):
        raise ValueError("checkpoint is missing its model state SHA-256")
    if state_hash != declared_hash:
        raise ValueError("checkpoint model state SHA-256 does not match its metadata")
    module.eval()
    return module, payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    supplied_input = Path(args.input_npz).expanduser()
    supplied_checkpoint = Path(args.checkpoint).expanduser()
    supplied_output = Path(args.output_json).expanduser()
    if supplied_output.exists() or supplied_output.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite prediction report: {supplied_output}"
        )
    if not supplied_checkpoint.is_file() or supplied_checkpoint.is_symlink():
        raise FileNotFoundError(
            f"checkpoint must be a regular file: {supplied_checkpoint}"
        )
    input_path = supplied_input.resolve()
    checkpoint_path = supplied_checkpoint.resolve()
    output_path = supplied_output.resolve()
    artifact = load_placegen_observation(input_path, args.manifest)
    scene_center = artifact.scene_center_world
    source_pose = artifact.source_world_from_object

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device but CUDA is unavailable")
    module, checkpoint_payload = _load_checkpoint(checkpoint_path, device)
    checkpoint_smoke = checkpoint_payload.get("smoke")
    if not isinstance(checkpoint_smoke, dict):
        raise ValueError("checkpoint is missing its smoke provenance")
    expected_provenance = {
        "sample_id": artifact.sample_id,
        "descriptor_sha256": artifact.descriptor_sha256,
        "input_npz_sha256": artifact.input_npz_sha256,
        "model_input_sha256": artifact.model_input_sha256,
    }
    mismatched = [
        name
        for name, expected in expected_provenance.items()
        if checkpoint_smoke.get(name) != expected
    ]
    if mismatched:
        raise ValueError(
            "checkpoint was not trained on this exact smoke input: "
            f"{sorted(mismatched)}"
        )
    action = (
        torch.from_numpy(np.array(artifact.pc_action_centered, copy=True, order="C"))
        .unsqueeze(0)
        .to(device)
    )
    anchor = (
        torch.from_numpy(np.array(artifact.pc_anchor_centered, copy=True, order="C"))
        .unsqueeze(0)
        .to(device)
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    with torch.inference_mode():
        candidate_tensor = module.sample_candidates(
            action,
            anchor,
            num_trials=int(args.candidates),
            seed=int(args.seed),
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    candidates = candidate_tensor[0].cpu().numpy().astype(np.float64)

    candidate_records: list[dict[str, Any]] = []
    center_to_world = _translation_matrix(scene_center)
    world_to_center = _translation_matrix(-scene_center)
    for index, predicted_centered in enumerate(candidates):
        fit = estimate_ordered_rigid_transform(
            artifact.pc_action_centered.astype(np.float64, copy=False),
            predicted_centered,
        )
        relative_world = center_to_world @ fit.matrix @ world_to_center
        absolute_world = relative_world @ source_pose
        candidate_records.append(
            {
                "candidate_index": index,
                "relative_world_from_source": relative_world.tolist(),
                "world_from_object": absolute_world.tolist(),
                "rigid_fit_rmse_m": fit.rmse,
                "rigid_fit_singular_values": fit.singular_values.tolist(),
                "det_rotation": float(np.linalg.det(relative_world[:3, :3])),
            }
        )

    state_hash = state_dict_sha256(module.network.state_dict())
    checkpoint_state_hash = checkpoint_smoke.get("model_state_sha256")
    report = {
        "schema": "placegen.taxdpd-pose-prediction/0.1",
        "quality_claim": False,
        "model": "TAX-DPD-reconstructed-fixed-frame-w/o-GMM",
        "model_input_profile": MODEL_INPUT_PROFILE,
        "sample_id": artifact.sample_id,
        "descriptor_sha256": artifact.descriptor_sha256,
        "input_npz": str(input_path),
        "input_npz_sha256": artifact.input_npz_sha256,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256_file(Path(args.manifest).resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_model_state_sha256": state_hash,
        "checkpoint_reload_verified": checkpoint_state_hash == state_hash,
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "peak_vram_mib": (
            float(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
            if device.type == "cuda"
            else None
        ),
        "seed": int(args.seed),
        "candidate_count": len(candidate_records),
        "model_input_sha256": artifact.model_input_sha256,
        "input_tensor_sha256": dict(artifact.tensor_sha256),
        "model_input": artifact.model_input_metadata(),
        "inference_latency_seconds": latency_ms / 1000.0,
        "latency_ms": latency_ms,
        "ground_truth_free": True,
        "observation_fields": [
            "action_indices",
            "anchor_indices",
            "child_points_world",
            "parent_points_world",
            "scene_center_world",
            "source_world_from_object",
        ],
        "source_pose_used_only_for_world_composition": True,
        "diagnostics": {
            "candidate_world_poses_are_not_scored_here": True,
            "target_pose_or_goal_point_cloud_read": False,
        },
        "candidates": candidate_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.candidates <= 0:
        raise SystemExit("--candidates must be positive")
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
