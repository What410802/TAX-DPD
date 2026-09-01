"""Run one Utonia encoder forward and verify TAX-DPD slot preservation.

This script intentionally has no TAX-DPD model import.  It is a small
environment gate for the optional Utonia backend: the two role clouds are
transformed independently, encoded with the published frozen checkpoint, and
upcast through Utonia's pooling/grid inverse chain to the original 1024 slots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import utonia
from non_rigid.models.utonia_slot_adapter import (
    build_utonia_input,
    upcast_slot_features,
    validate_role_slots,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args()


def encode_role(model: torch.nn.Module, coordinates: np.ndarray, *, scale: float, device: torch.device):
    # Utonia's documented custom-data contract requires zero placeholders when
    # color/normal are unavailable; TAX-DPD flow features are not modalities.
    transformed = utonia.transform.default(
        scale=scale, apply_z_positive=True, normalize_coord=False
    )(build_utonia_input(np.asarray(coordinates, dtype=np.float32).copy()))
    sampled_count = int(transformed["coord"].shape[0])
    for key, value in tuple(transformed.items()):
        if isinstance(value, torch.Tensor):
            transformed[key] = value.to(device=device, non_blocking=True)
    with torch.inference_mode():
        encoded = model(transformed)
        features = upcast_slot_features(encoded)
    return features, sampled_count


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    payload = np.load(args.input_npz, allow_pickle=False)
    child = np.asarray(payload["child_points_world"], dtype=np.float32)
    anchor = np.asarray(payload["parent_points_world"], dtype=np.float32)
    if child.shape != (1024, 3) or anchor.shape != (1024, 3):
        raise ValueError(f"expected two [1024, 3] clouds, got {child.shape} and {anchor.shape}")

    # Utonia's loader applies the checkpoint config.  Disable FlashAttention
    # because this is also the documented fallback for environments without
    # the optional flash_attn wheel; reduce no other model capacity.
    model = utonia.load(
        str(args.checkpoint),
        custom_config=dict(
            enc_patch_size=[1024 for _ in range(5)],
            enable_flash=False,
        ),
    ).to(device)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    start = time.perf_counter()
    child_features, child_sampled = encode_role(
        model, child, scale=args.scale, device=device
    )
    anchor_features, anchor_sampled = encode_role(
        model, anchor, scale=args.scale, device=device
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    validate_role_slots(child_features, child, role="action")
    validate_role_slots(anchor_features, anchor, role="anchor")
    if child_features.shape[1] != anchor_features.shape[1]:
        raise ValueError("Utonia role feature widths differ")
    result = {
        "schema": "taxdpd.utonia-frozen-forward-smoke/0.1",
        "input_npz": str(args.input_npz),
        "input_npz_sha256": sha256_file(args.input_npz),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "device": str(device),
        "model_parameters": int(parameter_count),
        "scale": float(args.scale),
        "original_slot_count": {"action": int(child.shape[0]), "anchor": int(anchor.shape[0])},
        "voxel_slot_count": {"action": child_sampled, "anchor": anchor_sampled},
        "feature_shape": {
            "action": list(child_features.shape),
            "anchor": list(anchor_features.shape),
        },
        "finite": bool(torch.isfinite(child_features).all() and torch.isfinite(anchor_features).all()),
        "elapsed_seconds": elapsed,
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "flash_attention_enabled": False,
        "frozen": True,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
