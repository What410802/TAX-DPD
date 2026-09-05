"""Serialization and strict reload for the reconstructed PlaceGen DDPM baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import omegaconf
import torch

from non_rigid.models.tax3d_v2 import TAX3Dv2FixedFrameModule, TAX3Dv2Network
from non_rigid.utils.state_digest import state_dict_sha256

DDPM_CHECKPOINT_SCHEMA = "placegen.tax3dv2-ddpm-checkpoint/0.1"
DDPM_MODEL_CAPABILITY = "TAX-DPD-reconstructed-fixed-frame-DDPM"


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class LoadedDDPMCheckpoint:
    path: Path
    payload: dict[str, Any]
    module: TAX3Dv2FixedFrameModule
    model_state_sha256: str


def validate_ddpm_checkpoint_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema") != DDPM_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a PlaceGen TAX3Dv2 DDPM checkpoint")
    required = {
        "schema",
        "model",
        "config",
        "dataset",
        "training",
        "model_state_dict",
        "model_state_sha256",
    }
    if set(payload) != required:
        raise ValueError("DDPM checkpoint fields must match schema exactly")
    if payload.get("model") != DDPM_MODEL_CAPABILITY:
        raise ValueError("checkpoint model capability is unsupported")
    if not isinstance(payload["config"], dict):
        raise TypeError("checkpoint config must be a resolved mapping")
    dataset = payload["dataset"]
    dataset_required = {
        "training_manifest_sha256",
        "inference_manifest_sha256",
        "split_assignment_sha256",
        "native_dataset_sha256",
        "data_root",
        "point_counts",
        "split_counts",
    }
    if not isinstance(dataset, dict) or set(dataset) != dataset_required:
        raise ValueError("DDPM checkpoint dataset provenance fields are invalid")
    if not isinstance(dataset["point_counts"], dict) or set(dataset["point_counts"]) != {
        "action",
        "anchor",
    }:
        raise ValueError("checkpoint point_counts are invalid")
    if not isinstance(dataset["split_counts"], dict) or set(dataset["split_counts"]) != {
        "train",
        "validation",
        "test",
    }:
        raise ValueError("checkpoint split_counts are invalid")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in dataset["split_counts"].values()
    ):
        raise ValueError("checkpoint split_counts must be positive integers")
    for name in (
        "training_manifest_sha256",
        "inference_manifest_sha256",
        "split_assignment_sha256",
        "native_dataset_sha256",
    ):
        _sha256(dataset[name], name)
    training = payload["training"]
    training_required = {
        "seed",
        "epochs",
        "train_count",
        "validation_count",
        "best_validation_loss",
        "history",
        "selection_split",
        "test_split_used_for_selection",
    }
    training_optional = {
        "batch_size",
        "validation_batch_size",
        "optimizer_steps_per_epoch",
        "validation_batches_per_epoch",
        "peak_vram_mib",
        "peak_reserved_mib",
    }
    if (
        not isinstance(training, dict)
        or not training_required.issubset(training)
        or set(training).difference(training_required | training_optional)
    ):
        raise ValueError("DDPM checkpoint training provenance fields are invalid")
    if training["selection_split"] != "validation":
        raise ValueError("DDPM checkpoint must be selected on validation")
    if training["test_split_used_for_selection"] is not False:
        raise ValueError("DDPM checkpoint selection permits test leakage")
    if not isinstance(payload["model_state_dict"], Mapping):
        raise TypeError("DDPM checkpoint model_state_dict must be a mapping")
    _sha256(payload["model_state_sha256"], "model_state_sha256")
    return payload


def load_ddpm_checkpoint(
    path: Path | str,
    device: torch.device,
    *,
    expected_inference_manifest_sha256: str | None = None,
) -> LoadedDDPMCheckpoint:
    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise FileNotFoundError(f"checkpoint must be a regular file: {supplied}")
    checkpoint_path = supplied.resolve(strict=True)
    payload = validate_ddpm_checkpoint_payload(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )
    if (
        expected_inference_manifest_sha256 is not None
        and payload["dataset"]["inference_manifest_sha256"]
        != expected_inference_manifest_sha256
    ):
        raise ValueError("checkpoint inference manifest SHA-256 does not match")
    cfg = omegaconf.OmegaConf.create(payload["config"])
    if cfg.model.name != "tax3dv2" or cfg.model.frame_type != "fixed":
        raise ValueError("DDPM checkpoint config is not fixed-frame tax3dv2")
    network = TAX3Dv2Network(cfg.model)
    module = TAX3Dv2FixedFrameModule(network=network, cfg=cfg).to(device)
    module.network.load_state_dict(payload["model_state_dict"], strict=True)
    actual_sha = state_dict_sha256(module.network.state_dict())
    if actual_sha != payload["model_state_sha256"]:
        raise ValueError("DDPM checkpoint model state SHA-256 does not match metadata")
    module.eval()
    return LoadedDDPMCheckpoint(
        path=checkpoint_path,
        payload=payload,
        module=module,
        model_state_sha256=actual_sha,
    )


__all__ = [
    "DDPM_CHECKPOINT_SCHEMA",
    "DDPM_MODEL_CAPABILITY",
    "LoadedDDPMCheckpoint",
    "load_ddpm_checkpoint",
    "validate_ddpm_checkpoint_payload",
]
