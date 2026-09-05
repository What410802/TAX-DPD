"""Serialization and strict reload for the Euclidean FM PlaceGen pilot."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import omegaconf
import torch

from non_rigid.models.tax3d_v2 import (
    TAX3Dv2FixedFrameFlowMatchingModule,
    TAX3Dv2Network,
)
from non_rigid.utils.state_digest import state_dict_sha256

FM_CHECKPOINT_SCHEMA = "placegen.tax3dv2-fm-checkpoint/0.1"
FM_MODEL_CAPABILITY = "TAX-DPD-reconstructed-fixed-frame-euclidean-fm"


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedFMCheckpoint:
    """A validated FM module and immutable provenance."""

    path: Path
    payload: dict[str, Any]
    module: TAX3Dv2FixedFrameFlowMatchingModule
    model_state_sha256: str


def validate_fm_checkpoint_payload(payload: Any) -> dict[str, Any]:
    """Validate the fail-closed FM checkpoint envelope before construction."""

    if not isinstance(payload, dict) or payload.get("schema") != FM_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a PlaceGen TAX3Dv2 FM checkpoint")
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
        raise ValueError(
            "FM checkpoint fields must match schema exactly; "
            f"missing={sorted(required.difference(payload))}, "
            f"unexpected={sorted(set(payload).difference(required))}"
        )
    if payload.get("model") != FM_MODEL_CAPABILITY:
        raise ValueError("checkpoint model capability is unsupported")
    if not isinstance(payload["config"], dict):
        raise TypeError("checkpoint config must be a resolved mapping")
    dataset = payload["dataset"]
    if not isinstance(dataset, dict):
        raise TypeError("checkpoint dataset provenance must be a mapping")
    dataset_required = {
        "training_manifest_sha256",
        "inference_manifest_sha256",
        "split_assignment_sha256",
        "native_dataset_sha256",
        "data_root",
        "point_counts",
        "split_counts",
    }
    if set(dataset) != dataset_required:
        raise ValueError("FM checkpoint dataset provenance fields are invalid")
    _sha256(dataset["training_manifest_sha256"], "training_manifest_sha256")
    _sha256(dataset["inference_manifest_sha256"], "inference_manifest_sha256")
    _sha256(dataset["split_assignment_sha256"], "split_assignment_sha256")
    _sha256(dataset["native_dataset_sha256"], "native_dataset_sha256")
    if not isinstance(dataset["data_root"], str) or not dataset["data_root"]:
        raise ValueError("checkpoint data_root must be a non-empty string")
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
    training = payload["training"]
    if not isinstance(training, dict):
        raise TypeError("checkpoint training provenance must be a mapping")
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
    if not training_required.issubset(training) or set(training).difference(
        training_required | training_optional
    ):
        raise ValueError("FM checkpoint training provenance fields are invalid")
    if training["selection_split"] != "validation":
        raise ValueError("FM checkpoint must be selected on validation")
    if training["test_split_used_for_selection"] is not False:
        raise ValueError("FM checkpoint selection permits test leakage")
    if not isinstance(training["history"], list) or not training["history"]:
        raise ValueError("FM checkpoint history must be non-empty")
    if not isinstance(payload["model_state_dict"], Mapping):
        raise TypeError("FM checkpoint model_state_dict must be a mapping")
    _sha256(payload["model_state_sha256"], "model_state_sha256")
    return payload


def load_fm_checkpoint(
    path: Path | str,
    device: torch.device,
    *,
    expected_training_manifest_sha256: str | None = None,
    expected_inference_manifest_sha256: str | None = None,
) -> LoadedFMCheckpoint:
    """Reload a complete FM checkpoint and verify config/state identities."""

    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise FileNotFoundError(f"checkpoint must be a regular file: {supplied}")
    checkpoint_path = supplied.resolve(strict=True)
    payload = validate_fm_checkpoint_payload(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )
    dataset = payload["dataset"]
    if (
        expected_training_manifest_sha256 is not None
        and dataset["training_manifest_sha256"] != expected_training_manifest_sha256
    ):
        raise ValueError("checkpoint training manifest SHA-256 does not match")
    if (
        expected_inference_manifest_sha256 is not None
        and dataset["inference_manifest_sha256"] != expected_inference_manifest_sha256
    ):
        raise ValueError("checkpoint inference manifest SHA-256 does not match")
    cfg = omegaconf.OmegaConf.create(payload["config"])
    if cfg.model.name != "tax3dv2_fm" or cfg.model.frame_type != "fixed":
        raise ValueError("FM checkpoint config is not fixed-frame tax3dv2_fm")
    network = TAX3Dv2Network(cfg.model)
    module = TAX3Dv2FixedFrameFlowMatchingModule(network=network, cfg=cfg).to(device)
    module.network.load_state_dict(payload["model_state_dict"], strict=True)
    actual_sha = state_dict_sha256(module.network.state_dict())
    if actual_sha != payload["model_state_sha256"]:
        raise ValueError("FM checkpoint model state SHA-256 does not match metadata")
    module.eval()
    return LoadedFMCheckpoint(
        path=checkpoint_path,
        payload=payload,
        module=module,
        model_state_sha256=actual_sha,
    )


__all__ = [
    "FM_CHECKPOINT_SCHEMA",
    "FM_MODEL_CAPABILITY",
    "LoadedFMCheckpoint",
    "load_fm_checkpoint",
    "validate_fm_checkpoint_payload",
]
