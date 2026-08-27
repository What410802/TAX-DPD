"""Grouped TAX-DPD checkpoint serialization and reload validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import omegaconf
import torch

from non_rigid.models.tax3d_v2 import TAX3Dv2FixedFrameModule, TAX3Dv2Network
from non_rigid.utils.placegen_grouped import MODEL_CAPABILITY
from non_rigid.utils.state_digest import state_dict_sha256

GROUPED_CHECKPOINT_SCHEMA = "placegen.taxdpd-grouped-checkpoint/0.1"


def _nested_state_update(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    elif value is None:
        digest.update(b"none\0")
    elif isinstance(value, bool):
        digest.update(b"bool\0" + (b"1" if value else b"0"))
    elif isinstance(value, int):
        digest.update(b"int\0" + str(value).encode() + b"\0")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("optimizer state contains a non-finite float")
        digest.update(b"float\0" + value.hex().encode() + b"\0")
    elif isinstance(value, str):
        digest.update(b"str\0" + value.encode() + b"\0")
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence\0" + str(len(value)).encode() + b"\0")
        for item in value:
            _nested_state_update(digest, item)
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0" + str(len(value)).encode() + b"\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _nested_state_update(digest, key)
            _nested_state_update(digest, value[key])
    else:
        raise TypeError(f"unsupported optimizer state value: {type(value).__name__}")


def optimizer_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash nested optimizer tensors and scalar hyperparameters deterministically."""

    digest = hashlib.sha256()
    digest.update(b"placegen.torch-optimizer-state/0.1\0")
    _nested_state_update(digest, state)
    return digest.hexdigest()


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class LoadedGroupedCheckpoint:
    """Reloaded model plus immutable checkpoint metadata."""

    path: Path
    payload: dict[str, Any]
    module: TAX3Dv2FixedFrameModule
    model_state_sha256: str
    optimizer_state_sha256: str


def validate_checkpoint_payload(
    payload: Any,
    *,
    expected_dataset_manifest_sha256: str | None = None,
    expected_split_assignment_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate schema and selection provenance before state construction."""

    if (
        not isinstance(payload, dict)
        or payload.get("schema") != GROUPED_CHECKPOINT_SCHEMA
    ):
        raise ValueError("checkpoint is not a grouped PlaceGen TAX-DPD checkpoint")
    if payload.get("model") != MODEL_CAPABILITY:
        raise ValueError("checkpoint model capability is unsupported")
    required = {
        "schema",
        "model",
        "config",
        "dataset",
        "training",
        "model_state_dict",
        "optimizer_state_dict",
        "model_state_sha256",
        "optimizer_state_sha256",
    }
    if set(payload) != required:
        raise ValueError(
            "checkpoint fields must match schema exactly; "
            f"missing={sorted(required.difference(payload))}, "
            f"unexpected={sorted(set(payload).difference(required))}"
        )
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise TypeError("checkpoint dataset provenance must be a mapping")
    dataset_required = {
        "manifest_sha256",
        "split_assignment_sha256",
        "native_dataset_sha256",
        "task_name",
        "task_type",
        "point_counts",
        "sample_counts",
        "group_counts",
        "quality_scope",
    }
    if set(dataset) != dataset_required:
        raise ValueError("checkpoint dataset provenance fields are invalid")
    manifest_sha = _sha256(dataset.get("manifest_sha256"), "dataset.manifest_sha256")
    assignment_sha = _sha256(
        dataset.get("split_assignment_sha256"), "dataset.split_assignment_sha256"
    )
    _sha256(dataset.get("native_dataset_sha256"), "dataset.native_dataset_sha256")
    if (
        expected_dataset_manifest_sha256 is not None
        and manifest_sha != expected_dataset_manifest_sha256
    ):
        raise ValueError("checkpoint dataset manifest SHA-256 does not match")
    if (
        expected_split_assignment_sha256 is not None
        and assignment_sha != expected_split_assignment_sha256
    ):
        raise ValueError("checkpoint split assignment SHA-256 does not match")
    training = payload.get("training")
    if not isinstance(training, dict):
        raise TypeError("checkpoint training provenance must be a mapping")
    training_required = {
        "epoch",
        "global_step",
        "seed",
        "validation_seed",
        "train_loss_mean",
        "gradient_norm_mean",
        "validation_candidate0_ordered_rmse_m",
        "selection_split",
        "selection_candidate_index",
        "validation_sample_ids",
        "test_split_used_for_selection",
    }
    if set(training) != training_required:
        raise ValueError("checkpoint training provenance fields are invalid")
    if training.get("selection_split") != "validation":
        raise ValueError("checkpoint was not selected on validation")
    if training.get("selection_candidate_index") != 0:
        raise ValueError("checkpoint selection must use candidate zero")
    if training.get("test_split_used_for_selection") is not False:
        raise ValueError("checkpoint selection provenance permits test leakage")
    validation_ids = training.get("validation_sample_ids")
    if (
        not isinstance(validation_ids, list)
        or not validation_ids
        or any(not isinstance(value, str) or not value for value in validation_ids)
        or len(set(validation_ids)) != len(validation_ids)
    ):
        raise ValueError("checkpoint validation sample IDs are invalid")
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise TypeError("checkpoint model_state_dict must be a mapping")
    if not isinstance(payload.get("optimizer_state_dict"), Mapping):
        raise TypeError("checkpoint optimizer_state_dict must be a mapping")
    _sha256(payload.get("model_state_sha256"), "model_state_sha256")
    _sha256(payload.get("optimizer_state_sha256"), "optimizer_state_sha256")
    if not isinstance(payload.get("config"), dict):
        raise TypeError("checkpoint config must be a resolved mapping")
    return payload


def load_grouped_checkpoint(
    path: Path | str,
    device: torch.device,
    *,
    expected_dataset_manifest_sha256: str | None = None,
    expected_split_assignment_sha256: str | None = None,
) -> LoadedGroupedCheckpoint:
    """Reload a grouped model checkpoint and verify model state bytes."""

    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise FileNotFoundError(f"checkpoint must be a regular file: {supplied}")
    checkpoint_path = supplied.resolve(strict=True)
    payload = validate_checkpoint_payload(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
        expected_dataset_manifest_sha256=expected_dataset_manifest_sha256,
        expected_split_assignment_sha256=expected_split_assignment_sha256,
    )
    cfg = omegaconf.OmegaConf.create(payload["config"])
    if cfg.model.frame_type != "fixed":
        raise ValueError("grouped checkpoint is not fixed-frame TAX-DPD")
    network = TAX3Dv2Network(cfg.model)
    module = TAX3Dv2FixedFrameModule(network=network, cfg=cfg).to(device)
    module.network.load_state_dict(payload["model_state_dict"], strict=True)
    actual_model_sha = state_dict_sha256(module.network.state_dict())
    if actual_model_sha != payload["model_state_sha256"]:
        raise ValueError("checkpoint model state SHA-256 does not match metadata")
    actual_optimizer_sha = optimizer_state_sha256(payload["optimizer_state_dict"])
    if actual_optimizer_sha != payload["optimizer_state_sha256"]:
        raise ValueError("checkpoint optimizer state SHA-256 does not match metadata")
    module.eval()
    return LoadedGroupedCheckpoint(
        path=checkpoint_path,
        payload=payload,
        module=module,
        model_state_sha256=actual_model_sha,
        optimizer_state_sha256=actual_optimizer_sha,
    )


__all__ = [
    "GROUPED_CHECKPOINT_SCHEMA",
    "LoadedGroupedCheckpoint",
    "load_grouped_checkpoint",
    "optimizer_state_sha256",
    "validate_checkpoint_payload",
]
