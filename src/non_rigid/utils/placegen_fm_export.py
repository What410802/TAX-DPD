"""Strict native PlaceGen export reader for TAX3Dv2 FM experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

TRAINING_PROFILE = "placegen.taxpose-rpdiff/0.1"
INFERENCE_PROFILE = "placegen.taxpose-inference-index/0.1"
SPLITS = ("train", "validation", "test")
EXPECTED_POINT_COUNTS = {"action": 1024, "anchor": 1024}
OBSERVATION_FIELDS = frozenset(
    {
        "action_indices",
        "anchor_indices",
        "child_points_world",
        "parent_points_world",
        "scene_center_world",
        "source_world_from_object",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _split_counts(value: Any, name: str) -> dict[str, int]:
    counts = _mapping(value, name)
    if set(counts) != set(SPLITS):
        raise ValueError(f"{name} must contain exactly {list(SPLITS)}")
    for split, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{name}.{split} must be a positive integer")
    return {split: int(counts[split]) for split in SPLITS}


def _exact_fields(value: dict[str, Any], expected: frozenset[str], name: str) -> None:
    if frozenset(value) != expected:
        raise ValueError(
            f"{name} fields must match schema exactly; "
            f"missing={sorted(expected.difference(value))}, "
            f"unexpected={sorted(set(value).difference(expected))}"
        )


def _exact_fields_with_optional(
    value: dict[str, Any],
    expected: frozenset[str],
    optional: frozenset[str],
    name: str,
) -> None:
    """Accept a versioned optional extension without permitting arbitrary fields."""

    allowed = expected | optional
    if not expected.issubset(value) or not frozenset(value).issubset(allowed):
        raise ValueError(
            f"{name} fields must match schema exactly; "
            f"missing={sorted(expected.difference(value))}, "
            f"unexpected={sorted(set(value).difference(allowed))}"
        )


def _validate_sampling_record(value: Any, name: str) -> None:
    record = _mapping(value, name)
    _exact_fields(
        record,
        frozenset(
            {
                "action_indices_sha256",
                "anchor_indices_sha256",
                "method",
                "sample_seed",
            }
        ),
        name,
    )
    if record.get("method") != "seeded_random_permutation":
        raise ValueError(f"{name}.method is unsupported")
    sample_seed = record.get("sample_seed")
    if isinstance(sample_seed, bool) or not isinstance(sample_seed, int) or sample_seed < 0:
        raise ValueError(f"{name}.sample_seed must be a non-negative integer")
    _sha256(record.get("action_indices_sha256"), f"{name}.action_indices_sha256")
    _sha256(record.get("anchor_indices_sha256"), f"{name}.anchor_indices_sha256")


def _regular_json(path: Path | str, name: str) -> tuple[Path, dict[str, Any]]:
    supplied = Path(path).expanduser()
    if supplied.is_symlink() or not supplied.is_file():
        raise FileNotFoundError(f"{name} must be a regular file: {supplied}")
    resolved = supplied.resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    return resolved, _mapping(value, name)


@dataclass(frozen=True)
class NativeExportIdentity:
    training_manifest_path: Path
    training_manifest_sha256: str
    inference_manifest_path: Path
    inference_manifest_sha256: str
    split_assignment_sha256: str
    native_dataset_sha256: str
    point_counts: dict[str, int]
    split_counts: dict[str, int]


@dataclass(frozen=True)
class NativeInferenceSample:
    sample_id: str
    split: str
    split_group_id: str
    descriptor_sha256: str
    input_npz: Path
    input_npz_sha256: str


@dataclass(frozen=True)
class NativeInferenceIndex:
    manifest_path: Path
    manifest_sha256: str
    split_assignment_sha256: str
    native_dataset_sha256: str
    point_counts: dict[str, int]
    split_counts: dict[str, int]
    samples_by_split: dict[str, tuple[NativeInferenceSample, ...]]


@dataclass(frozen=True)
class FMObservation:
    sample: NativeInferenceSample
    action_model: np.ndarray
    anchor_model: np.ndarray
    source_world_from_object: np.ndarray
    model_center_world: np.ndarray
    scale_factor: float
    model_input_sha256: dict[str, str]


def load_native_export_identity(
    training_manifest_path: Path | str,
    inference_manifest_path: Path | str,
) -> NativeExportIdentity:
    """Bind training metadata to the target-free native inference index."""

    training_path, training = _regular_json(training_manifest_path, "training manifest")
    inference = load_native_inference_index(inference_manifest_path)
    if training.get("profile") != TRAINING_PROFILE:
        raise ValueError("unexpected native PlaceGen training manifest profile")
    if training.get("evaluation_valid") is not True or training.get("correlated_split") is not False:
        raise ValueError("training manifest is not evaluation-valid and disjoint")
    if training.get("quality_claim") is not False:
        raise ValueError("training export cannot make a quality claim")
    if training.get("point_counts") != EXPECTED_POINT_COUNTS:
        raise ValueError("unexpected native PlaceGen point counts")
    split_counts = _split_counts(training.get("split_counts"), "split_counts")
    if training.get("sample_count") != sum(split_counts.values()):
        raise ValueError("unexpected native PlaceGen sample count")
    assignment = _mapping(training.get("split_assignment"), "split_assignment")
    assignment_sha = _sha256(assignment.get("manifest_sha256"), "manifest_sha256")
    native_sha = _sha256(assignment.get("native_dataset_sha256"), "native_dataset_sha256")
    if assignment_sha != inference.split_assignment_sha256:
        raise ValueError("training/inference split assignment identities differ")
    if native_sha != inference.native_dataset_sha256:
        raise ValueError("training/inference native dataset identities differ")
    if dict(training["point_counts"]) != inference.point_counts:
        raise ValueError("training/inference point counts differ")
    if split_counts != inference.split_counts:
        raise ValueError("training/inference split counts differ")
    return NativeExportIdentity(
        training_manifest_path=training_path,
        training_manifest_sha256=sha256_file(training_path),
        inference_manifest_path=inference.manifest_path,
        inference_manifest_sha256=inference.manifest_sha256,
        split_assignment_sha256=assignment_sha,
        native_dataset_sha256=native_sha,
        point_counts=dict(training["point_counts"]),
        split_counts=split_counts,
    )


def load_native_inference_index(path: Path | str) -> NativeInferenceIndex:
    """Load only the ground-truth-free PlaceGen inference manifest."""

    manifest_path, manifest = _regular_json(path, "inference manifest")
    expected_top = frozenset(
        {
            "centering",
            "correlated_split",
            "evaluation_valid",
            "ground_truth_free",
            "model",
            "model_input_profile",
            "point_counts",
            "profile",
            "quality_scope",
            "sample_count",
            "samples",
            "split_assignment",
        }
    )
    _exact_fields_with_optional(manifest, expected_top, frozenset({"sampling"}), "inference manifest")
    if manifest.get("profile") != INFERENCE_PROFILE:
        raise ValueError("unexpected native PlaceGen inference manifest profile")
    if (
        manifest.get("evaluation_valid") is not True
        or manifest.get("correlated_split") is not False
        or manifest.get("ground_truth_free") is not True
    ):
        raise ValueError("inference manifest must be target-free and evaluation-valid")
    if manifest.get("point_counts") != EXPECTED_POINT_COUNTS:
        raise ValueError("unexpected native PlaceGen inference point counts")
    if "sampling" in manifest:
        sampling = _mapping(manifest["sampling"], "sampling")
        if sampling.get("method") != "seeded_random_permutation":
            raise ValueError("unsupported input sampling method")
        if sampling.get("profile") != "placegen.fixed-count-seeded-random-permutation/0.1":
            raise ValueError("unexpected input sampling profile")
        if sampling.get("replacement") is not False:
            raise ValueError("input sampling must be without replacement")
        if sampling.get("draw_order") != ["parent", "child"]:
            raise ValueError("unexpected input sampling draw order")
    assignment = _mapping(manifest.get("split_assignment"), "split_assignment")
    _exact_fields(
        assignment,
        frozenset({"manifest_sha256", "native_dataset_sha256", "profile"}),
        "split_assignment",
    )
    assignment_sha = _sha256(assignment.get("manifest_sha256"), "manifest_sha256")
    native_sha = _sha256(assignment.get("native_dataset_sha256"), "native_dataset_sha256")
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != manifest.get("sample_count"):
        raise ValueError("inference samples do not match sample_count")
    root = manifest_path.parent
    samples: dict[str, list[NativeInferenceSample]] = {split: [] for split in SPLITS}
    seen: set[str] = set()
    sample_fields = frozenset(
        {
            "descriptor_sha256",
            "input_npz",
            "input_tensor_sha256",
            "sample_id",
            "split",
            "split_group_id",
        }
    )
    for index, raw in enumerate(raw_samples):
        sample = _mapping(raw, f"samples[{index}]")
        _exact_fields_with_optional(sample, sample_fields, frozenset({"sampling"}), f"samples[{index}]")
        if "sampling" in sample:
            _validate_sampling_record(sample["sampling"], f"samples[{index}].sampling")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError("inference sample IDs must be unique non-empty strings")
        seen.add(sample_id)
        split = sample.get("split")
        if split not in SPLITS:
            raise ValueError(f"unsupported inference split: {split!r}")
        group_id = sample.get("split_group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("split_group_id must be a non-empty string")
        descriptor_sha = _sha256(sample.get("descriptor_sha256"), "descriptor_sha256")
        input_record = _mapping(sample.get("input_npz"), "input_npz")
        _exact_fields(input_record, frozenset({"path", "sha256"}), "input_npz")
        declared = input_record.get("path")
        if not isinstance(declared, str) or not declared or declared.startswith("/") or "\\" in declared:
            raise ValueError("input_npz.path must be a relative POSIX path")
        relative = PurePosixPath(declared)
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("input_npz.path contains an unsafe component")
        input_path = root.joinpath(*relative.parts)
        if input_path.is_symlink() or not input_path.is_file():
            raise FileNotFoundError(f"input NPZ must be a regular file: {input_path}")
        input_path = input_path.resolve(strict=True)
        try:
            input_path.relative_to(root)
        except ValueError as error:
            raise ValueError("input_npz.path escapes the inference root") from error
        input_sha = _sha256(input_record.get("sha256"), "input_npz.sha256")
        if sha256_file(input_path) != input_sha:
            raise ValueError("input NPZ SHA-256 does not match its manifest")
        samples[split].append(
            NativeInferenceSample(
                sample_id=sample_id,
                split=split,
                split_group_id=group_id,
                descriptor_sha256=descriptor_sha,
                input_npz=input_path,
                input_npz_sha256=input_sha,
            )
        )
    counts = {split: len(samples[split]) for split in SPLITS}
    _split_counts(counts, "inference split counts")
    return NativeInferenceIndex(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        split_assignment_sha256=assignment_sha,
        native_dataset_sha256=native_sha,
        point_counts=dict(manifest["point_counts"]),
        split_counts=counts,
        samples_by_split={
            split: tuple(sorted(samples[split], key=lambda item: item.sample_id))
            for split in SPLITS
        },
    )


def load_fm_observation(
    sample: NativeInferenceSample,
    *,
    scale_factor: float,
) -> FMObservation:
    """Reconstruct the exact joint-centered, scaled FM model tensors."""

    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("scale_factor must be finite and positive")
    if sha256_file(sample.input_npz) != sample.input_npz_sha256:
        raise ValueError("input NPZ changed after manifest validation")
    with np.load(sample.input_npz, allow_pickle=False) as data:
        if frozenset(data.files) != OBSERVATION_FIELDS:
            raise ValueError("inference NPZ must contain exactly observation-only fields")
        action = np.asarray(data["child_points_world"])
        anchor = np.asarray(data["parent_points_world"])
        action_indices = np.asarray(data["action_indices"])
        anchor_indices = np.asarray(data["anchor_indices"])
        source_pose = np.asarray(data["source_world_from_object"])
    for points, count, name in (
        (action, EXPECTED_POINT_COUNTS["action"], "child_points_world"),
        (anchor, EXPECTED_POINT_COUNTS["anchor"], "parent_points_world"),
    ):
        if points.dtype != np.float32 or points.shape != (count, 3) or not np.isfinite(points).all():
            raise ValueError(f"{name} must be finite float32 [{count}, 3]")
    for indices, count, name in (
        (action_indices, len(action), "action_indices"),
        (anchor_indices, len(anchor), "anchor_indices"),
    ):
        if indices.dtype != np.int64 or indices.shape != (count,) or len(np.unique(indices)) != count:
            raise ValueError(f"{name} must be unique int64 [{count}]")
    if source_pose.dtype != np.float64 or source_pose.shape != (4, 4) or not np.isfinite(source_pose).all():
        raise ValueError("source_world_from_object must be finite float64 [4, 4]")
    rotation = source_pose[:3, :3]
    if (
        not np.array_equal(source_pose[3], np.asarray([0.0, 0.0, 0.0, 1.0]))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0)
    ):
        raise ValueError("source_world_from_object must be a proper SE(3)")
    scale32 = np.float32(scale_factor)
    action_scaled = np.ascontiguousarray(action * scale32, dtype=np.float32)
    anchor_scaled = np.ascontiguousarray(anchor * scale32, dtype=np.float32)
    center_scaled = np.concatenate((action_scaled, anchor_scaled), axis=0).mean(
        axis=0, dtype=np.float32
    )
    action_model = np.ascontiguousarray(action_scaled - center_scaled, dtype=np.float32)
    anchor_model = np.ascontiguousarray(anchor_scaled - center_scaled, dtype=np.float32)
    center_world = np.asarray(center_scaled, dtype=np.float64) / float(scale_factor)
    return FMObservation(
        sample=sample,
        action_model=action_model,
        anchor_model=anchor_model,
        source_world_from_object=np.array(source_pose, copy=True),
        model_center_world=center_world,
        scale_factor=float(scale_factor),
        model_input_sha256={
            "pc_action_joint_centered_scaled": sha256_array(action_model),
            "pc_anchor_joint_centered_scaled": sha256_array(anchor_model),
        },
    )


__all__ = [
    "EXPECTED_POINT_COUNTS",
    "FMObservation",
    "INFERENCE_PROFILE",
    "NativeExportIdentity",
    "NativeInferenceIndex",
    "NativeInferenceSample",
    "SPLITS",
    "TRAINING_PROFILE",
    "load_fm_observation",
    "load_native_export_identity",
    "load_native_inference_index",
    "sha256_array",
    "sha256_file",
]
