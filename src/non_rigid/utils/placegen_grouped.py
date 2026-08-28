"""Strict grouped PlaceGen dataset and held-out inference contracts.

The training manifest is a supervision-bearing trust boundary.  Deployment is
deliberately separate: :func:`load_grouped_inference_manifest` only accepts the
target-free ``inference/manifest.json`` index emitted by PlaceGen N3.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from non_rigid.utils.placegen_artifact import (
    MODEL_INPUT_PROFILE,
    OBSERVATION_FIELDS,
    PlaceGenObservationArtifact,
    model_input_sha256,
    sha256_array,
    sha256_file,
)

GROUPED_TRAINING_PROFILE = "placegen.taxdpd-rpdiff-grouped/0.1"
GROUPED_INFERENCE_PROFILE = "placegen.taxdpd-inference-index/0.1"
SPLIT_ASSIGNMENT_PROFILE = "placegen.pose-supervision-split-assignment/0.1"
QUALITY_SCOPE = "fixed-asset-fixed-slot-heldout-pose"
MODEL_CAPABILITY = "TAX-DPD-reconstructed-fixed-frame-w/o-GMM"
SPLITS = ("train", "validation", "test")
DEFAULT_GROUP_COUNTS = {"train": 72, "validation": 12, "test": 12}

_TRAINING_TOP_FIELDS = frozenset(
    {
        "profile",
        "task_name",
        "task_type",
        "sample_count",
        "group_count",
        "split_counts",
        "point_counts",
        "evaluation_valid",
        "correlated_split",
        "quality_claim",
        "quality_scope",
        "split_assignment",
        "oracle_kabsch_round_trip",
        "samples",
    }
)
_TRAINING_SAMPLE_FIELDS = frozenset(
    {
        "sample_id",
        "split",
        "split_group_id",
        "descriptor",
        "physical_setup",
        "training_npz",
        "inference_npz",
        "tensor_sha256",
        "oracle_kabsch_round_trip",
    }
)
_INFERENCE_TOP_FIELDS = frozenset(
    {
        "profile",
        "quality_scope",
        "evaluation_valid",
        "correlated_split",
        "ground_truth_free",
        "sample_count",
        "point_counts",
        "split_assignment",
        "samples",
    }
)
_INFERENCE_SAMPLE_FIELDS = frozenset(
    {
        "sample_id",
        "split",
        "split_group_id",
        "descriptor_sha256",
        "input_npz",
        "input_tensor_sha256",
    }
)
_TRAINING_NPZ_FIELDS = frozenset(
    {
        "multi_obj_final_obj_pose",
        "multi_obj_final_pcd",
        "multi_obj_names",
        "multi_obj_start_obj_pose",
        "multi_obj_start_pcd",
        "placegen_scene_center_world",
    }
)


def target_free_report_semantics() -> dict[str, str | bool]:
    """Return the conservative semantics shared by grouped prediction reports."""

    return {
        "ground_truth_free": True,
        "ground_truth_free_semantics": "target-and-goal-supervision-not-read",
        "target_supervision_free": True,
        "source_pose_provenance": "provenance-not-encoded-in-inference-manifest",
        "source_pose_is_deployment_observation": False,
        "deployment_input_valid": False,
    }


def canonical_json_sha256(value: Any) -> str:
    """Hash one canonical, finite JSON value."""

    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    fields = frozenset(value)
    if fields != expected:
        missing = sorted(expected.difference(fields))
        unexpected = sorted(fields.difference(expected))
        raise ValueError(
            f"{name} fields must match schema exactly; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _safe_identifier(value: Any, name: str) -> str:
    result = _nonempty_string(value, name)
    if not result[0].isalnum() or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in result
    ):
        raise ValueError(f"{name} must be a safe identifier")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _regular_json(path: Path | str, name: str) -> tuple[Path, Mapping[str, Any]]:
    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise ValueError(f"{name} cannot be a symlink: {supplied}")
    if not supplied.is_file():
        raise FileNotFoundError(f"{name} does not exist: {supplied}")
    resolved = supplied.resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    return resolved, _mapping(payload, name)


def _relative_regular_file(
    root: Path,
    value: Mapping[str, Any],
    name: str,
) -> tuple[Path, str]:
    _exact_fields(value, frozenset({"path", "sha256"}), name)
    declared_path = _nonempty_string(value.get("path"), f"{name}.path")
    path = root / declared_path
    if path.is_symlink():
        raise ValueError(f"{name} cannot reference a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name}.path escapes its manifest root") from error
    expected_sha256 = _sha256(value.get("sha256"), f"{name}.sha256")
    if sha256_file(resolved) != expected_sha256:
        raise ValueError(f"{name} SHA-256 does not match its manifest")
    return resolved, expected_sha256


def _relative_declared_file(
    root: Path,
    value: Mapping[str, Any],
    name: str,
) -> tuple[Path, str]:
    """Validate a file declaration lexically without touching the file."""

    _exact_fields(value, frozenset({"path", "sha256"}), name)
    declared_path = _nonempty_string(value.get("path"), f"{name}.path")
    if declared_path.startswith("/") or "\\" in declared_path:
        raise ValueError(f"{name}.path must be a relative POSIX path")
    relative = PurePosixPath(declared_path)
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{name}.path contains an unsafe component")
    if relative.as_posix() != declared_path:
        raise ValueError(f"{name}.path must use canonical POSIX spelling")
    expected_sha256 = _sha256(value.get("sha256"), f"{name}.sha256")
    return root.joinpath(*relative.parts), expected_sha256


def _supervision_split_policy(value: tuple[str, ...]) -> tuple[str, ...]:
    if not value or len(set(value)) != len(value):
        raise ValueError("supervision_splits must be non-empty and unique")
    unsupported = sorted(set(value).difference(SPLITS))
    if unsupported:
        raise ValueError(
            f"supervision_splits contains unsupported splits: {unsupported}"
        )
    return tuple(split for split in SPLITS if split in value)


def _point_counts(value: Any, name: str) -> dict[str, int]:
    counts = _mapping(value, name)
    _exact_fields(counts, frozenset({"action", "anchor"}), name)
    return {
        "action": _positive_int(counts.get("action"), f"{name}.action"),
        "anchor": _positive_int(counts.get("anchor"), f"{name}.anchor"),
    }


def _split_counts(value: Any, name: str) -> dict[str, int]:
    counts = _mapping(value, name)
    _exact_fields(counts, frozenset(SPLITS), name)
    return {
        split: _positive_int(counts.get(split), f"{name}.{split}") for split in SPLITS
    }


def _split_assignment(
    value: Any,
    name: str,
    *,
    training: bool,
) -> tuple[str, str]:
    assignment = _mapping(value, name)
    expected = {
        "profile",
        "manifest_sha256",
        "native_dataset_sha256",
    }
    if training:
        expected.update(
            {
                "identity_ledger",
                "identity_ledger_sha256",
                "assignment_algorithm",
                "seed",
            }
        )
    _exact_fields(assignment, frozenset(expected), name)
    if assignment.get("profile") != SPLIT_ASSIGNMENT_PROFILE:
        raise ValueError(f"{name}.profile is not the supported assignment schema")
    manifest_sha256 = _sha256(
        assignment.get("manifest_sha256"), f"{name}.manifest_sha256"
    )
    native_sha256 = _sha256(
        assignment.get("native_dataset_sha256"), f"{name}.native_dataset_sha256"
    )
    if training:
        ledger = _mapping(assignment.get("identity_ledger"), f"{name}.identity_ledger")
        expected_ledger_sha = _sha256(
            assignment.get("identity_ledger_sha256"),
            f"{name}.identity_ledger_sha256",
        )
        if canonical_json_sha256(ledger) != expected_ledger_sha:
            raise ValueError(f"{name}.identity_ledger SHA-256 does not match")
        _nonempty_string(
            assignment.get("assignment_algorithm"), f"{name}.assignment_algorithm"
        )
        _nonnegative_int(assignment.get("seed"), f"{name}.seed")
    return manifest_sha256, native_sha256


def _tensor_hashes(value: Any, name: str, *, supervision: bool) -> dict[str, str]:
    hashes = _mapping(value, name)
    expected = {"pc_action_centered", "pc_anchor_centered"}
    if supervision:
        expected.add("pc_goal_centered")
    _exact_fields(hashes, frozenset(expected), name)
    return {key: _sha256(hashes.get(key), f"{name}.{key}") for key in sorted(expected)}


def _object_mapping(array: np.ndarray, name: str) -> Mapping[str, Any]:
    value = np.asarray(array)
    if value.shape != () or value.dtype != np.dtype(object):
        raise ValueError(f"{name} must be a scalar object mapping")
    return _mapping(value.item(), name)


def _training_arrays(
    path: Path,
    *,
    point_counts: Mapping[str, int],
    expected_tensor_sha256: Mapping[str, str],
) -> None:
    # PlaceGen's RPDiff-compatible format necessarily contains object arrays.
    # The containing file is hashed before this trusted training-only load.
    with np.load(path, allow_pickle=True) as data:
        fields = frozenset(data.files)
        if fields != _TRAINING_NPZ_FIELDS:
            missing = sorted(_TRAINING_NPZ_FIELDS.difference(fields))
            unexpected = sorted(fields.difference(_TRAINING_NPZ_FIELDS))
            raise ValueError(
                "training NPZ fields must match schema exactly; "
                f"missing={missing}, unexpected={unexpected}"
            )
        start = _object_mapping(data["multi_obj_start_pcd"], "multi_obj_start_pcd")
        final = _object_mapping(data["multi_obj_final_pcd"], "multi_obj_final_pcd")
        _exact_fields(start, frozenset({"parent", "child"}), "multi_obj_start_pcd")
        _exact_fields(final, frozenset({"parent", "child"}), "multi_obj_final_pcd")
        action = np.asarray(start["child"])
        anchor = np.asarray(start["parent"])
        goal = np.asarray(final["child"])
        final_anchor = np.asarray(final["parent"])
        for value, expected_count, name in (
            (action, point_counts["action"], "start child"),
            (goal, point_counts["action"], "final child"),
            (anchor, point_counts["anchor"], "start parent"),
            (final_anchor, point_counts["anchor"], "final parent"),
        ):
            if (
                value.dtype != np.dtype(np.float32)
                or value.shape != (expected_count, 3)
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"training NPZ {name} must be finite float32 [N, 3]")
        if not np.array_equal(anchor, final_anchor):
            raise ValueError("training NPZ parent point clouds must be identical")
        center = np.asarray(data["placegen_scene_center_world"])
        if (
            center.dtype != np.dtype(np.float32)
            or center.shape != (3,)
            or not np.isfinite(center).all()
        ):
            raise ValueError("training NPZ scene center must be finite float32 [3]")
        actual_hashes = {
            "pc_action_centered": sha256_array(action - center),
            "pc_anchor_centered": sha256_array(anchor - center),
            "pc_goal_centered": sha256_array(goal - center),
        }
        if actual_hashes != dict(expected_tensor_sha256):
            raise ValueError(
                "training NPZ tensor SHA-256 values do not match its manifest"
            )


@dataclass(frozen=True)
class GroupedTrainingSample:
    """One validated supervision-bearing TAX-DPD sample."""

    sample_id: str
    split: str
    split_group_id: str
    descriptor_sha256: str
    training_npz: Path
    training_npz_sha256: str
    tensor_sha256: dict[str, str]


@dataclass(frozen=True)
class GroupedTrainingManifest:
    """Immutable grouped dataset identity and disjoint sample partitions."""

    manifest_path: Path
    manifest_sha256: str
    root: Path
    task_name: str
    task_type: str
    point_counts: dict[str, int]
    split_counts: dict[str, int]
    group_counts: dict[str, int]
    split_assignment_sha256: str
    native_dataset_sha256: str
    supervision_splits: tuple[str, ...]
    samples_by_split: dict[str, tuple[GroupedTrainingSample, ...]]

    @property
    def validation_sample_ids(self) -> tuple[str, ...]:
        return tuple(sample.sample_id for sample in self.samples_by_split["validation"])


def load_grouped_training_manifest(
    manifest_path: Path | str,
    *,
    expected_group_counts: Mapping[str, int] | None = None,
    supervision_splits: tuple[str, ...] = SPLITS,
) -> GroupedTrainingManifest:
    """Validate metadata and open supervision only for explicitly allowed splits.

    The default preserves the historical full-validation behavior.  Training must
    pass ``("train", "validation")`` so test supervision paths remain declarations:
    their schema, lexical canonical path, and SHA-256 text are checked, but the files
    are never stat'ed, hashed, or opened.
    """

    validated_supervision_splits = _supervision_split_policy(supervision_splits)
    validated_supervision_set = frozenset(validated_supervision_splits)
    path, manifest = _regular_json(manifest_path, "grouped training manifest")
    _exact_fields(manifest, _TRAINING_TOP_FIELDS, "grouped training manifest")
    if manifest.get("profile") != GROUPED_TRAINING_PROFILE:
        raise ValueError("unexpected grouped training manifest profile")
    if manifest.get("evaluation_valid") is not True:
        raise ValueError("grouped training manifest must be evaluation-valid")
    if manifest.get("correlated_split") is not False:
        raise ValueError("grouped training manifest must use disjoint splits")
    if manifest.get("quality_claim") is not False:
        raise ValueError("input export cannot make a quality claim")
    if manifest.get("quality_scope") != QUALITY_SCOPE:
        raise ValueError("unexpected grouped training quality scope")
    task_name = _safe_identifier(manifest.get("task_name"), "task_name")
    task_type = _safe_identifier(manifest.get("task_type"), "task_type")
    sample_count = _positive_int(manifest.get("sample_count"), "sample_count")
    declared_group_count = _positive_int(manifest.get("group_count"), "group_count")
    declared_split_counts = _split_counts(manifest.get("split_counts"), "split_counts")
    point_counts = _point_counts(manifest.get("point_counts"), "point_counts")
    assignment_sha, native_sha = _split_assignment(
        manifest.get("split_assignment"), "split_assignment", training=True
    )
    oracle = _mapping(
        manifest.get("oracle_kabsch_round_trip"), "oracle_kabsch_round_trip"
    )
    _exact_fields(
        oracle,
        frozenset(
            {
                "translation_limit_mm",
                "rotation_limit_deg",
                "maximum_translation_error_mm",
                "maximum_rotation_error_deg",
                "all_samples_passed",
            }
        ),
        "oracle_kabsch_round_trip",
    )
    if oracle.get("all_samples_passed") is not True:
        raise ValueError("all samples must pass the export Kabsch gate")
    translation_limit = _finite_float(
        oracle.get("translation_limit_mm"), "oracle translation_limit_mm"
    )
    rotation_limit = _finite_float(
        oracle.get("rotation_limit_deg"), "oracle rotation_limit_deg"
    )
    maximum_translation = _finite_float(
        oracle.get("maximum_translation_error_mm"),
        "oracle maximum_translation_error_mm",
    )
    maximum_rotation = _finite_float(
        oracle.get("maximum_rotation_error_deg"), "oracle maximum_rotation_error_deg"
    )
    if (
        translation_limit <= 0.0
        or rotation_limit <= 0.0
        or maximum_translation < 0.0
        or maximum_rotation < 0.0
        or maximum_translation >= translation_limit
        or maximum_rotation >= rotation_limit
    ):
        raise ValueError("grouped training oracle Kabsch summary is inconsistent")

    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != sample_count:
        raise ValueError("samples length does not match sample_count")
    root = path.parent.resolve()
    samples: dict[str, list[GroupedTrainingSample]] = {split: [] for split in SPLITS}
    seen_sample_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    for index, raw_sample in enumerate(raw_samples):
        name = f"samples[{index}]"
        sample = _mapping(raw_sample, name)
        _exact_fields(sample, _TRAINING_SAMPLE_FIELDS, name)
        sample_id = _safe_identifier(sample.get("sample_id"), f"{name}.sample_id")
        if sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        split = _nonempty_string(sample.get("split"), f"{name}.split")
        if split not in SPLITS:
            raise ValueError(f"{name}.split is unsupported")
        group_id = _nonempty_string(
            sample.get("split_group_id"), f"{name}.split_group_id"
        )
        previous_split = group_splits.setdefault(group_id, split)
        if previous_split != split:
            raise ValueError(f"split group {group_id!r} crosses dataset splits")
        descriptor = _mapping(sample.get("descriptor"), f"{name}.descriptor")
        _exact_fields(descriptor, frozenset({"path", "sha256"}), f"{name}.descriptor")
        _nonempty_string(descriptor.get("path"), f"{name}.descriptor.path")
        descriptor_sha = _sha256(descriptor.get("sha256"), f"{name}.descriptor.sha256")
        physical = _mapping(sample.get("physical_setup"), f"{name}.physical_setup")
        _exact_fields(
            physical,
            frozenset({"setup_id", "setup_sha256", "physical_state_sha256"}),
            f"{name}.physical_setup",
        )
        _nonempty_string(physical.get("setup_id"), f"{name}.physical_setup.setup_id")
        _sha256(physical.get("setup_sha256"), f"{name}.physical_setup.setup_sha256")
        _sha256(
            physical.get("physical_state_sha256"),
            f"{name}.physical_setup.physical_state_sha256",
        )
        training_record = _mapping(sample.get("training_npz"), f"{name}.training_npz")
        training_npz, training_sha = _relative_declared_file(
            root,
            training_record,
            f"{name}.training_npz",
        )
        expected_training = root / task_name / task_type / f"{sample_id}.npz"
        if training_npz != expected_training:
            raise ValueError(
                f"{name}.training_npz path is not the canonical RPDiff path"
            )
        if split in validated_supervision_set:
            validated_path, validated_sha = _relative_regular_file(
                root,
                training_record,
                f"{name}.training_npz",
            )
            if validated_path != training_npz or validated_sha != training_sha:
                raise RuntimeError(
                    f"{name}.training_npz declaration changed during validation"
                )
        inference_record = _mapping(
            sample.get("inference_npz"), f"{name}.inference_npz"
        )
        _exact_fields(
            inference_record,
            frozenset({"path", "sha256"}),
            f"{name}.inference_npz",
        )
        _nonempty_string(inference_record.get("path"), f"{name}.inference_npz.path")
        _sha256(inference_record.get("sha256"), f"{name}.inference_npz.sha256")
        tensor_hashes = _tensor_hashes(
            sample.get("tensor_sha256"), f"{name}.tensor_sha256", supervision=True
        )
        sample_oracle = _mapping(
            sample.get("oracle_kabsch_round_trip"), f"{name}.oracle_kabsch_round_trip"
        )
        _exact_fields(
            sample_oracle,
            frozenset({"translation_error_mm", "rotation_error_deg"}),
            f"{name}.oracle_kabsch_round_trip",
        )
        if (
            _finite_float(
                sample_oracle.get("translation_error_mm"),
                f"{name}.oracle translation_error_mm",
            )
            >= translation_limit
            or _finite_float(
                sample_oracle.get("rotation_error_deg"),
                f"{name}.oracle rotation_error_deg",
            )
            >= rotation_limit
        ):
            raise ValueError(f"{name} exceeds the ordered Kabsch export gate")
        if split in validated_supervision_set:
            _training_arrays(
                training_npz,
                point_counts=point_counts,
                expected_tensor_sha256=tensor_hashes,
            )
        samples[split].append(
            GroupedTrainingSample(
                sample_id=sample_id,
                split=split,
                split_group_id=group_id,
                descriptor_sha256=descriptor_sha,
                training_npz=training_npz,
                training_npz_sha256=training_sha,
                tensor_sha256=tensor_hashes,
            )
        )

    actual_split_counts = {split: len(samples[split]) for split in SPLITS}
    if actual_split_counts != declared_split_counts:
        raise ValueError("manifest split_counts do not match sample records")
    group_counts = {
        split: len({sample.split_group_id for sample in samples[split]})
        for split in SPLITS
    }
    if sum(group_counts.values()) != declared_group_count:
        raise ValueError("manifest group_count does not match unique split groups")
    expected_counts = dict(
        DEFAULT_GROUP_COUNTS if expected_group_counts is None else expected_group_counts
    )
    if frozenset(expected_counts) != frozenset(SPLITS):
        raise ValueError("expected_group_counts must define train/validation/test")
    expected_counts = {
        split: _positive_int(expected_counts[split], f"expected_group_counts.{split}")
        for split in SPLITS
    }
    if group_counts != expected_counts:
        raise ValueError(
            f"expected grouped split counts {expected_counts}, got {group_counts}"
        )

    split_root = root / task_name / task_type / "split_info"
    split_files = {
        "train": "train_split.txt",
        "validation": "train_val_split.txt",
        "test": "test_split.txt",
    }
    for split in SPLITS:
        split_path = split_root / split_files[split]
        if split_path.is_symlink() or not split_path.is_file():
            raise ValueError(f"{split} split file must be a regular file")
        actual_names = split_path.read_text(encoding="utf-8").splitlines()
        expected_names = [
            f"{sample.sample_id}.npz"
            for sample in sorted(samples[split], key=lambda x: x.sample_id)
        ]
        if actual_names != expected_names:
            raise ValueError(f"{split} split file differs from the grouped manifest")

    return GroupedTrainingManifest(
        manifest_path=path,
        manifest_sha256=sha256_file(path),
        root=root,
        task_name=task_name,
        task_type=task_type,
        point_counts=point_counts,
        split_counts=actual_split_counts,
        group_counts=group_counts,
        split_assignment_sha256=assignment_sha,
        native_dataset_sha256=native_sha,
        supervision_splits=validated_supervision_splits,
        samples_by_split={
            split: tuple(sorted(samples[split], key=lambda sample: sample.sample_id))
            for split in SPLITS
        },
    )


@dataclass(frozen=True)
class GroupedInferenceSample:
    """One target-free indexed observation."""

    sample_id: str
    split: str
    split_group_id: str
    descriptor_sha256: str
    input_npz: Path
    input_npz_sha256: str
    input_tensor_sha256: dict[str, str]


@dataclass(frozen=True)
class GroupedInferenceManifest:
    """Strict target-free inference index; no supervision fields are retained."""

    manifest_path: Path
    manifest_sha256: str
    root: Path
    point_counts: dict[str, int]
    split_assignment_sha256: str
    native_dataset_sha256: str
    samples_by_split: dict[str, tuple[GroupedInferenceSample, ...]]

    @property
    def samples_by_id(self) -> dict[str, GroupedInferenceSample]:
        return {
            sample.sample_id: sample
            for split in SPLITS
            for sample in self.samples_by_split[split]
        }


def load_grouped_inference_manifest(
    manifest_path: Path | str,
) -> GroupedInferenceManifest:
    """Accept only N3's ``inference/manifest.json`` target-free index."""

    supplied = Path(manifest_path).expanduser()
    if supplied.name != "manifest.json" or supplied.parent.name != "inference":
        raise ValueError("predictor requires the dedicated inference/manifest.json")
    path, manifest = _regular_json(supplied, "grouped inference manifest")
    _exact_fields(manifest, _INFERENCE_TOP_FIELDS, "grouped inference manifest")
    if manifest.get("profile") != GROUPED_INFERENCE_PROFILE:
        raise ValueError("unexpected grouped inference manifest profile")
    if manifest.get("quality_scope") != QUALITY_SCOPE:
        raise ValueError("unexpected grouped inference quality scope")
    if (
        manifest.get("evaluation_valid") is not True
        or manifest.get("correlated_split") is not False
    ):
        raise ValueError(
            "grouped inference index must describe a valid disjoint evaluation"
        )
    if manifest.get("ground_truth_free") is not True:
        raise ValueError("grouped inference index must be ground-truth-free")
    sample_count = _positive_int(manifest.get("sample_count"), "sample_count")
    point_counts = _point_counts(manifest.get("point_counts"), "point_counts")
    assignment_sha, native_sha = _split_assignment(
        manifest.get("split_assignment"), "split_assignment", training=False
    )
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != sample_count:
        raise ValueError("inference samples length does not match sample_count")
    root = path.parent.resolve()
    samples: dict[str, list[GroupedInferenceSample]] = {split: [] for split in SPLITS}
    seen: set[str] = set()
    group_splits: dict[str, str] = {}
    for index, raw_sample in enumerate(raw_samples):
        name = f"samples[{index}]"
        sample = _mapping(raw_sample, name)
        _exact_fields(sample, _INFERENCE_SAMPLE_FIELDS, name)
        sample_id = _safe_identifier(sample.get("sample_id"), f"{name}.sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate inference sample_id: {sample_id}")
        seen.add(sample_id)
        split = _nonempty_string(sample.get("split"), f"{name}.split")
        if split not in SPLITS:
            raise ValueError(f"{name}.split is unsupported")
        group_id = _nonempty_string(
            sample.get("split_group_id"), f"{name}.split_group_id"
        )
        if group_splits.setdefault(group_id, split) != split:
            raise ValueError(f"inference group {group_id!r} crosses dataset splits")
        descriptor_sha = _sha256(
            sample.get("descriptor_sha256"), f"{name}.descriptor_sha256"
        )
        input_record = _mapping(sample.get("input_npz"), f"{name}.input_npz")
        input_npz, input_sha = _relative_regular_file(
            root, input_record, f"{name}.input_npz"
        )
        expected_path = root / split / f"{sample_id}.npz"
        if input_npz != expected_path.resolve():
            raise ValueError(f"{name}.input_npz is not at its canonical split path")
        hashes = _tensor_hashes(
            sample.get("input_tensor_sha256"),
            f"{name}.input_tensor_sha256",
            supervision=False,
        )
        samples[split].append(
            GroupedInferenceSample(
                sample_id=sample_id,
                split=split,
                split_group_id=group_id,
                descriptor_sha256=descriptor_sha,
                input_npz=input_npz,
                input_npz_sha256=input_sha,
                input_tensor_sha256=hashes,
            )
        )
    if any(not samples[split] for split in SPLITS):
        raise ValueError(
            "inference index requires non-empty train/validation/test splits"
        )
    return GroupedInferenceManifest(
        manifest_path=path,
        manifest_sha256=sha256_file(path),
        root=root,
        point_counts=point_counts,
        split_assignment_sha256=assignment_sha,
        native_dataset_sha256=native_sha,
        samples_by_split={
            split: tuple(sorted(samples[split], key=lambda sample: sample.sample_id))
            for split in SPLITS
        },
    )


def _points(value: np.ndarray, expected_count: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.dtype != np.dtype(np.float32)
        or array.shape != (expected_count, 3)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be finite float32 [{expected_count}, 3]")
    return np.array(array, dtype=np.float32, copy=True, order="C")


def _indices(value: np.ndarray, expected_count: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.int64) or array.shape != (expected_count,):
        raise ValueError(f"{name} must be int64 [{expected_count}]")
    if np.any(array < 0) or len(np.unique(array)) != expected_count:
        raise ValueError(f"{name} must contain unique non-negative indices")
    return np.array(array, dtype=np.int64, copy=True, order="C")


def _source_pose(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value)
    if (
        matrix.dtype != np.dtype(np.float64)
        or matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("source_world_from_object must be finite float64 [4, 4]")
    rotation = matrix[:3, :3]
    if (
        not np.array_equal(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]))
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6, rtol=0.0)
    ):
        raise ValueError("source_world_from_object must be a proper SE(3) matrix")
    return np.array(matrix, dtype=np.float64, copy=True, order="C")


def load_grouped_observation(
    manifest: GroupedInferenceManifest,
    sample_id: str,
    *,
    required_split: str = "test",
) -> PlaceGenObservationArtifact:
    """Load one indexed observation without accessing a supervision manifest."""

    if required_split not in SPLITS:
        raise ValueError("required_split must be train, validation, or test")
    sample = manifest.samples_by_id.get(sample_id)
    if sample is None:
        raise KeyError(f"sample_id is absent from inference index: {sample_id}")
    if sample.split != required_split:
        raise ValueError(
            f"sample {sample_id!r} belongs to {sample.split}, not required {required_split}"
        )
    if sha256_file(sample.input_npz) != sample.input_npz_sha256:
        raise ValueError("inference NPZ SHA-256 changed after index validation")
    with np.load(sample.input_npz, allow_pickle=False) as data:
        fields = frozenset(data.files)
        if fields != OBSERVATION_FIELDS:
            missing = sorted(OBSERVATION_FIELDS.difference(fields))
            unexpected = sorted(fields.difference(OBSERVATION_FIELDS))
            raise ValueError(
                "TAX-DPD inference NPZ must contain exactly observation fields; "
                f"missing={missing}, unexpected={unexpected}"
            )
        child = _points(
            data["child_points_world"],
            manifest.point_counts["action"],
            "child_points_world",
        )
        parent = _points(
            data["parent_points_world"],
            manifest.point_counts["anchor"],
            "parent_points_world",
        )
        action_indices = _indices(
            data["action_indices"], manifest.point_counts["action"], "action_indices"
        )
        anchor_indices = _indices(
            data["anchor_indices"], manifest.point_counts["anchor"], "anchor_indices"
        )
        center_value = np.asarray(data["scene_center_world"])
        if (
            center_value.dtype != np.dtype(np.float64)
            or center_value.shape != (3,)
            or not np.isfinite(center_value).all()
        ):
            raise ValueError("scene_center_world must be finite float64 [3]")
        center = np.array(center_value, dtype=np.float64, copy=True, order="C")
        source_pose = _source_pose(data["source_world_from_object"])
    recomputed_center = (
        np.concatenate((child.astype(np.float64), parent.astype(np.float64)))
        .mean(axis=0)
        .astype(np.float32)
        .astype(np.float64)
    )
    if not np.array_equal(center, recomputed_center):
        raise ValueError(
            "scene_center_world does not exactly match observed point clouds"
        )
    center32 = center.astype(np.float32)
    action = np.ascontiguousarray(child - center32, dtype=np.float32)
    anchor = np.ascontiguousarray(parent - center32, dtype=np.float32)
    actual_hashes = {
        "pc_action_centered": sha256_array(action),
        "pc_anchor_centered": sha256_array(anchor),
    }
    if actual_hashes != sample.input_tensor_sha256:
        raise ValueError("inference tensor SHA-256 values do not match their index")
    for array in (
        child,
        parent,
        center,
        source_pose,
        action_indices,
        anchor_indices,
        action,
        anchor,
    ):
        array.setflags(write=False)
    return PlaceGenObservationArtifact(
        sample_id=sample.sample_id,
        descriptor_sha256=sample.descriptor_sha256,
        input_npz_sha256=sample.input_npz_sha256,
        child_points_world=child,
        parent_points_world=parent,
        scene_center_world=center,
        source_world_from_object=source_pose,
        action_indices=action_indices,
        anchor_indices=anchor_indices,
        pc_action_centered=action,
        pc_anchor_centered=anchor,
        tensor_sha256=actual_hashes,
        supervision_tensor_sha256=None,
        model_input_sha256=model_input_sha256(action, anchor),
    )


@dataclass(frozen=True)
class ValidationCheckpointDecision:
    """Best checkpoint identity selected only by validation candidate zero."""

    epoch: int
    validation_candidate0_ordered_rmse_m: float
    validation_sample_ids: tuple[str, ...]
    selection_split: str = "validation"
    candidate_index: int = 0


def select_validation_checkpoint(
    *,
    previous: ValidationCheckpointDecision | None,
    epoch: int,
    validation_candidate0_ordered_rmse_m: float,
    validation_sample_ids: tuple[str, ...],
    expected_validation_sample_ids: tuple[str, ...] | None = None,
) -> ValidationCheckpointDecision:
    """Choose the lower validation candidate-0 RMSE; test IDs fail closed."""

    epoch_value = _positive_int(epoch, "epoch")
    metric = _finite_float(
        validation_candidate0_ordered_rmse_m,
        "validation_candidate0_ordered_rmse_m",
    )
    if metric < 0.0:
        raise ValueError("validation candidate-0 RMSE cannot be negative")
    if not validation_sample_ids or len(set(validation_sample_ids)) != len(
        validation_sample_ids
    ):
        raise ValueError("validation sample IDs must be non-empty and unique")
    if expected_validation_sample_ids is not None and tuple(
        validation_sample_ids
    ) != tuple(expected_validation_sample_ids):
        raise ValueError(
            "validation sample IDs differ from the dataset validation split"
        )
    current = ValidationCheckpointDecision(
        epoch=epoch_value,
        validation_candidate0_ordered_rmse_m=metric,
        validation_sample_ids=tuple(validation_sample_ids),
    )
    if previous is None or metric < previous.validation_candidate0_ordered_rmse_m:
        return current
    return previous


__all__ = [
    "DEFAULT_GROUP_COUNTS",
    "GROUPED_INFERENCE_PROFILE",
    "GROUPED_TRAINING_PROFILE",
    "MODEL_CAPABILITY",
    "MODEL_INPUT_PROFILE",
    "QUALITY_SCOPE",
    "SPLITS",
    "GroupedInferenceManifest",
    "GroupedInferenceSample",
    "GroupedTrainingManifest",
    "GroupedTrainingSample",
    "ValidationCheckpointDecision",
    "canonical_json_sha256",
    "load_grouped_inference_manifest",
    "load_grouped_observation",
    "load_grouped_training_manifest",
    "select_validation_checkpoint",
    "sha256_file",
    "target_free_report_semantics",
]
