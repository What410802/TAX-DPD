"""Render a validated N12 one-sample overfit report without rerunning TAX-DPD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from non_rigid.utils.placegen_artifact import sha256_file

REPORT_SCHEMA = "placegen.taxdpd-one-sample-overfit-report/0.1"
SUMMARY_SCHEMA = "placegen.taxdpd-overfit-visualization-summary/0.1"
TRAINING_NPZ_FIELDS = frozenset(
    {
        "multi_obj_final_obj_pose",
        "multi_obj_final_pcd",
        "multi_obj_names",
        "multi_obj_start_obj_pose",
        "multi_obj_start_pcd",
        "placegen_scene_center_world",
    }
)
REPORT_FIELDS = frozenset(
    {
        "candidate_count",
        "candidate_index",
        "diagnostic_actual_state",
        "diagnostic_flags",
        "device",
        "diagnostic_scope",
        "final_model_state_sha256",
        "final_sampled_ordered_rmse_m",
        "finite",
        "fixed_loss_probes",
        "gate",
        "gradient_norm",
        "initial_model_state_sha256",
        "input_npz",
        "input_npz_sha256",
        "learned_sigma_loss_override",
        "loss",
        "manifest",
        "manifest_sha256",
        "model",
        "model_input_sha256",
        "noop_ordered_rmse_m",
        "parameter_changed",
        "peak_vram_mib",
        "predicted_kabsch_fit_rmse_m",
        "predicted_world_from_object",
        "quality_claim",
        "rotation_noise_override",
        "sample_id",
        "sample_seed",
        "sampled_coordinate_boundaries",
        "sampled_point_range",
        "sampling_duration_seconds",
        "sampling_only_after_final_step",
        "schema",
        "seed",
        "steps",
        "supervision_tensor_sha256",
        "target_decomposition",
        "target_kabsch_fit_rmse_m",
        "target_world_from_object",
        "timestep_schedule",
        "total_duration_seconds",
        "training_duration_seconds",
        "training_npz",
        "training_npz_sha256",
        "world_rotation_error_deg",
        "world_translation_error_mm",
    }
)
OPTIONAL_REPORT_FIELDS = frozenset(
    {
        "diagnostic_actual_state",
        "diagnostic_flags",
        "learned_sigma_loss_override",
        "rotation_noise_override",
    }
)
REQUIRED_REPORT_FIELDS = REPORT_FIELDS.difference(OPTIONAL_REPORT_FIELDS)
DIAGNOSTIC_FLAG_FIELDS = frozenset({"disable_rotation_noise", "rescale_learned_sigmas"})
DIAGNOSTIC_ACTUAL_STATE_FIELDS = frozenset(
    {
        "config_learn_sigma",
        "diffusion_loss_type",
        "diffusion_model_var_type",
        "diffusion_rotation_noise_scale",
        "network_learn_sigma",
    }
)
ROTATION_NOISE_OVERRIDE_FIELDS = frozenset(
    {
        "changed",
        "enabled",
        "rotation_noise_scale_after",
        "rotation_noise_scale_before",
    }
)
COLOR_SEMANTICS = {
    "rack_parent": "#5B6573 (gray)",
    "source_child": "#1976D2 (blue)",
    "target_child": "#2E8B57 (green)",
    "predicted_child": "#D1495B (red)",
}
TARGET_ALIGNMENT_ATOL_M = 2.0e-5
METRIC_ATOL = 1.0e-6


@dataclass(frozen=True)
class VisualizationData:
    """Validated world-frame point clouds and their provenance."""

    sample_id: str
    report_path: Path
    report_sha256: str
    training_npz: Path
    training_npz_sha256: str
    rack_parent_world: np.ndarray
    source_child_world: np.ndarray
    target_child_world: np.ndarray
    predicted_child_world: np.ndarray
    target_pose: np.ndarray
    predicted_pose: np.ndarray
    translation_error_mm: float
    rotation_error_deg: float
    sampled_ordered_rmse_m: float


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


def _regular_file(path: Path | str, name: str) -> Path:
    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise ValueError(f"{name} cannot be a symlink: {supplied}")
    if not supplied.is_file():
        raise FileNotFoundError(f"{name} does not exist: {supplied}")
    return supplied.resolve(strict=True)


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _safe_sample_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value[0].isalnum()
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in value
        )
    ):
        raise ValueError("report.sample_id must be a safe non-empty identifier")
    return value


def _finite_float(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _finite_nonnegative_or_false(value: Any, name: str) -> bool | float:
    """Validate a diffusion scale, preserving the config's native ``False``."""

    if value is False:
        return False
    if isinstance(value, bool):
        raise TypeError(f"{name} must be False or a finite non-negative number")
    return _finite_float(value, name, nonnegative=True)


def _scale_float(value: bool | float) -> float:
    """Return a numeric value for comparing a scale that may be ``False``."""

    return 0.0 if value is False else float(value)


def _se3(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite [4, 4] matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-9, rtol=0.0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6, rtol=0.0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6, rtol=0.0)
    ):
        raise ValueError(f"{name} must be a proper SE(3) matrix")
    return np.array(matrix, dtype=np.float64, copy=True, order="C")


def _pose_vector_to_se3(value: Any, name: str) -> np.ndarray:
    pose = np.asarray(value)
    if (
        pose.dtype != np.dtype(np.float64)
        or pose.shape != (7,)
        or not np.isfinite(pose).all()
    ):
        raise ValueError(f"{name} must be finite float64 [x,y,z,qx,qy,qz,qw]")
    quaternion = pose[3:]
    norm = float(np.linalg.norm(quaternion))
    if not np.isclose(norm, 1.0, atol=1.0e-8, rtol=0.0):
        raise ValueError(f"{name} quaternion must have unit norm")
    x, y, z, w = quaternion / norm
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = pose[:3]
    return result


def _object_mapping(value: np.ndarray, name: str) -> Mapping[str, Any]:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype(object):
        raise ValueError(f"{name} must be a scalar object mapping")
    mapping = _mapping(array.item(), name)
    _exact_fields(mapping, frozenset({"parent", "child"}), name)
    return mapping


def _points(value: Any, name: str) -> np.ndarray:
    points = np.asarray(value)
    if (
        points.dtype != np.dtype(np.float32)
        or points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) < 4
        or not np.isfinite(points).all()
    ):
        raise ValueError(f"{name} must be finite float32 [N, 3] with N >= 4")
    return np.array(points, dtype=np.float64, copy=True, order="C")


def _transform_object_points(
    object_points: np.ndarray, world_from_object: np.ndarray
) -> np.ndarray:
    """Apply column-vector SE(3) semantics to row-major object points."""

    return object_points @ world_from_object[:3, :3].T + world_from_object[:3, 3]


def _object_points_from_world(
    world_points: np.ndarray, world_from_object: np.ndarray
) -> np.ndarray:
    """Invert column-vector SE(3) semantics for row-major world points."""

    return (world_points - world_from_object[:3, 3]) @ world_from_object[:3, :3]


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3] @ second[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _load_report(path: Path | str) -> tuple[Path, Mapping[str, Any]]:
    report_path = _regular_file(path, "diagnostic report")
    try:
        report = _mapping(
            json.loads(report_path.read_text(encoding="utf-8")), "diagnostic report"
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("diagnostic report must be valid UTF-8 JSON") from error
    # Early 0.1 reports predate the optional learned-sigma and rotation-noise
    # diagnostics.  Keep the top-level schema closed while accepting reports
    # with or without those explicitly listed fields; all other additions
    # remain errors.
    report_fields = frozenset(report)
    missing = REQUIRED_REPORT_FIELDS.difference(report_fields)
    unexpected = report_fields.difference(REPORT_FIELDS)
    if missing or unexpected:
        raise ValueError(
            "diagnostic report fields must match the supported 0.1 revisions; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if "learned_sigma_loss_override" in report:
        override = _mapping(
            report["learned_sigma_loss_override"],
            "report.learned_sigma_loss_override",
        )
        _exact_fields(
            override,
            frozenset({"enabled", "loss_type_before", "loss_type_after", "vb_scale"}),
            "report.learned_sigma_loss_override",
        )
        if not isinstance(override["enabled"], bool):
            raise TypeError(
                "report.learned_sigma_loss_override.enabled must be boolean"
            )
        for key in ("loss_type_before", "loss_type_after"):
            if not isinstance(override[key], str) or not override[key]:
                raise ValueError(
                    "report.learned_sigma_loss_override loss types must be strings"
                )
        _finite_float(
            override["vb_scale"],
            "report.learned_sigma_loss_override.vb_scale",
            nonnegative=True,
        )
    if "diagnostic_flags" in report:
        flags = _mapping(report["diagnostic_flags"], "report.diagnostic_flags")
        _exact_fields(flags, DIAGNOSTIC_FLAG_FIELDS, "report.diagnostic_flags")
        for key in DIAGNOSTIC_FLAG_FIELDS:
            if not isinstance(flags[key], bool):
                raise TypeError(f"report.diagnostic_flags.{key} must be boolean")
        if "learned_sigma_loss_override" in report:
            learned_override = _mapping(
                report["learned_sigma_loss_override"],
                "report.learned_sigma_loss_override",
            )
            if flags["rescale_learned_sigmas"] != learned_override["enabled"]:
                raise ValueError(
                    "report diagnostic_flags.rescale_learned_sigmas disagrees with "
                    "learned_sigma_loss_override.enabled"
                )
    if "diagnostic_actual_state" in report:
        state = _mapping(
            report["diagnostic_actual_state"],
            "report.diagnostic_actual_state",
        )
        _exact_fields(
            state,
            DIAGNOSTIC_ACTUAL_STATE_FIELDS,
            "report.diagnostic_actual_state",
        )
        for key in ("config_learn_sigma", "network_learn_sigma"):
            if not isinstance(state[key], bool):
                raise TypeError(f"report.diagnostic_actual_state.{key} must be boolean")
        if state["config_learn_sigma"] != state["network_learn_sigma"]:
            raise ValueError(
                "report.diagnostic_actual_state config/network learn_sigma values "
                "disagree"
            )
        for key in ("diffusion_loss_type", "diffusion_model_var_type"):
            if not isinstance(state[key], str) or not state[key]:
                raise ValueError(
                    f"report.diagnostic_actual_state.{key} must be a non-empty string"
                )
        actual_scale = _finite_nonnegative_or_false(
            state["diffusion_rotation_noise_scale"],
            "report.diagnostic_actual_state.diffusion_rotation_noise_scale",
        )
        if "diagnostic_flags" in report:
            flags = report["diagnostic_flags"]
            if flags["disable_rotation_noise"] and _scale_float(actual_scale) != 0.0:
                raise ValueError(
                    "report diagnostic flags claim rotation noise is disabled, "
                    "but diagnostic_actual_state reports a non-zero scale"
                )
        if "learned_sigma_loss_override" in report:
            learned_override = report["learned_sigma_loss_override"]
            if state["diffusion_loss_type"] != learned_override["loss_type_after"]:
                raise ValueError(
                    "report diagnostic_actual_state loss type disagrees with "
                    "learned_sigma_loss_override"
                )
    if "rotation_noise_override" in report:
        override = _mapping(
            report["rotation_noise_override"],
            "report.rotation_noise_override",
        )
        _exact_fields(
            override,
            ROTATION_NOISE_OVERRIDE_FIELDS,
            "report.rotation_noise_override",
        )
        for key in ("enabled", "changed"):
            if not isinstance(override[key], bool):
                raise TypeError(f"report.rotation_noise_override.{key} must be boolean")
        before = _finite_nonnegative_or_false(
            override["rotation_noise_scale_before"],
            "report.rotation_noise_override.rotation_noise_scale_before",
        )
        after = _finite_nonnegative_or_false(
            override["rotation_noise_scale_after"],
            "report.rotation_noise_override.rotation_noise_scale_after",
        )
        expected_changed = bool(override["enabled"] and _scale_float(before) != 0.0)
        if override["changed"] != expected_changed:
            raise ValueError(
                "report.rotation_noise_override.changed disagrees with enabled "
                "and the before scale"
            )
        if override["enabled"] and _scale_float(after) != 0.0:
            raise ValueError(
                "report.rotation_noise_override must end at zero when enabled"
            )
        if not override["enabled"] and _scale_float(before) != _scale_float(after):
            raise ValueError(
                "report.rotation_noise_override changed a scale while disabled"
            )
        if "diagnostic_flags" in report:
            flags = report["diagnostic_flags"]
            if flags["disable_rotation_noise"] != override["enabled"]:
                raise ValueError(
                    "report diagnostic_flags.disable_rotation_noise disagrees with "
                    "rotation_noise_override.enabled"
                )
        if "diagnostic_actual_state" in report:
            state = report["diagnostic_actual_state"]
            actual = _finite_nonnegative_or_false(
                state["diffusion_rotation_noise_scale"],
                "report.diagnostic_actual_state.diffusion_rotation_noise_scale",
            )
            if _scale_float(actual) != _scale_float(after):
                raise ValueError(
                    "report diagnostic_actual_state scale disagrees with "
                    "rotation_noise_override"
                )
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(
            f"unsupported diagnostic report schema: {report.get('schema')!r}"
        )
    if report.get("quality_claim") is not False:
        raise ValueError("diagnostic report must not make a quality claim")
    if report.get("diagnostic_scope") != "single-export-overfit-not-generalization":
        raise ValueError(
            "diagnostic report is not the bounded one-export overfit diagnostic"
        )
    if (
        report.get("candidate_index") != 0
        or report.get("candidate_count") != 1
        or report.get("sampling_only_after_final_step") is not True
    ):
        raise ValueError("diagnostic report candidate semantics are unsupported")
    if report.get("finite") is not True:
        raise ValueError("diagnostic report marks its sampled result non-finite")
    _safe_sample_id(report.get("sample_id"))
    _sha256(report.get("training_npz_sha256"), "report.training_npz_sha256")
    _se3(report.get("predicted_world_from_object"), "predicted_world_from_object")
    _se3(report.get("target_world_from_object"), "target_world_from_object")
    for field in (
        "final_sampled_ordered_rmse_m",
        "world_translation_error_mm",
        "world_rotation_error_deg",
    ):
        _finite_float(report.get(field), f"report.{field}", nonnegative=True)
    return report_path, report


def load_visualization_data(
    report_path: Path | str,
    *,
    training_npz_override: Path | str | None = None,
) -> VisualizationData:
    """Validate a diagnostic report and reconstruct all four world-frame clouds."""

    resolved_report, report = _load_report(report_path)
    sample_id = _safe_sample_id(report.get("sample_id"))
    declared_training = report.get("training_npz")
    if not isinstance(declared_training, str) or not declared_training:
        raise ValueError("report.training_npz must be a non-empty path string")
    selected_training = (
        declared_training if training_npz_override is None else training_npz_override
    )
    training_path = _regular_file(selected_training, "training NPZ")
    if training_path.name != f"{sample_id}.npz":
        raise ValueError("training NPZ basename does not match report.sample_id")
    expected_sha256 = _sha256(
        report.get("training_npz_sha256"), "report.training_npz_sha256"
    )
    actual_sha256 = sha256_file(training_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("training NPZ SHA-256 does not match the diagnostic report")

    # The RPDiff-compatible training format contains object arrays.  Hash it first,
    # then open only the six-field, report-bound artifact used by this diagnostic.
    with np.load(training_path, allow_pickle=True) as data:
        fields = frozenset(data.files)
        if fields != TRAINING_NPZ_FIELDS:
            missing = sorted(TRAINING_NPZ_FIELDS.difference(fields))
            unexpected = sorted(fields.difference(TRAINING_NPZ_FIELDS))
            raise ValueError(
                "training NPZ fields must match schema exactly; "
                f"missing={missing}, unexpected={unexpected}"
            )
        start_points = _object_mapping(data["multi_obj_start_pcd"], "start points")
        final_points = _object_mapping(data["multi_obj_final_pcd"], "final points")
        start_poses = _object_mapping(data["multi_obj_start_obj_pose"], "start poses")
        final_poses = _object_mapping(data["multi_obj_final_obj_pose"], "final poses")
        names = _object_mapping(data["multi_obj_names"], "object names")
        if not all(
            isinstance(names[key], str) and names[key] for key in ("parent", "child")
        ):
            raise ValueError("training NPZ object names must be non-empty strings")
        source_child = _points(start_points["child"], "start child")
        rack_parent = _points(start_points["parent"], "start parent")
        final_child = _points(final_points["child"], "final child")
        final_parent = _points(final_points["parent"], "final parent")
        source_pose = _pose_vector_to_se3(start_poses["child"], "start child pose")
        target_training_pose = _pose_vector_to_se3(
            final_poses["child"], "final child pose"
        )
        start_parent_pose = _pose_vector_to_se3(
            start_poses["parent"], "start parent pose"
        )
        final_parent_pose = _pose_vector_to_se3(
            final_poses["parent"], "final parent pose"
        )
        center = np.asarray(data["placegen_scene_center_world"])
        if (
            center.dtype != np.dtype(np.float32)
            or center.shape != (3,)
            or not np.isfinite(center).all()
        ):
            raise ValueError("training NPZ scene center must be finite float32 [3]")

    if (
        source_child.shape != final_child.shape
        or rack_parent.shape != final_parent.shape
    ):
        raise ValueError("training NPZ start/final point counts must match")
    if not np.array_equal(rack_parent, final_parent):
        raise ValueError("training NPZ parent point clouds must remain fixed")
    if not np.allclose(start_parent_pose, final_parent_pose, atol=1.0e-9, rtol=0.0):
        raise ValueError("training NPZ parent pose must remain fixed")

    predicted_pose = _se3(
        report.get("predicted_world_from_object"), "predicted_world_from_object"
    )
    target_pose = _se3(
        report.get("target_world_from_object"), "target_world_from_object"
    )
    if not np.allclose(target_pose, target_training_pose, atol=1.0e-5, rtol=0.0):
        raise ValueError(
            "report target pose does not match the training NPZ final pose"
        )
    object_points = _object_points_from_world(source_child, source_pose)
    source_round_trip = _transform_object_points(object_points, source_pose)
    target_child = _transform_object_points(object_points, target_pose)
    predicted_child = _transform_object_points(object_points, predicted_pose)
    if not np.allclose(source_round_trip, source_child, atol=1.0e-9, rtol=0.0):
        raise RuntimeError(
            "source object/world round trip violated the pose convention"
        )
    target_max_error = float(np.max(np.linalg.norm(target_child - final_child, axis=1)))
    if target_max_error > TARGET_ALIGNMENT_ATOL_M:
        raise ValueError(
            "target reconstruction disagrees with final child points: "
            f"maximum error {target_max_error:.9f} m"
        )

    translation_error_mm = float(
        np.linalg.norm(predicted_pose[:3, 3] - target_pose[:3, 3]) * 1000.0
    )
    rotation_error_deg = _rotation_error_deg(predicted_pose, target_pose)
    reported_translation = _finite_float(
        report.get("world_translation_error_mm"),
        "report.world_translation_error_mm",
        nonnegative=True,
    )
    reported_rotation = _finite_float(
        report.get("world_rotation_error_deg"),
        "report.world_rotation_error_deg",
        nonnegative=True,
    )
    if not math.isclose(
        translation_error_mm, reported_translation, abs_tol=METRIC_ATOL
    ):
        raise ValueError(
            "reported translation error disagrees with the two world poses"
        )
    if not math.isclose(rotation_error_deg, reported_rotation, abs_tol=METRIC_ATOL):
        raise ValueError("reported rotation error disagrees with the two world poses")

    arrays = (
        rack_parent,
        source_child,
        target_child,
        predicted_child,
        target_pose,
        predicted_pose,
    )
    for array in arrays:
        array.setflags(write=False)
    return VisualizationData(
        sample_id=sample_id,
        report_path=resolved_report,
        report_sha256=sha256_file(resolved_report),
        training_npz=training_path,
        training_npz_sha256=actual_sha256,
        rack_parent_world=rack_parent,
        source_child_world=source_child,
        target_child_world=target_child,
        predicted_child_world=predicted_child,
        target_pose=target_pose,
        predicted_pose=predicted_pose,
        translation_error_mm=translation_error_mm,
        rotation_error_deg=rotation_error_deg,
        sampled_ordered_rmse_m=_finite_float(
            report.get("final_sampled_ordered_rmse_m"),
            "report.final_sampled_ordered_rmse_m",
            nonnegative=True,
        ),
    )


def render_html(data: VisualizationData, *, show: bool = False) -> str:
    """Build one self-contained Plotly document using rpad's point-cloud primitive."""

    import plotly.graph_objects as go
    import rpad.visualize_3d.plots as vpl

    figure = go.Figure()
    clouds = (
        ("Rack parent", data.rack_parent_world, "#5B6573", 2),
        ("Source child", data.source_child_world, "#1976D2", 3),
        ("Target child", data.target_child_world, "#2E8B57", 3),
        ("Predicted child", data.predicted_child_world, "#D1495B", 3),
    )
    for name, points, color, size in clouds:
        figure.add_trace(
            vpl.pointcloud(
                points,
                downsample=1,
                colors=[color] * len(points),
                scene="scene",
                name=name,
                size=size,
            )
        )
    figure.update_layout(
        title=(
            f"{data.sample_id}: TAX-DPD candidate 0 | "
            f"translation {data.translation_error_mm:.3f} mm | "
            f"rotation {data.rotation_error_deg:.3f} deg"
        ),
        scene={
            "aspectmode": "data",
            "xaxis_title": "world x (m)",
            "yaxis_title": "world y (m)",
            "zaxis_title": "world z (m)",
        },
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 0, "r": 0, "b": 0, "t": 48},
        showlegend=True,
    )
    if show:
        figure.show()
    return figure.to_html(full_html=True, include_plotlyjs=True)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(content)
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


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def run(
    args: argparse.Namespace,
    *,
    html_renderer: Callable[..., str] = render_html,
) -> dict[str, Any]:
    output = Path(args.output_html).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite HTML: {output}")
    if output.suffix.lower() != ".html":
        raise ValueError("--output-html must end in .html")
    summary_path: Path | None = None
    if args.summary is not None:
        summary_path = Path(args.summary).expanduser().resolve()
        if summary_path.exists() or summary_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite summary: {summary_path}")
        if summary_path == output:
            raise ValueError("summary path must differ from HTML path")

    data = load_visualization_data(args.report, training_npz_override=args.training_npz)
    html = html_renderer(data, show=args.show)
    if not isinstance(html, str) or "<html" not in html.lower():
        raise RuntimeError("renderer did not return a full HTML document")
    html_sha256 = _sha256_text(html)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "sample_id": data.sample_id,
        "frame": "world-meters",
        "coordinate_convention": (
            "object points = inverse(source_world_from_object) * start child; "
            "target/predicted world points = world_from_object * object points"
        ),
        "report": str(data.report_path),
        "report_sha256": data.report_sha256,
        "training_npz": str(data.training_npz),
        "training_npz_sha256": data.training_npz_sha256,
        "output_html": str(output),
        "output_html_sha256": html_sha256,
        "self_contained_html": True,
        "point_counts": {
            "rack_parent": len(data.rack_parent_world),
            "source_child": len(data.source_child_world),
            "target_child": len(data.target_child_world),
            "predicted_child": len(data.predicted_child_world),
        },
        "world_translation_error_mm": data.translation_error_mm,
        "world_rotation_error_deg": data.rotation_error_deg,
        "final_sampled_ordered_rmse_m": data.sampled_ordered_rmse_m,
        "color_semantics": COLOR_SEMANTICS,
        "model_rerun": False,
        "quality_claim": False,
    }
    _atomic_text(output, html)
    if sha256_file(output) != html_sha256:
        raise RuntimeError("published HTML SHA-256 differs from rendered content")
    if summary_path is not None:
        _atomic_text(summary_path, _json_text(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize one validated N12 overfit report without model inference."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--training-npz",
        type=Path,
        help="Relocated copy of report.training_npz; SHA and sample ID must still match.",
    )
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
