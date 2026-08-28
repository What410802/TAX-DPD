"""CPU-only contracts for the offline N12 point-cloud visualizer."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.placegen_visualize_overfit import (
    OPTIONAL_REPORT_FIELDS,
    REPORT_FIELDS,
    REPORT_SCHEMA,
    SUMMARY_SCHEMA,
    load_visualization_data,
    render_html,
    run,
)


def _transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (points @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def _pose(translation: tuple[float, float, float], angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result[:3, 3] = translation
    return result


def _pose_vector(matrix: np.ndarray) -> np.ndarray:
    angle = math.atan2(matrix[1, 0], matrix[0, 0])
    return np.asarray(
        [*matrix[:3, 3], 0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)],
        dtype=np.float64,
    )


def _write_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object], dict[str, np.ndarray]]:
    sample_id = "rack-plate-000000"
    object_points = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.00, 0.03, 0.00],
            [0.02, 0.01, 0.02],
        ],
        dtype=np.float64,
    )
    rack = np.asarray(
        [[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [-0.1, 0.1, 0.0], [0.1, 0.1, 0.0]],
        dtype=np.float32,
    )
    source_pose = _pose((0.2, -0.1, 0.3), 90.0)
    target_pose = _pose((-0.2, 0.15, 0.4), -45.0)
    predicted_pose = _pose((0.1, 0.25, 0.5), 15.0)
    source = _transform(object_points, source_pose)
    target = _transform(object_points, target_pose)
    training_path = tmp_path / f"{sample_id}.npz"
    np.savez(
        training_path,
        multi_obj_final_obj_pose=np.asarray(
            {"parent": _pose_vector(np.eye(4)), "child": _pose_vector(target_pose)},
            dtype=object,
        ),
        multi_obj_final_pcd=np.asarray({"parent": rack, "child": target}, dtype=object),
        multi_obj_names=np.asarray({"parent": "rack", "child": "plate"}, dtype=object),
        multi_obj_start_obj_pose=np.asarray(
            {"parent": _pose_vector(np.eye(4)), "child": _pose_vector(source_pose)},
            dtype=object,
        ),
        multi_obj_start_pcd=np.asarray({"parent": rack, "child": source}, dtype=object),
        placegen_scene_center_world=np.zeros(3, dtype=np.float32),
    )
    translation_error = float(
        np.linalg.norm(predicted_pose[:3, 3] - target_pose[:3, 3]) * 1000
    )
    rotation_error = 60.0
    # This fixture intentionally models the original producer, before the
    # optional diagnostic metadata fields were added.
    report = {
        field: None for field in REPORT_FIELDS if field not in OPTIONAL_REPORT_FIELDS
    }
    report.update(
        {
            "schema": REPORT_SCHEMA,
            "quality_claim": False,
            "diagnostic_scope": "single-export-overfit-not-generalization",
            "sample_id": sample_id,
            "training_npz": str(training_path),
            "training_npz_sha256": hashlib.sha256(
                training_path.read_bytes()
            ).hexdigest(),
            "predicted_world_from_object": predicted_pose.tolist(),
            "target_world_from_object": target_pose.tolist(),
            "world_translation_error_mm": translation_error,
            "world_rotation_error_deg": rotation_error,
            "final_sampled_ordered_rmse_m": 0.42,
            "candidate_index": 0,
            "candidate_count": 1,
            "sampling_only_after_final_step": True,
            "finite": True,
        }
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    expected = {
        "source": source.astype(np.float64),
        "target": target.astype(np.float64),
        "predicted": _transform(object_points, predicted_pose).astype(np.float64),
    }
    return report_path, training_path, report, expected


def test_world_reconstruction_uses_world_from_object_column_convention(
    tmp_path: Path,
) -> None:
    report_path, _, _, expected = _write_fixture(tmp_path)

    data = load_visualization_data(report_path)

    np.testing.assert_allclose(data.source_child_world, expected["source"], atol=1e-7)
    np.testing.assert_allclose(data.target_child_world, expected["target"], atol=1e-7)
    np.testing.assert_allclose(
        data.predicted_child_world, expected["predicted"], atol=1e-7
    )


def _add_rotation_metadata(report: dict[str, object]) -> None:
    report.update(
        {
            "diagnostic_flags": {
                "disable_rotation_noise": True,
                "rescale_learned_sigmas": True,
            },
            "diagnostic_actual_state": {
                "config_learn_sigma": True,
                "network_learn_sigma": True,
                "diffusion_model_var_type": "LEARNED_RANGE",
                "diffusion_loss_type": "RESCALED_MSE",
                "diffusion_rotation_noise_scale": 0.0,
            },
            "learned_sigma_loss_override": {
                "enabled": True,
                "loss_type_before": "MSE",
                "loss_type_after": "RESCALED_MSE",
                "vb_scale": 0.1,
            },
            "rotation_noise_override": {
                "enabled": True,
                "changed": True,
                "rotation_noise_scale_before": 45.0,
                "rotation_noise_scale_after": 0.0,
            },
        }
    )


def test_optional_rotation_metadata_is_accepted_and_cross_checked(
    tmp_path: Path,
) -> None:
    report_path, _, report, _ = _write_fixture(tmp_path)
    _add_rotation_metadata(report)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    data = load_visualization_data(report_path)

    assert data.sample_id == "rack-plate-000000"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "diagnostic_flags",
            {"disable_rotation_noise": True},
            "fields must match schema exactly",
        ),
        (
            "diagnostic_flags",
            {
                "disable_rotation_noise": True,
                "rescale_learned_sigmas": 1,
            },
            "must be boolean",
        ),
        (
            "diagnostic_actual_state",
            {
                "config_learn_sigma": True,
                "network_learn_sigma": True,
                "diffusion_model_var_type": "LEARNED_RANGE",
                "diffusion_loss_type": "RESCALED_MSE",
                "diffusion_rotation_noise_scale": "0",
            },
            "must be a finite number",
        ),
        (
            "diagnostic_actual_state",
            {
                "config_learn_sigma": True,
                "network_learn_sigma": True,
                "diffusion_model_var_type": "LEARNED_RANGE",
                "diffusion_loss_type": "RESCALED_MSE",
                "diffusion_rotation_noise_scale": 0.0,
                "unexpected": False,
            },
            "fields must match schema exactly",
        ),
        (
            "rotation_noise_override",
            {
                "enabled": True,
                "changed": False,
                "rotation_noise_scale_before": 45.0,
                "rotation_noise_scale_after": 0.0,
            },
            "changed disagrees",
        ),
        (
            "rotation_noise_override",
            {
                "enabled": True,
                "changed": True,
                "rotation_noise_scale_before": 45.0,
                "rotation_noise_scale_after": 0.0,
                "unexpected": 0,
            },
            "fields must match schema exactly",
        ),
    ],
)
def test_optional_rotation_metadata_is_strict(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    report_path, _, report, _ = _write_fixture(tmp_path)
    _add_rotation_metadata(report)
    report[field] = value
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=match):
        load_visualization_data(report_path)


def test_optional_rotation_metadata_rejects_cross_field_scale_mismatch(
    tmp_path: Path,
) -> None:
    report_path, _, report, _ = _write_fixture(tmp_path)
    _add_rotation_metadata(report)
    state = report["diagnostic_actual_state"]
    assert isinstance(state, dict)
    state["diffusion_rotation_noise_scale"] = 45.0
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="reports a non-zero scale"):
        load_visualization_data(report_path)


def test_optional_rotation_metadata_rejects_flag_and_override_mismatch(
    tmp_path: Path,
) -> None:
    report_path, _, report, _ = _write_fixture(tmp_path)
    _add_rotation_metadata(report)
    flags = report["diagnostic_flags"]
    assert isinstance(flags, dict)
    flags["disable_rotation_noise"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="disagrees with rotation_noise_override"):
        load_visualization_data(report_path)


def test_optional_rotation_metadata_rejects_loss_type_mismatch(
    tmp_path: Path,
) -> None:
    report_path, _, report, _ = _write_fixture(tmp_path)
    _add_rotation_metadata(report)
    report["learned_sigma_loss_override"] = {
        "enabled": True,
        "loss_type_before": "MSE",
        "loss_type_after": "RESCALED_MSE",
        "vb_scale": 0.1,
    }
    state = report["diagnostic_actual_state"]
    assert isinstance(state, dict)
    state["diffusion_loss_type"] = "MSE"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="loss type disagrees"):
        load_visualization_data(report_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda report: report.update(schema="wrong/0.1"), "unsupported.*schema"),
        (lambda report: report.update(sample_id="different"), "basename.*sample_id"),
        (lambda report: report.update(training_npz_sha256="0" * 64), "SHA-256"),
        (
            lambda report: report.update(
                predicted_world_from_object=np.zeros((4, 4)).tolist()
            ),
            "proper SE\\(3\\)",
        ),
        (
            lambda report: report.update(world_translation_error_mm=0.0),
            "translation error disagrees",
        ),
    ],
)
def test_report_identity_pose_and_metric_validation_fail_closed(
    tmp_path: Path, mutation, match: str
) -> None:
    report_path, _, report, _ = _write_fixture(tmp_path)
    modified = copy.deepcopy(report)
    mutation(modified)
    report_path.write_text(json.dumps(modified), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_visualization_data(report_path)


def test_relocated_training_override_still_requires_same_id_and_sha(
    tmp_path: Path,
) -> None:
    report_path, training_path, _, _ = _write_fixture(tmp_path)
    relocated_dir = tmp_path / "relocated"
    relocated_dir.mkdir()
    relocated = relocated_dir / training_path.name
    relocated.write_bytes(training_path.read_bytes())

    data = load_visualization_data(report_path, training_npz_override=relocated)

    assert data.training_npz == relocated.resolve()


def test_training_clouds_must_be_finite_even_when_file_sha_is_updated(
    tmp_path: Path,
) -> None:
    report_path, training_path, report, _ = _write_fixture(tmp_path)
    with np.load(training_path, allow_pickle=True) as data:
        values = {name: data[name] for name in data.files}
    start = dict(values["multi_obj_start_pcd"].item())
    start["child"] = start["child"].copy()
    start["child"][0, 0] = np.nan
    values["multi_obj_start_pcd"] = np.asarray(start, dtype=object)
    np.savez(training_path, **values)
    report["training_npz_sha256"] = hashlib.sha256(
        training_path.read_bytes()
    ).hexdigest()
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="start child must be finite"):
        load_visualization_data(report_path)


def test_run_publishes_no_replace_html_and_summary_with_hashes(tmp_path: Path) -> None:
    report_path, training_path, _, _ = _write_fixture(tmp_path)
    html_path = tmp_path / "visualization.html"
    summary_path = tmp_path / "visualization.json"
    args = Namespace(
        report=report_path,
        training_npz=training_path,
        output_html=html_path,
        summary=summary_path,
        show=False,
    )

    summary = run(
        args,
        html_renderer=lambda data, show: "<html><body>validated</body></html>",
    )

    assert summary["schema"] == SUMMARY_SCHEMA
    assert summary["model_rerun"] is False
    assert summary["quality_claim"] is False
    assert summary["color_semantics"]["predicted_child"].endswith("(red)")
    assert (
        summary["output_html_sha256"]
        == hashlib.sha256(html_path.read_bytes()).hexdigest()
    )
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    with pytest.raises(FileExistsError, match="overwrite HTML"):
        run(args, html_renderer=lambda data, show: "<html></html>")


def test_renderer_calls_rpad_pointcloud_for_all_four_world_clouds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path, _, _, _ = _write_fixture(tmp_path)
    data = load_visualization_data(report_path)
    calls: list[dict[str, object]] = []

    class FakeFigure:
        def __init__(self) -> None:
            self.traces = []

        def add_trace(self, trace) -> None:
            self.traces.append(trace)

        def update_layout(self, **kwargs) -> None:
            self.layout = kwargs

        def show(self) -> None:
            raise AssertionError("show must remain opt-in")

        def to_html(self, **kwargs) -> str:
            assert kwargs == {"full_html": True, "include_plotlyjs": True}
            assert len(self.traces) == 4
            return "<html>self-contained</html>"

    plots = types.ModuleType("rpad.visualize_3d.plots")

    def pointcloud(points, **kwargs):
        calls.append({"points": points, **kwargs})
        return kwargs["name"]

    plots.pointcloud = pointcloud
    graph_objects = types.ModuleType("plotly.graph_objects")
    graph_objects.Figure = FakeFigure
    modules = {
        "rpad": types.ModuleType("rpad"),
        "rpad.visualize_3d": types.ModuleType("rpad.visualize_3d"),
        "rpad.visualize_3d.plots": plots,
        "plotly": types.ModuleType("plotly"),
        "plotly.graph_objects": graph_objects,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    html = render_html(data)

    assert html == "<html>self-contained</html>"
    assert [call["name"] for call in calls] == [
        "Rack parent",
        "Source child",
        "Target child",
        "Predicted child",
    ]
    assert all(call["scene"] == "scene" for call in calls)
