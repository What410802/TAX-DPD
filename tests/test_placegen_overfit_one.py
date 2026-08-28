"""Pure gate contracts for the bounded one-sample overfit diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.placegen_overfit_one import _atomic_json, evaluate_overfit_gate


def _green_metrics() -> dict[str, float | bool]:
    return {
        "finite": True,
        "parameter_changed": True,
        "first_cycle_loss_median": 1.0,
        "second_cycle_loss_median": 0.2,
        "noop_ordered_rmse_m": 0.2,
        "sampled_ordered_rmse_m": 0.02,
        "world_translation_error_mm": 20.0,
        "world_rotation_error_deg": 10.0,
    }


def test_overfit_gate_accepts_preregistered_boundaries() -> None:
    result = evaluate_overfit_gate(_green_metrics(), steps=200)

    assert result["passed"] is True
    assert result["severe_red_flag"] is False
    assert result["derived"]["second_vs_first_cycle_loss_ratio"] == pytest.approx(0.2)
    assert result["derived"]["sampled_rmse_reduction"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("finite", False),
        ("parameter_changed", False),
        ("second_cycle_loss_median", 0.200001),
        ("sampled_ordered_rmse_m", 0.020001),
        ("world_translation_error_mm", 20.001),
        ("world_rotation_error_deg", 10.001),
    ],
)
def test_overfit_gate_fails_each_preregistered_check(
    field: str, value: float | bool
) -> None:
    metrics = _green_metrics()
    metrics[field] = value

    assert evaluate_overfit_gate(metrics, steps=200)["passed"] is False


def test_overfit_gate_marks_half_meter_outputs_as_severe() -> None:
    metrics = _green_metrics()
    metrics["sampled_ordered_rmse_m"] = 0.5
    metrics["world_translation_error_mm"] = 500.0
    result = evaluate_overfit_gate(metrics, steps=200)

    assert result["passed"] is False
    assert result["severe_red_flag"] is True
    assert all(result["severe_red_flags"].values())


def test_overfit_gate_requires_two_matched_timestep_cycles() -> None:
    with pytest.raises(ValueError, match="must equal 200"):
        evaluate_overfit_gate(_green_metrics(), steps=199)
    with pytest.raises(ValueError, match="must equal 200"):
        evaluate_overfit_gate(_green_metrics(), steps=201)


def test_overfit_gate_rejects_missing_metrics() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        evaluate_overfit_gate({}, steps=200)


def test_atomic_report_publish_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    _atomic_json(output, {"version": 1})

    with pytest.raises(FileExistsError):
        _atomic_json(output, {"version": 2})

    assert json.loads(output.read_text(encoding="utf-8")) == {"version": 1}
    assert not list(tmp_path.glob(".*.tmp-*"))
