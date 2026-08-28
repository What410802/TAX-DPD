"""Pure gate contracts for the bounded one-sample overfit diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.placegen_overfit_one import (
    _atomic_json,
    apply_rotation_noise_override,
    build_parser,
    diagnostic_flag_state,
    evaluate_overfit_gate,
    learned_sigma_loss_override,
    rotation_noise_override,
    scalar_loss_terms,
    scalar_term_ratios,
)


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


@pytest.mark.parametrize("steps", [200, 1000])
def test_overfit_gate_accepts_complete_matched_timestep_cycles(steps: int) -> None:
    result = evaluate_overfit_gate(_green_metrics(), steps=steps)

    assert result["passed"] is True
    assert result["thresholds"]["steps"] == steps


@pytest.mark.parametrize("steps", [199, 201])
def test_overfit_gate_rejects_incomplete_timestep_cycles(steps: int) -> None:
    with pytest.raises(ValueError, match="at least 200 and a multiple of 100"):
        evaluate_overfit_gate(_green_metrics(), steps=steps)


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


def test_scalar_loss_probe_helpers_keep_all_scalars_and_json_safe_ratios() -> None:
    before = scalar_loss_terms(
        {"loss": 2.0, "mse": 1.0, "vb": [0.0], "not_scalar": [1.0, 2.0]}
    )
    after = scalar_loss_terms({"loss": 0.5, "mse": 0.25, "vb": 0.0})

    assert before == {"loss": 2.0, "mse": 1.0, "vb": 0.0}
    assert scalar_term_ratios(before, after) == {
        "loss": 0.25,
        "mse": 0.25,
        "vb": None,
    }


def test_learned_sigma_loss_override_is_opt_in_and_scales_vb() -> None:
    default = learned_sigma_loss_override(
        False, learn_sigma=True, original_loss_type="MSE", diffusion_steps=100
    )
    enabled = learned_sigma_loss_override(
        True, learn_sigma=True, original_loss_type="MSE", diffusion_steps=100
    )

    assert default == {
        "enabled": False,
        "loss_type_before": "MSE",
        "loss_type_after": "MSE",
        "vb_scale": 1.0,
    }
    assert enabled == {
        "enabled": True,
        "loss_type_before": "MSE",
        "loss_type_after": "RESCALED_MSE",
        "vb_scale": 0.1,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learn_sigma": False, "original_loss_type": "MSE", "diffusion_steps": 100},
        {
            "learn_sigma": True,
            "original_loss_type": "RESCALED_MSE",
            "diffusion_steps": 100,
        },
        {"learn_sigma": True, "original_loss_type": "MSE", "diffusion_steps": 99},
    ],
)
def test_learned_sigma_loss_override_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        learned_sigma_loss_override(True, **kwargs)


def test_rescaled_learned_sigma_cli_flag_is_opt_in() -> None:
    required = [
        "--data-root",
        "/data",
        "--input-npz",
        "/input.npz",
        "--manifest",
        "/manifest.json",
        "--report",
        "/report.json",
    ]

    assert build_parser().parse_args(required).rescale_learned_sigmas is False
    assert (
        build_parser()
        .parse_args([*required, "--rescale-learned-sigmas"])
        .rescale_learned_sigmas
        is True
    )


def test_rotation_noise_override_default_preserves_config_scale() -> None:
    result = rotation_noise_override(False, original_scale=45.0)

    assert result == {
        "enabled": False,
        "rotation_noise_scale_before": 45.0,
        "rotation_noise_scale_after": 45.0,
        "changed": False,
    }


def test_rotation_noise_override_only_changes_diagnostic_instance() -> None:
    result = rotation_noise_override(True, original_scale=45.0)

    assert result == {
        "enabled": True,
        "rotation_noise_scale_before": 45.0,
        "rotation_noise_scale_after": 0.0,
        "changed": True,
    }


def test_apply_rotation_noise_override_does_not_mutate_source_config() -> None:
    source_config = SimpleNamespace(diff_rotation_noise_scale=45.0)
    diffusion = SimpleNamespace(
        rotation_noise_scale=source_config.diff_rotation_noise_scale
    )
    override = rotation_noise_override(
        True, original_scale=source_config.diff_rotation_noise_scale
    )

    actual = apply_rotation_noise_override(diffusion, override)

    assert actual == 0.0
    assert diffusion.rotation_noise_scale == 0.0
    assert source_config.diff_rotation_noise_scale == 45.0


def test_apply_rotation_noise_override_default_preserves_instance() -> None:
    diffusion = SimpleNamespace(rotation_noise_scale=45.0)
    override = rotation_noise_override(False, original_scale=45.0)

    assert apply_rotation_noise_override(diffusion, override) == 45.0
    assert diffusion.rotation_noise_scale == 45.0


def test_apply_rotation_noise_override_accepts_native_false_default() -> None:
    diffusion = SimpleNamespace(rotation_noise_scale=False)
    override = rotation_noise_override(False, original_scale=False)

    assert apply_rotation_noise_override(diffusion, override) is False
    assert diffusion.rotation_noise_scale is False


def test_apply_rotation_noise_override_fails_closed_when_instance_ignores_write() -> (
    None
):
    class ReadOnlyDiffusion:
        rotation_noise_scale = 45.0

        def __setattr__(self, name: str, value: object) -> None:
            if name == "rotation_noise_scale":
                return
            super().__setattr__(name, value)

    diffusion = ReadOnlyDiffusion()
    override = rotation_noise_override(True, original_scale=45.0)

    with pytest.raises(RuntimeError, match="did not reach"):
        apply_rotation_noise_override(diffusion, override)


def test_rotation_noise_override_keeps_native_false_scale_disabled() -> None:
    result = rotation_noise_override(True, original_scale=False)

    assert result == {
        "enabled": True,
        "rotation_noise_scale_before": False,
        "rotation_noise_scale_after": 0.0,
        "changed": False,
    }


@pytest.mark.parametrize("value", [True, -1.0, float("nan"), float("inf"), "45"])
def test_rotation_noise_override_rejects_invalid_scales(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        rotation_noise_override(False, original_scale=value)


def test_cli_parses_rotation_flag_independently() -> None:
    required = [
        "--data-root",
        "/data",
        "--input-npz",
        "/input.npz",
        "--manifest",
        "/manifest.json",
        "--report",
        "/report.json",
    ]
    defaults = build_parser().parse_args(required)
    assert defaults.disable_rotation_noise is False

    enabled = build_parser().parse_args([*required, "--disable-rotation-noise"])
    assert enabled.disable_rotation_noise is True


def test_diagnostic_flag_state_keeps_supported_switches_independent() -> None:
    assert diagnostic_flag_state(
        disable_rotation_noise=False, rescale_learned_sigmas=False
    ) == {
        "disable_rotation_noise": False,
        "rescale_learned_sigmas": False,
    }
    assert diagnostic_flag_state(
        disable_rotation_noise=True, rescale_learned_sigmas=False
    ) == {
        "disable_rotation_noise": True,
        "rescale_learned_sigmas": False,
    }
    assert diagnostic_flag_state(
        disable_rotation_noise=False, rescale_learned_sigmas=True
    ) == {
        "disable_rotation_noise": False,
        "rescale_learned_sigmas": True,
    }


@pytest.mark.parametrize("field", ["disable_rotation_noise", "rescale_learned_sigmas"])
def test_diagnostic_flag_state_rejects_non_boolean_values(field: str) -> None:
    values: dict[str, object] = {
        "disable_rotation_noise": False,
        "rescale_learned_sigmas": False,
    }
    values[field] = 1

    with pytest.raises(TypeError, match=f"{field} must be a boolean"):
        diagnostic_flag_state(**values)
