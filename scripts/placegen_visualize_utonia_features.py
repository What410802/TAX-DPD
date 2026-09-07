"""Render cached Utonia slot features on one target-free PlaceGen observation.

The script is intentionally read-only: it verifies the inference manifest, feature-cache
manifest, input archive, geometry key, and feature archive before creating a self-contained
Plotly HTML document and a machine-readable summary.  It never invokes Utonia, TAX3Dv2, a
planner, or a simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from non_rigid.models.encoders import utonia_geometry_sha256


CACHE_SCHEMA = "taxdpd.utonia-static-feature-cache/0.1"
INFERENCE_PROFILE = "placegen.taxpose-inference-index/0.1"
SUMMARY_SCHEMA = "placegen.utonia-feature-visualization/0.1"
EXPECTED_INPUT_FIELDS = frozenset(
    {
        "action_indices",
        "anchor_indices",
        "child_points_world",
        "parent_points_world",
        "scene_center_world",
        "source_world_from_object",
    }
)


@dataclass(frozen=True)
class UtoniaFeatureVisualizationData:
    """Validated Utonia features and their original ordered world-frame slots."""

    sample_id: str
    split: str
    input_npz: Path
    input_npz_sha256: str
    inference_manifest: Path
    inference_manifest_sha256: str
    cache_manifest: Path
    cache_manifest_sha256: str
    feature_npz: Path
    feature_npz_sha256: str
    geometry_sha256: str
    child_points_world: np.ndarray
    parent_points_world: np.ndarray
    features: np.ndarray
    scale_factor: float
    transform_scale: float
    center_shift: bool
    transform_seed: int


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path | str, label: str) -> Path:
    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise ValueError(f"{label} cannot be a symlink: {supplied}")
    if not supplied.is_file():
        raise FileNotFoundError(f"{label} does not exist: {supplied}")
    return supplied.resolve(strict=True)


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(document, dict):
        raise TypeError(f"{label} must be a JSON object")
    return document


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
        raise ValueError("sample_id must be a safe non-empty identifier")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _relative_regular_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path must be a non-empty string")
    pure_path = PurePosixPath(relative)
    if pure_path.is_absolute() or any(part in {"", ".", ".."} for part in pure_path.parts):
        raise ValueError(f"{label} path is unsafe")
    candidate = root.joinpath(*pure_path.parts)
    if candidate.is_symlink():
        raise ValueError(f"{label} cannot be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes its manifest root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _find_record(records: Any, sample_id: str, label: str) -> Mapping[str, Any]:
    if not isinstance(records, list):
        raise TypeError(f"{label} records must be a JSON array")
    matches = [record for record in records if isinstance(record, dict) and record.get("sample_id") == sample_id]
    if len(matches) != 1:
        raise ValueError(f"{label} must contain exactly one record for {sample_id!r}")
    return matches[0]


def _finite_array(value: Any, shape: tuple[int, ...], label: str, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite {dtype.name}{shape}")
    return np.ascontiguousarray(array)


def _finite_positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _load_input_points(input_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(input_npz, allow_pickle=False) as archive:
        fields = frozenset(archive.files)
        if fields != EXPECTED_INPUT_FIELDS:
            missing = sorted(EXPECTED_INPUT_FIELDS.difference(fields))
            unexpected = sorted(fields.difference(EXPECTED_INPUT_FIELDS))
            raise ValueError(
                "inference NPZ fields must match the target-free contract exactly; "
                f"missing={missing}, unexpected={unexpected}"
            )
        child = _finite_array(
            archive["child_points_world"], (1024, 3), "child_points_world", np.dtype(np.float32)
        )
        parent = _finite_array(
            archive["parent_points_world"], (1024, 3), "parent_points_world", np.dtype(np.float32)
        )
        _finite_array(archive["action_indices"], (1024,), "action_indices", np.dtype(np.int64))
        _finite_array(archive["anchor_indices"], (1024,), "anchor_indices", np.dtype(np.int64))
        _finite_array(archive["scene_center_world"], (3,), "scene_center_world", np.dtype(np.float64))
        _finite_array(
            archive["source_world_from_object"],
            (4, 4),
            "source_world_from_object",
            np.dtype(np.float64),
        )
    return child, parent


def _feature_statistics(features: np.ndarray) -> dict[str, Any]:
    """Return a deterministic PCA scalar and distribution facts for 2048 slot features."""

    centered = features.astype(np.float64, copy=False) - features.mean(axis=0, dtype=np.float64)
    _left, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if not np.isfinite(singular_values).all() or not np.isfinite(right).all():
        raise RuntimeError("feature PCA produced a non-finite result")
    total_variance = float(np.dot(singular_values, singular_values))
    if total_variance <= 0.0:
        raise ValueError("Utonia feature PCA is undefined for constant features")
    principal_component = centered @ right[0]
    norms = np.linalg.norm(features.astype(np.float64, copy=False), axis=1)
    if not np.isfinite(principal_component).all() or not np.isfinite(norms).all():
        raise RuntimeError("feature statistics contain non-finite values")
    return {
        "pc1": principal_component,
        "feature_norm": norms,
        "pc1_explained_variance_ratio": float(singular_values[0] ** 2 / total_variance),
        "pc1_min": float(np.min(principal_component)),
        "pc1_median": float(np.median(principal_component)),
        "pc1_max": float(np.max(principal_component)),
        "feature_norm_min": float(np.min(norms)),
        "feature_norm_median": float(np.median(norms)),
        "feature_norm_max": float(np.max(norms)),
    }


def load_visualization_data(
    *,
    inference_manifest: Path | str,
    cache_manifest: Path | str,
    sample_id: str,
) -> UtoniaFeatureVisualizationData:
    """Load one identity-checked feature cache record and its original point slots."""

    selected_sample_id = _safe_sample_id(sample_id)
    inference_path = _require_regular_file(inference_manifest, "inference manifest")
    cache_path = _require_regular_file(cache_manifest, "feature-cache manifest")
    inference = _load_json(inference_path, "inference manifest")
    cache = _load_json(cache_path, "feature-cache manifest")
    if inference.get("profile") != INFERENCE_PROFILE:
        raise ValueError("unsupported inference manifest profile")
    if inference.get("ground_truth_free") is not True:
        raise ValueError("feature visualization requires a target-free inference manifest")
    if cache.get("schema") != CACHE_SCHEMA:
        raise ValueError("unsupported Utonia feature-cache schema")
    if cache.get("inference_manifest_sha256") != sha256_file(inference_path):
        raise ValueError("feature cache is not bound to the supplied inference manifest")
    if cache.get("feature_width") != 576:
        raise ValueError("Utonia feature cache must contain 576-D slot features")
    if cache.get("center_shift") is not False:
        raise ValueError("this visualizer supports only utonia_center_shift=false")
    if isinstance(cache.get("transform_seed"), bool) or not isinstance(cache.get("transform_seed"), int):
        raise TypeError("feature cache transform_seed must be an integer")

    inference_record = _find_record(inference.get("samples"), selected_sample_id, "inference manifest")
    cache_record = _find_record(cache.get("records"), selected_sample_id, "feature cache")
    split = inference_record.get("split")
    if split not in {"train", "validation", "test"} or cache_record.get("split") != split:
        raise ValueError("cache and inference record split must agree")
    input_metadata = inference_record.get("input_npz")
    if not isinstance(input_metadata, dict):
        raise TypeError("inference record input_npz must be an object")
    input_path = _relative_regular_file(inference_path.parent, input_metadata.get("path"), "input NPZ")
    input_sha256 = _sha256(input_metadata.get("sha256"), "inference record input NPZ SHA-256")
    if sha256_file(input_path) != input_sha256:
        raise ValueError("input NPZ SHA-256 does not match the inference manifest")
    child, parent = _load_input_points(input_path)

    scale_factor = _finite_positive_float(cache.get("scale_factor"), "cache scale_factor")
    transform_scale = _finite_positive_float(cache.get("transform_scale"), "cache transform_scale")
    scene = np.concatenate((child * np.float32(scale_factor), parent * np.float32(scale_factor)), axis=0)
    scene -= scene.mean(axis=0, dtype=np.float32)
    geometry_sha256 = utonia_geometry_sha256(torch.from_numpy(scene).t().unsqueeze(0))
    if cache_record.get("geometry_sha256") != geometry_sha256:
        raise ValueError("input geometry key does not match the Utonia feature-cache record")
    if cache_record.get("slot_count") != 2048:
        raise ValueError("Utonia feature-cache record must contain 2048 slots")
    feature_path = _relative_regular_file(cache_path.parent, cache_record.get("feature_npz"), "feature NPZ")
    feature_sha256 = _sha256(cache_record.get("feature_npz_sha256"), "feature NPZ SHA-256")
    if sha256_file(feature_path) != feature_sha256:
        raise ValueError("feature NPZ SHA-256 does not match the feature cache")
    with np.load(feature_path, allow_pickle=False) as archive:
        if frozenset(archive.files) != {"features"}:
            raise ValueError("feature NPZ must contain only a features array")
        features = _finite_array(archive["features"], (2048, 576), "features", np.dtype(np.float32))

    values = (child, parent, features)
    for value in values:
        value.setflags(write=False)
    return UtoniaFeatureVisualizationData(
        sample_id=selected_sample_id,
        split=split,
        input_npz=input_path,
        input_npz_sha256=input_sha256,
        inference_manifest=inference_path,
        inference_manifest_sha256=sha256_file(inference_path),
        cache_manifest=cache_path,
        cache_manifest_sha256=sha256_file(cache_path),
        feature_npz=feature_path,
        feature_npz_sha256=feature_sha256,
        geometry_sha256=geometry_sha256,
        child_points_world=child,
        parent_points_world=parent,
        features=features,
        scale_factor=scale_factor,
        transform_scale=transform_scale,
        center_shift=False,
        transform_seed=int(cache["transform_seed"]),
    )


def render_html(data: UtoniaFeatureVisualizationData, *, show: bool = False) -> str:
    """Render input geometry, feature PC1, and feature norm as three 3D views."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    statistics = _feature_statistics(data.features)
    pc1 = statistics["pc1"]
    norm = statistics["feature_norm"]
    parent_slice = slice(1024, 2048)
    child_slice = slice(0, 1024)
    figure = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Original ordered slots",
            "Utonia 576-D feature: PC1",
            "Utonia 576-D feature: L2 norm",
        ),
    )

    def add_geometry(points: np.ndarray, name: str, color: str) -> None:
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                name=name,
                marker={"size": 2.4, "color": color},
                hovertemplate=f"{name}<br>slot=%{{pointNumber}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    def add_scalar(
        points: np.ndarray,
        values: np.ndarray,
        name: str,
        colorbar_title: str,
        column: int,
        show_scale: bool,
    ) -> None:
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                name=name,
                marker={
                    "size": 2.6,
                    "color": values,
                    "colorscale": "Viridis",
                    "cmin": float(np.min(values)),
                    "cmax": float(np.max(values)),
                    "showscale": show_scale,
                    "colorbar": {"title": colorbar_title},
                },
                hovertemplate=(
                    f"{name}<br>slot=%{{pointNumber}}<br>{colorbar_title}=%{{marker.color:.5g}}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )

    add_geometry(data.parent_points_world, "Parent/rack slots", "#5B6573")
    add_geometry(data.child_points_world, "Child/plate slots", "#0173B2")
    pc1_range = (float(np.min(pc1)), float(np.max(pc1)))
    norm_range = (float(np.min(norm)), float(np.max(norm)))
    for points, role, slot_slice in (
        (data.parent_points_world, "Parent/rack", parent_slice),
        (data.child_points_world, "Child/plate", child_slice),
    ):
        add_scalar(
            points,
            pc1[slot_slice],
            f"{role} PC1",
            "PC1",
            2,
            role == "Parent/rack",
        )
        add_scalar(
            points,
            norm[slot_slice],
            f"{role} feature norm",
            "L2 norm",
            3,
            role == "Parent/rack",
        )
    for trace in figure.data:
        if trace.name and "PC1" in trace.name:
            trace.marker.cmin, trace.marker.cmax = pc1_range
        if trace.name and "feature norm" in trace.name:
            trace.marker.cmin, trace.marker.cmax = norm_range
    scene_layout = {"aspectmode": "data", "xaxis_title": "world x (m)", "yaxis_title": "world y (m)", "zaxis_title": "world z (m)"}
    figure.update_layout(
        title=(
            f"{data.sample_id}: cached Utonia output mapped back to 2048 original slots "
            f"(PC1 explains {statistics['pc1_explained_variance_ratio']:.2%})"
        ),
        scene=scene_layout,
        scene2=scene_layout,
        scene3=scene_layout,
        legend={"x": 0.01, "y": 1.0},
        margin={"l": 0, "r": 0, "b": 0, "t": 64},
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


def run(
    args: argparse.Namespace,
    *,
    html_renderer: Callable[..., str] = render_html,
) -> dict[str, Any]:
    """Publish a no-overwrite HTML visualization and optional JSON summary."""

    output_html = Path(args.output_html).expanduser().resolve()
    if output_html.suffix.lower() != ".html":
        raise ValueError("--output-html must end in .html")
    if output_html.exists() or output_html.is_symlink():
        raise FileExistsError(f"refusing to overwrite HTML: {output_html}")
    summary_path = None if args.summary is None else Path(args.summary).expanduser().resolve()
    if summary_path is not None:
        if summary_path.exists() or summary_path.is_symlink():
            raise FileExistsError(f"refusing to overwrite summary: {summary_path}")
        if summary_path == output_html:
            raise ValueError("summary path must differ from HTML output")
    data = load_visualization_data(
        inference_manifest=args.inference_manifest,
        cache_manifest=args.cache_manifest,
        sample_id=args.sample_id,
    )
    html = html_renderer(data, show=bool(args.show))
    if not isinstance(html, str) or "<html" not in html.lower():
        raise RuntimeError("renderer did not return a full HTML document")
    statistics = _feature_statistics(data.features)
    html_sha256 = hashlib.sha256(html.encode("utf-8")).hexdigest()
    summary = {
        "schema": SUMMARY_SCHEMA,
        "sample_id": data.sample_id,
        "split": data.split,
        "model_rerun": False,
        "quality_claim": False,
        "self_contained_html": True,
        "frame": "world-meters",
        "input_npz": str(data.input_npz),
        "input_npz_sha256": data.input_npz_sha256,
        "inference_manifest": str(data.inference_manifest),
        "inference_manifest_sha256": data.inference_manifest_sha256,
        "cache_manifest": str(data.cache_manifest),
        "cache_manifest_sha256": data.cache_manifest_sha256,
        "feature_npz": str(data.feature_npz),
        "feature_npz_sha256": data.feature_npz_sha256,
        "utonia_geometry_sha256": data.geometry_sha256,
        "utonia_config": {
            "feature_width": int(data.features.shape[1]),
            "slot_count": int(data.features.shape[0]),
            "scale_factor": data.scale_factor,
            "transform_scale": data.transform_scale,
            "center_shift": data.center_shift,
            "transform_seed": data.transform_seed,
        },
        "point_counts": {"child": 1024, "parent": 1024},
        "render_semantics": {
            "original_slots": "parent rack gray; child plate blue",
            "pc1": "one joint-PCA scalar, shared Viridis range across child and parent",
            "feature_norm": "per-slot 576-D L2 norm, shared Viridis range across child and parent",
        },
        "feature_statistics": {
            key: value
            for key, value in statistics.items()
            if key not in {"pc1", "feature_norm"}
        },
        "output_html": str(output_html),
        "output_html_sha256": html_sha256,
    }
    _atomic_text(output_html, html)
    if sha256_file(output_html) != html_sha256:
        raise RuntimeError("published HTML SHA-256 differs from rendered content")
    if summary_path is not None:
        _atomic_text(
            summary_path,
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
