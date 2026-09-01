"""Deterministic random-prior overfit gate for the PlaceGen TAX3Dv2 FM path."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from non_rigid.datasets.rigid import PlaceGenTaxDpdDataset
from non_rigid.models.flow_matching import flow_matching_loss, integrate_velocity
from non_rigid.utils.script_utils import create_model


def _proper_fit(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _singular, right_transpose = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1] *= -1
        rotation = right_transpose.T @ left.T
    translation = target_center - rotation @ source_center
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.linalg.norm(translation) * 1000.0), float(np.degrees(np.arccos(cosine)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--noise-scale", type=float, default=0.08)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--translation-limit-mm", type=float, default=20.0)
    parser.add_argument("--rotation-limit-deg", type=float, default=10.0)
    parser.add_argument("--point-rmse-limit-mm", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.output_json.exists() or args.output_json.is_symlink():
        raise FileExistsError(f"overfit report already exists: {args.output_json}")
    if args.steps <= 0 or args.candidate_count <= 0:
        raise ValueError("steps and candidate-count must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = Path(__file__).parents[1]
    model_cfg = OmegaConf.load(root / "configs/model/tax3dv2_fm.yaml")
    training_cfg = OmegaConf.load(root / "configs/training/placegen_taxdpd_tax3dv2_fm.yaml")
    dataset_cfg = OmegaConf.load(root / "configs/dataset/placegen_taxdpd.yaml")
    dataset_cfg.data_dir = str(args.data_root.expanduser().resolve())
    dataset_cfg.train_dataset_size = 1
    dataset_cfg.val_dataset_size = 1
    dataset_cfg.test_dataset_size = 1
    _, module = create_model(
        OmegaConf.create(
            {"model": model_cfg, "training": training_cfg, "dataset": dataset_cfg}
        )
    )
    module = module.to(device)
    module.noise_scale = float(args.noise_scale)
    dataset = PlaceGenTaxDpdDataset(dataset_cfg.data_dir, dataset_cfg, "train")
    raw = dataset[0]
    batch = {
        key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
        for key, value in raw.items()
    }
    target = module._get_x_start(batch)
    frame, shape = module._split_target(target)
    endpoint = torch.cat((frame, shape), dim=2)
    model_kwargs = module._model_kwargs(batch)
    optimizer = torch.optim.Adam(module.parameters(), lr=args.learning_rate)
    loss_history: list[float] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        source = module._source_from_observation(batch["pc_action"].permute(0, 2, 1))
        loss = flow_matching_loss(
            module._velocity_model(model_kwargs), endpoint, source=source
        )[0].mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite FM loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        if step in {0, args.steps - 1} or (step + 1) % 100 == 0:
            loss_history.append(float(loss.detach().cpu()))

    scale = float(dataset.scale_factor)
    target_world = (batch["pc"][0] / scale).detach().cpu().numpy()
    candidates: list[dict[str, float | int]] = []
    module.eval()
    with torch.no_grad():
        for candidate_index in range(args.candidate_count):
            candidate_seed = args.seed + 1_000_000 + candidate_index
            torch.manual_seed(candidate_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(candidate_seed)
            source = module._source_from_observation(batch["pc_action"].permute(0, 2, 1))
            final = integrate_velocity(
                module._velocity_model(model_kwargs),
                source,
                steps=int(model_cfg.fm_steps),
                method=str(model_cfg.fm_solver),
            )
            predicted = ((final[:, :, :1] + final[:, :, 1:]).permute(0, 2, 1)[0] / scale)
            predicted_world = predicted.cpu().numpy()
            point_rmse_mm = float(
                np.sqrt(np.mean((predicted_world - target_world) ** 2)) * 1000.0
            )
            translation_mm, rotation_deg = _proper_fit(predicted_world, target_world)
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "seed": candidate_seed,
                    "point_rmse_mm": point_rmse_mm,
                    "translation_error_mm": translation_mm,
                    "rotation_error_deg": rotation_deg,
                }
            )
    maxima = {
        "point_rmse_mm": max(float(item["point_rmse_mm"]) for item in candidates),
        "translation_error_mm": max(
            float(item["translation_error_mm"]) for item in candidates
        ),
        "rotation_error_deg": max(float(item["rotation_error_deg"]) for item in candidates),
    }
    passed = (
        maxima["point_rmse_mm"] < args.point_rmse_limit_mm
        and maxima["translation_error_mm"] < args.translation_limit_mm
        and maxima["rotation_error_deg"] < args.rotation_limit_deg
    )
    report = {
        "profile": "placegen.tax3dv2-fm-random-prior-overfit/0.1",
        "passed": passed,
        "sample_index": 0,
        "training_seed": args.seed,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "coordinate_scale": scale,
        "noise_scale_model_units": module.noise_scale,
        "noise_scale_world_mm": module.noise_scale / scale * 1000.0,
        "candidate_count": args.candidate_count,
        "limits": {
            "point_rmse_mm": args.point_rmse_limit_mm,
            "translation_error_mm": args.translation_limit_mm,
            "rotation_error_deg": args.rotation_limit_deg,
        },
        "maxima": maxima,
        "loss_history": loss_history,
        "candidates": candidates,
        "planner_called": False,
        "simulator_called": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_json": str(args.output_json), "passed": passed, **maxima}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
