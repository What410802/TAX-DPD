"""Strict PlaceGen process-seam runner for target-free TAX-DPD prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from non_rigid.utils.placegen_checkpoint import load_grouped_checkpoint
from non_rigid.utils.placegen_grouped import MODEL_CAPABILITY, sha256_file
from non_rigid.utils.placegen_prediction import predict_pose_candidates
from non_rigid.utils.placegen_request import (
    PREDICT_RESPONSE_PROFILE,
    load_prediction_request,
    load_request_observation,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    request = load_prediction_request(args.request)
    supplied_checkpoint = Path(args.checkpoint).expanduser()
    if supplied_checkpoint.is_symlink() or not supplied_checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint must be a regular file: {supplied_checkpoint}"
        )
    checkpoint_path = supplied_checkpoint.resolve(strict=True)
    if sha256_file(checkpoint_path) != request.checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 does not match the prediction request")
    supplied_output = Path(args.output).expanduser()
    if supplied_output.exists() or supplied_output.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite prediction response: {supplied_output}"
        )
    output_path = supplied_output.resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA prediction but CUDA is unavailable")
    observation = load_request_observation(request)
    loaded = load_grouped_checkpoint(checkpoint_path, device)
    point_counts = loaded.payload["dataset"]["point_counts"]
    if point_counts != {
        "action": len(observation.pc_action_centered),
        "anchor": len(observation.pc_anchor_centered),
    }:
        raise ValueError("request point counts differ from the checkpoint dataset")
    result = predict_pose_candidates(
        loaded.module,
        observation,
        device,
        candidate_count=request.candidate_count,
        seed=request.seed,
    )
    response: dict[str, object] = {
        "profile": PREDICT_RESPONSE_PROFILE,
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
        "model": MODEL_CAPABILITY,
        "checkpoint_sha256": request.checkpoint_sha256,
        "input_npz_sha256": request.input_npz_sha256,
        "seed": request.seed,
        "candidate_count": len(result.candidates),
        "ground_truth_free": True,
        "candidates": list(result.candidates),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(response, allow_nan=False, indent=2, sort_keys=True) + "\n"
        )
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    print(json.dumps(run(parser.parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
