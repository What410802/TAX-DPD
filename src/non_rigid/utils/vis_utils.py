"""Visualization utilities for point cloud logging."""

from typing import List

import numpy as np
import torch

# Named colour map → RGB (0-255)
_COLORS = {
    "red":    (255,   0,   0),
    "green":  (  0, 255,   0),
    "blue":   (  0,   0, 255),
    "yellow": (255, 255,   0),
    "cyan":   (  0, 255, 255),
    "magenta":(255,   0, 255),
    "white":  (255, 255, 255),
    "orange": (255, 165,   0),
    "purple": (128,   0, 128),
    "grey":   (128, 128, 128),
}


def get_color(
    tensor_list: List[torch.Tensor],
    color_list: List[str],
) -> np.ndarray:
    """
    Concatenate point clouds with per-cloud colours into a single XYZRGB array.

    Args:
        tensor_list: List of [N, 3] float tensors (x, y, z coordinates).
        color_list:  List of colour name strings, one per tensor.

    Returns:
        np.ndarray of shape [sum(N_i), 6] with columns [x, y, z, r, g, b].
        Suitable for passing directly to ``wandb.Object3D``.
    """
    assert len(tensor_list) == len(color_list), (
        "tensor_list and color_list must have the same length"
    )

    chunks = []
    for tensor, color_name in zip(tensor_list, color_list):
        pts = tensor.detach().cpu().numpy()  # [N, 3]
        r, g, b = _COLORS.get(color_name.lower(), (200, 200, 200))
        rgb = np.full((pts.shape[0], 3), [r, g, b], dtype=np.float32)
        chunks.append(np.concatenate([pts, rgb], axis=1))  # [N, 6]

    return np.concatenate(chunks, axis=0)  # [sum(N), 6]
