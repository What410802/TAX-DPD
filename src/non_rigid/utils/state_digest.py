"""Stable identities for tensor state dictionaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import torch


STATE_DICT_PROFILE = "placegen.torch-state-dict/0.1"


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash sorted state keys with each tensor's dtype, shape, and bytes."""

    digest = hashlib.sha256()
    digest.update(STATE_DICT_PROFILE.encode())
    digest.update(b"\0")
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict[{name!r}] must be a torch.Tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def clone_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Create an owning CPU snapshot suitable for comparison and serialization."""

    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in state_dict.items()
    }


__all__ = ["STATE_DICT_PROFILE", "clone_state_dict", "state_dict_sha256"]
