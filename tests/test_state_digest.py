"""Unit contracts for checkpoint state identities."""

import torch

from non_rigid.utils.state_digest import clone_state_dict, state_dict_sha256


def test_state_digest_is_order_independent_and_value_sensitive() -> None:
    first = {"b": torch.ones(2), "a": torch.arange(3, dtype=torch.float32)}
    reordered = {"a": first["a"].clone(), "b": first["b"].clone()}
    changed = clone_state_dict(first)
    changed["a"][0] += 1.0

    assert state_dict_sha256(first) == state_dict_sha256(reordered)
    assert state_dict_sha256(first) != state_dict_sha256(changed)


def test_clone_state_dict_owns_unchanging_cpu_tensors() -> None:
    original = {"weight": torch.arange(4, dtype=torch.float32)}
    cloned = clone_state_dict(original)
    original["weight"].add_(10.0)

    torch.testing.assert_close(cloned["weight"], torch.arange(4, dtype=torch.float32))
