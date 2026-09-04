"""Dataset-size contracts for the native PlaceGen FM export reader."""

from __future__ import annotations

import pytest

from non_rigid.utils.placegen_fm_export import _split_counts


def test_split_counts_accept_positive_nonstandard_dataset_sizes() -> None:
    assert _split_counts(
        {"train": 383, "validation": 64, "test": 64},
        "split_counts",
    ) == {"train": 383, "validation": 64, "test": 64}


@pytest.mark.parametrize(
    "counts",
    (
        {"train": 0, "validation": 64, "test": 64},
        {"train": 383, "validation": True, "test": 64},
        {"train": 383, "validation": 64},
        {"train": 383, "validation": 64, "test": 64, "extra": 1},
    ),
)
def test_split_counts_reject_invalid_dataset_sizes(counts: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _split_counts(counts, "split_counts")
