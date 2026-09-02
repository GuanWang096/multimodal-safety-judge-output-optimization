from __future__ import annotations

import sys
from pathlib import Path

import pytest


RGG_SOURCE = Path(__file__).resolve().parents[2] / "rggpbft"
if str(RGG_SOURCE) not in sys.path:
    sys.path.insert(0, str(RGG_SOURCE))

from protocol import quorum


@pytest.mark.parametrize(
    ("member_count", "expected"),
    [
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 3),
        (5, 4),
        (6, 4),
        (7, 5),
        (8, 6),
        (9, 6),
        (16, 11),
        (20, 14),
        (24, 16),
    ],
)
def test_quorum_boundary_values(member_count: int, expected: int) -> None:
    assert quorum(member_count) == expected


def test_quorum_guarantees_honest_intersection_and_liveness() -> None:
    for member_count in range(1, 65):
        faults = (member_count - 1) // 3
        threshold = quorum(member_count)

        assert 2 * threshold - member_count > faults
        assert threshold <= member_count - faults


@pytest.mark.parametrize("member_count", [0, -1])
def test_quorum_rejects_nonpositive_membership(member_count: int) -> None:
    with pytest.raises(ValueError, match="member count must be positive"):
        quorum(member_count)
