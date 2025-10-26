from __future__ import annotations

from typing import Tuple

from .state import PuzzleState, GOAL, GOAL_POS, INDEX_TO_RC, N


def hamming(s: PuzzleState) -> int:
    """
    Count of tiles not in their goal positions (ignoring the blank).

    Parameters
    ----------
    s : PuzzleState

    Returns
    -------
    int
        Number of misplaced tiles ∈ [0, 8].
    """
    tiles = s.tiles
    # Compare against GOAL; skip blank (0)
    return sum(1 for i, v in enumerate(tiles) if v != 0 and v != GOAL[i])


def manhattan(s: PuzzleState) -> int:
    """
    Sum of Manhattan distances from each tile to its goal position (ignore blank).

    Parameters
    ----------
    s : PuzzleState

    Returns
    -------
    int
        Manhattan distance (non-negative), equals 0 on the goal state.
    """
    dist = 0
    for idx, tile in enumerate(s.tiles):
        if tile == 0:
            continue
        r, c = INDEX_TO_RC[idx]
        gr, gc = GOAL_POS[tile]
        dist += abs(r - gr) + abs(c - gc)
    return dist


# Optional: a trivial baseline for sanity checks
def zero_heuristic(_: PuzzleState) -> int:
    """Always returns 0 (equivalent to Uniform Cost Search on unit-cost edges)."""
    return 0


# ------------------------------- Self-test -----------------------------------

if __name__ == "__main__":
    g = PuzzleState(GOAL)
    assert hamming(g) == 0
    assert manhattan(g) == 0

    # One move from goal: blank left of 8 → moving Right solves it
    near = PuzzleState((1, 2, 3, 4, 5, 6, 7, 0, 8))
    # Tile '8' is at (2,2) in GOAL but here at (2,1): Hamming=1, Manhattan=1
    assert hamming(near) == 1
    assert manhattan(near) == 1

    # A slightly scrambled state (values just for a quick check)
    s = PuzzleState((2, 8, 3, 1, 6, 4, 7, 0, 5))
    print("Hamming =", hamming(s))
    print("Manhattan =", manhattan(s))