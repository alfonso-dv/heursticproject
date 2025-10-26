from __future__ import annotations

import math

from src.state import PuzzleState, GOAL
from src.utils import is_solvable
from src.heuristics import hamming, manhattan, zero_heuristic
from src.search import a_star


def test_goal_is_solvable():
    assert is_solvable(GOAL) is True


def test_unsolvable_example():
    # Swap two tiles in the goal (ignoring blank) -> unsolvable instance
    # Example: swap 1 and 2
    s = (2, 1, 3, 4, 5, 6, 7, 8, 0)
    assert is_solvable(s) is False


def test_heuristics_on_goal():
    g = PuzzleState(GOAL)
    assert hamming(g) == 0
    assert manhattan(g) == 0
    assert zero_heuristic(g) == 0


def test_heuristics_manhattan_dominates_on_sample():
    # A small scrambled state; Manhattan should be >= Hamming
    s = PuzzleState((2, 8, 3, 1, 6, 4, 7, 0, 5))
    assert manhattan(s) >= hamming(s)


def test_a_star_one_move():
    # One move from goal: blank left of 8, moving Right solves it.
    start = PuzzleState((1, 2, 3, 4, 5, 6, 7, 0, 8))

    # Using Hamming
    res_h = a_star(start, hamming)
    assert res_h.solved is True
    assert res_h.depth == 1
    assert res_h.expanded >= 1
    assert res_h.runtime_s >= 0.0

    # Using Manhattan
    res_m = a_star(start, manhattan)
    assert res_m.solved is True
    assert res_m.depth == 1
    assert res_m.expanded >= 1
    assert res_m.runtime_s >= 0.0


def test_a_star_zero_heuristic_matches_ucost():
    # Zero heuristic reduces to Uniform Cost Search; should still solve optimally.
    start = PuzzleState((1, 2, 3, 4, 5, 6, 0, 7, 8))  # depth=2
    res_z = a_star(start, zero_heuristic)
    assert res_z.solved is True
    assert res_z.depth == 2  # optimal depth
    # Manhattan should not exceed zero_heuristic expansions
    res_m = a_star(start, manhattan)
    assert res_m.depth == res_z.depth
    assert res_m.expanded <= res_z.expanded