from __future__ import annotations

import time
import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .state import PuzzleState, GOAL


# ---------------------------- Result structure -------------------------------

@dataclass
class SearchResult:
    solved: bool
    depth: int
    expanded: int
    runtime_s: float
    heuristic: str
    start_state: Tuple[int, ...]
    path: Optional[List[PuzzleState]] = field(default=None)


# ------------------------------- A* search -----------------------------------

def a_star(start: PuzzleState, h: Callable[[PuzzleState], int]) -> SearchResult:
    """
    Perform A* search using the given heuristic function.

    Parameters
    ----------
    start : PuzzleState
        Starting configuration.
    h : callable
        Heuristic function h(state) -> int (e.g., hamming or manhattan).

    Returns
    -------
    SearchResult
        Includes whether solved, depth, nodes expanded, runtime, and heuristic name.
    """
    t0 = time.perf_counter()
    heuristic_name = h.__name__.capitalize()

    open_heap: List[Tuple[int, int, int, PuzzleState]] = []  # (f, g, tie, state)
    g_score: Dict[PuzzleState, int] = {start: 0}
    came_from: Dict[PuzzleState, Optional[PuzzleState]] = {start: None}

    expanded_nodes = 0
    tie_counter = 0

    # initial push
    f0 = h(start)
    heapq.heappush(open_heap, (f0, 0, tie_counter, start))
    closed: set[PuzzleState] = set()

    # main loop
    while open_heap:
        f, g, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expanded_nodes += 1  # measure memory effort

        if current.is_goal():
            t1 = time.perf_counter()
            path = _reconstruct_path(came_from, current)
            return SearchResult(
                solved=True,
                depth=len(path) - 1,
                expanded=expanded_nodes,
                runtime_s=t1 - t0,
                heuristic=heuristic_name,
                start_state=start.tiles,
                path=path,
            )

        for neighbor, action, cost in current.neighbors():
            tentative_g = g + cost
            if neighbor in closed:
                continue

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                tie_counter += 1
                f_val = tentative_g + h(neighbor)
                heapq.heappush(open_heap, (f_val, tentative_g, tie_counter, neighbor))

    # unsolved case
    t1 = time.perf_counter()
    return SearchResult(
        solved=False,
        depth=0,
        expanded=expanded_nodes,
        runtime_s=t1 - t0,
        heuristic=heuristic_name,
        start_state=start.tiles,
        path=None,
    )


# ------------------------------ Path recovery --------------------------------

def _reconstruct_path(
    came_from: Dict[PuzzleState, Optional[PuzzleState]],
    goal_state: PuzzleState,
) -> List[PuzzleState]:
    """
    Reconstruct the path from start to the given goal_state using came_from mapping.
    """
    path: List[PuzzleState] = [goal_state]
    while came_from[path[-1]] is not None:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


# ------------------------------- Self-test -----------------------------------

if __name__ == "__main__":
    from .heuristics import manhattan, hamming

    start = PuzzleState((1, 2, 3, 4, 5, 6, 0, 7, 8))
    for fn in (hamming, manhattan):
        print(f"Running A* with {fn.__name__}")
        res = a_star(start, fn)
        print(f"Solved={res.solved}, depth={res.depth}, expanded={res.expanded}, time={res.runtime_s:.4f}s")