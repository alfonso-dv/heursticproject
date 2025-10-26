from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict

# ----- Board geometry ---------------------------------------------------------

N = 3  # 3x3 board

# Mapping from linear index -> (row, col)
INDEX_TO_RC: Tuple[Tuple[int, int], ...] = tuple((i // N, i % N) for i in range(N * N))

# Default goal configuration (1..8, blank=0 at bottom-right)
GOAL: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# For Manhattan and quick lookups: tile value -> (goal_row, goal_col)
# Note: tile 0 (blank) is excluded from heuristic calculations but mapped for completeness.
GOAL_POS: Dict[int, Tuple[int, int]] = {v: INDEX_TO_RC[i] for i, v in enumerate(GOAL)}


@dataclass(frozen=True)
class PuzzleState:
    """
    Immutable 8-puzzle state.

    Attributes
    ----------
    tiles : tuple[int, ...]
        A length-9 tuple with a permutation of 0..8 (0 = blank).
    """
    tiles: Tuple[int, ...]  # e.g., (1,2,3,4,5,6,7,8,0)

    # --- basic validation -----------------------------------------------------
    def __post_init__(self) -> None:
        if len(self.tiles) != N * N:
            raise ValueError(f"tiles must have length {N*N}, got {len(self.tiles)}")
        if set(self.tiles) != set(range(N * N)):
            raise ValueError("tiles must be a permutation of 0..8 (with 0 as blank)")

    # --- queries --------------------------------------------------------------
    def is_goal(self) -> bool:
        """Return True iff this state equals the default GOAL configuration."""
        return self.tiles == GOAL

    # --- neighbor generation --------------------------------------------------
    def neighbors(self) -> List[Tuple["PuzzleState", str, int]]:
        """
        Return all valid successor states.

        Returns
        -------
        list of (next_state, action, cost)
            action ∈ {"Up","Down","Left","Right"} describes the blank's movement.
            cost is always 1 (uniform step cost).
        """
        zero_idx = self.tiles.index(0)
        zr, zc = INDEX_TO_RC[zero_idx]

        succ: List[Tuple[PuzzleState, str, int]] = []

        # Possible moves: (dr, dc, action)
        moves = [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]

        for dr, dc, action in moves:
            nr, nc = zr + dr, zc + dc
            if 0 <= nr < N and 0 <= nc < N:
                neighbor_idx = nr * N + nc
                new_tiles = _swap(self.tiles, zero_idx, neighbor_idx)
                succ.append((PuzzleState(new_tiles), action, 1))

        return succ

    # --- formatting -----------------------------------------------------------
    def pretty(self) -> str:
        """Human-friendly 3x3 string representation (blank shown as a dot)."""
        rows = []
        for r in range(N):
            row_vals = []
            for c in range(N):
                v = self.tiles[r * N + c]
                row_vals.append("." if v == 0 else str(v))
            rows.append(" ".join(f"{x:>1}" for x in row_vals))
        return "\n".join(rows)

    # --- convenience accessors ----------------------------------------------
    def index_of(self, tile: int) -> int:
        """Return the linear index (0..8) of a given tile value."""
        return self.tiles.index(tile)

    def position_of(self, tile: int) -> Tuple[int, int]:
        """Return (row, col) of a given tile."""
        return INDEX_TO_RC[self.index_of(tile)]


# ----- tiny internal helpers --------------------------------------------------

def _swap(t: Tuple[int, ...], i: int, j: int) -> Tuple[int, ...]:
    """Return a new tuple with elements at indices i and j swapped."""
    if i == j:
        return t
    lst = list(t)
    lst[i], lst[j] = lst[j], lst[i]
    return tuple(lst)


# ----- quick self-test (optional, run as script) ------------------------------

if __name__ == "__main__":
    # Goal check
    s = PuzzleState(GOAL)
    assert s.is_goal()

    # One move from goal: swap blank with '8'
    near_goal = PuzzleState((1, 2, 3, 4, 5, 6, 7, 0, 8))
    print("Near-goal state:\n", near_goal.pretty(), sep="")
    for ns, a, c in near_goal.neighbors():
        print(f"Action={a}, cost={c}\n{ns.pretty()}\n")