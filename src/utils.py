from __future__ import annotations

from typing import Tuple
import random

from .state import PuzzleState, GOAL

N = 3  # dimension


# ---------------------------- Solvability ------------------------------------
# prüft, ob ein Puzzle-Zustand lösbar ist (über Inversionszählungs)
def is_solvable(tiles: Tuple[int, ...]) -> bool:
    """
    Return True iff the given 3x3 configuration is solvable.

    Parameters
    ----------
    tiles : tuple[int, ...]
        A permutation of 0..8, where 0 denotes the blank.

    Notes
    -----
    For the 8-puzzle (odd width), the puzzle is solvable iff the inversion count
    (count of pairs i<j with tiles[i] > tiles[j], excluding the blank) is even.
    """
    # Validierung: Müssen 9 Felder sein, eine Permutation von 0-8
    if len(tiles) != N * N:
        raise ValueError(f"tiles must have length {N*N}, got {len(tiles)}")
    if set(tiles) != set(range(N * N)):
        raise ValueError("tiles must be a permutation of 0..8")
    # Inversionen zählen (größere Zahl vor kleinerer -> falsche Reihenfolge)
    inv = _inversion_count(tiles)
    # Beimm 8-Puzzle ist der Zustand lösbar, wenn die Nazahl der Inversionen gerade ist
    return (inv % 2) == 0

# Hilfsfunktion: zählt wie viele Zahlenpaare in der falsche Reihenfolge stehen
def _inversion_count(tiles: Tuple[int, ...]) -> int:
    """Count inversions in `tiles`, skipping the blank (0)."""
    arr = [v for v in tiles if v != 0] # 0 (Leerfeld) ignorieren
    inv = 0
    # Für jedes Zahlenpaar prüfen, ob eine größere Zahl vor einer kleineren kommt
    for i in range(len(arr)):
        ai = arr[i]
        for j in range(i + 1, len(arr)):
            if ai > arr[j]:
                inv += 1
    return inv


# ----------------------- Random solvable generator ---------------------------
# Erzeugt einen zufälligen, aber garantiert lösbaren Startzustand
def random_solvable_state(
    rng: random.Random,
    goal: Tuple[int, ...] = GOAL,
    max_attempts: int = 10_000,
) -> PuzzleState:
    """
    Generate a random solvable 8-puzzle state using the provided RNG.

    Parameters
    ----------
    rng : random.Random
        Seedable RNG you control (e.g., random.Random(42)) for reproducibility.
    goal : tuple[int, ...], optional
        Goal configuration to avoid returning; defaults to GOAL.
    max_attempts : int, optional
        Safety cap on shuffling attempts.

    Returns
    -------
    PuzzleState
        A solvable configuration. Tries not to equal `goal`, but if it fails to
        find a different solvable state within `max_attempts`, it returns GOAL.
    """
    base = list(range(N * N))  # Zahlen 0-8
    for _ in range(max_attempts):
        rng.shuffle(base)      # zufälliger shuffle
        tiles = tuple(base)

        # Nur zurückgeben, wenn der Zustand lösbar ist und nicht das Ziel selbst
        if is_solvable(tiles) and tiles != tuple(goal):
            return PuzzleState(tiles)
    # Fallback: wenn nach vielen Versuchen nichts gefunden wird
    return PuzzleState(tuple(goal))


# ------------------------------ Small helpers --------------------------------
# tauscht zwei Positionen in einem Tuple (kleine Utility-Funktion)
def tuple_swap(t: Tuple[int, ...], i: int, j: int) -> Tuple[int, ...]:
    """
    Return a new tuple with elements at indices i and j swapped.
    (Public version of the small helper; convenient for tests/experiments.)
    """
    if i == j:
        return t
    lst = list(t)   # Tuple --> Liste
    lst[i], lst[j] = lst[j], lst[i]
    return tuple(lst) # Liste --> Tuple zurück


# ------------------------------- Self-test -----------------------------------

if __name__ == "__main__":
    rng = random.Random(123)
    # Quick checks
    assert is_solvable(GOAL)
    s = random_solvable_state(rng)
    print("Sample random solvable state:\n", PuzzleState(s.tiles).pretty(), sep="")