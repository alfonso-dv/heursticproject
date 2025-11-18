from __future__ import annotations # Aktiviert zukünftige Typunterstützung, damit Klassen referenziert werden können, bevor sie definiert sind

from dataclasses import dataclass # Ermöglicht automatische Erstellen einer unveränderbaren PuzzleState-Klasse
from typing import Iterable, List, Tuple, Dict # Import für Typangaben, damit Code verständlicher bleibt

# ----- Board geometry ---------------------------------------------------------

# Definiert die Größe des Puzzles: 3x3 = 8-Puzzle
N = 3  # Wird später für Positionsberechnungen (Index → Zeile/Spalte) gebraucht

# Erstellt eine Tabelle, die jeden linearen Index (0–8) in (row, col) umrechnet
# i // N liefert die Zeile, i % N die Spalte (z. B. 5 → (1,2)); wird für Heuristiken und Moves benötigt
INDEX_TO_RC: Tuple[Tuple[int, int], ...] = tuple((i // N, i % N) for i in range(N * N))

# Zielzustand des Puzzles: 1–8 in Reihenfolge und die 0 (Leerfeld) unten rechts
# Wird in is_goal() verwendet, um zu prüfen, ob das Puzzle gelöst ist
GOAL: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Ordnet jedem Stein seine Zielposition zu (für Manhattan-Distanz wichtig)
# wichtig, damit Manhattan-Heuristik später schnell berechnen kann,
# wie weit jeder Stein noch von seiner richtigen Position entfernt ist
GOAL_POS: Dict[int, Tuple[int, int]] = {v: INDEX_TO_RC[i] for i, v in enumerate(GOAL)}


@dataclass(frozen=True) # Erstellt eine unveränderbare Klasse (wichtig für Sets in A*)
class PuzzleState:
    """
    Immutable 8-puzzle state.

    Attributes
    ----------
    tiles : tuple[int, ...]
        A length-9 tuple with a permutation of 0..8 (0 = blank).
    """
    # Speichert den aktuellen Puzzle-Zustand als Tuple, z. B. (2,8,3,1,6,4,7,0,5)
    tiles: Tuple[int, ...]  # Wird in allen Berechnungen (neighbors, Heuristiken, Goal-Check) verwendet

    # --- VALIDIERUNG DES ZUSTANDS -----------------------------------------------------
    def __post_init__(self) -> None: # Wird nach Objekt-Erstellung automatisch ausgeführt
        # Prüft, ob genau 9 Elemente vorhanden sind (3x3-Puzzle):
        if len(self.tiles) != N * N:
            raise ValueError(f"tiles must have length {N*N}, got {len(self.tiles)}")

        # Prüft, ob die Werte eine echte Permutation von 0–8 sind:
        # Wichtig, um ungültige Puzzle-Konfigurationen zu verhindern
        if set(self.tiles) != set(range(N * N)):
            raise ValueError("tiles must be a permutation of 0..8 (with 0 as blank)")

    # --- ZIELPRÜFUNG --------------------------------------------------------------
    def is_goal(self) -> bool: # Prüft, ob Puzzle gelöst ist
        """Return True iff this state equals the default GOAL configuration."""
        return self.tiles == GOAL # Vergleicht direkt mit dem global definierten Zielzustand

    # --- GENERIEREN VON NACHBARZUSTÄNDEN --------------------------------------------------
    def neighbors(self) -> List[Tuple["PuzzleState", str, int]]:
        # Findet die Position des Leerfeldes (0), da nur dieses bewegt wird
        """
        Return all valid successor states.

        Returns
        -------
        list of (next_state, action, cost)
            action ∈ {"Up","Down","Left","Right"} describes the blank's movement.
            cost is always 1 (uniform step cost).
        """
        zero_idx = self.tiles.index(0)

        # Wandelt Index in (row, col) um, ist hilfreich zur Bestimmung erlaubter Moves
        zr, zc = INDEX_TO_RC[zero_idx]

        succ: List[Tuple[PuzzleState, str, int]] = [] # Liste für alle erzeugten Nachbarzustände

        # Definiert mögliche Bewegungen des Leerfeldes: nach oben, unten, links, rechts
        # Jeder Eintrag: (DeltaRow, DeltaCol, Aktionsname)
        moves = [(-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right")]

        # Durchläuft jede mögliche Bewegung:
        for dr, dc, action in moves:
            nr, nc = zr + dr, zc + dc # Berechnet neue Position des Leerfeldes
            # Erzeugt neuen Zustand, indem das Leerfeld mit der Zielposition getauscht wird:
            if 0 <= nr < N and 0 <= nc < N:
                neighbor_idx = nr * N + nc # Berechnet Index der Position, mit der getauscht wird
                new_tiles = _swap(self.tiles, zero_idx, neighbor_idx) # Erzeugt neuen Zustand, indem das Leerfeld mit Zielposition getauscht wird
                succ.append((PuzzleState(new_tiles), action, 1)) #fügt neuen Zustand + ausgeführte Aktion + Kosten (immer 1) zur Liste hinzu

        return succ # Gibt Liste aller gültigen Nachbarn zurück

    # --- formatting -----------------------------------------------------------
    def pretty(self) -> str: # Gibt das Puzzle in 3 Zeilen formatiert zurück
        """Human-friendly 3x3 string representation (blank shown as a dot)."""
        rows = [] # Hier werden die drei Puzzle-Zeilen gespeichert
        # Durchläuft nacheinander 3 Puzzle-Zeilen (oben, Mitte, unten):
        for r in range(N):
            row_vals = [] # Speichert Werte der aktuellen Zeile
            # Durchläuft nacheinander die 3 Spalten der aktuellen Zeile (links, Mitte, rechts):
            for c in range(N):
                v = self.tiles[r * N + c] # Holt den Wert an Position (r,c)
                row_vals.append("." if v == 0 else str(v)) # Zeigt 0 als '.' für bessere Lesbarkeit
            # Erstellt eine formatierte Zeile wie "2 8 3":
            rows.append(" ".join(f"{x:>1}" for x in row_vals))
        return "\n".join(rows) # Gibt das vollständige 3x3-Board zurück

    # --- HILFSMETHODEN ----------------------------------------------
    def index_of(self, tile: int) -> int: # Gibt linearen Index eines Steins zurück
        """Return the linear index (0..8) of a given tile value."""
        return self.tiles.index(tile) # Wird z. B. von position_of genutzt

    def position_of(self, tile: int) -> Tuple[int, int]: # Gibt (row, col) eines Steins zurück
        """Return (row, col) of a given tile."""
        return INDEX_TO_RC[self.index_of(tile)] # Nutzt bestehende Index→Position-Mapping


# ----- INTERNER HELFER: SWAP --------------------------------------------------
# Führt einen Tausch im Puzzle durch:
def _swap(t: Tuple[int, ...], i: int, j: int) -> Tuple[int, ...]:
    """Return a new tuple with elements at indices i and j swapped."""
    if i == j: # Wenn gleicher Index → keine Änderung nötig
        return t
    lst = list(t) # Wandelt Tuple in Liste um (Tuples sind unveränderbar)
    lst[i], lst[j] = lst[j], lst[i] # Tauscht die beiden Werte
    return tuple(lst) # Gibt neues Tuple nach dem Tausch zurück


# ----- SELBSTTEST (optional, run as script) ------------------------------

if __name__ == "__main__": # Wird nur ausgeführt, wenn Datei direkt gestartet wird
    # Goal check
    s = PuzzleState(GOAL) # Erstellt Puzzle im Zielzustand
    assert s.is_goal() # Prüft, ob dieser Zustand korrekt erkannt wird

    # Bspzustand, der nur einen Zug vom Ziel entfernt ist: swap blank with '8'
    near_goal = PuzzleState((1, 2, 3, 4, 5, 6, 7, 0, 8))
    print("Near-goal state:\n", near_goal.pretty(), sep="") # Gibt Zustand formatiert aus
    # Zeigt alle möglichen nächsten Zustände durch Nachbarschaftsberechnung:
    for ns, a, c in near_goal.neighbors(): # Für jeden möglichen Move
        print(f"Action={a}, cost={c}\n{ns.pretty()}\n") # Zeigt neuen Zustand + ausgeführte Aktion